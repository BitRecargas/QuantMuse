"""
Live FUTURES trading bot engine.
Runs as an async background task inside the FastAPI service.
Reuses the exact same signal generators from crypto_backtest.py.

Position convention: position_qty > 0 = LONG, < 0 = SHORT, 0 = flat.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import redis
from pydantic import BaseModel, Field
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

from ..backtest.crypto_backtest import fetch_binance_ohlcv, STRATEGY_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BotStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class BotConfig(BaseModel):
    bot_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    strategy_name: str = "Momentum"
    symbol: str = "ETHUSDT"
    interval: str = "4h"
    strategy_params: Dict[str, Any] = Field(default_factory=lambda: {"lookback": 168})
    capital: float = 100.0
    leverage: int = 5
    margin_type: str = "ISOLATED"  # ISOLATED or CROSSED
    max_drawdown: float = 0.15  # 15%
    daily_loss_limit: float = 0.05  # 5%
    candle_lookback: int = 300  # candles to fetch for signal computation
    stop_loss: float = 0.0     # 0 = disabled, e.g. 0.01 = 1% SL from entry
    take_profit: float = 0.0   # 0 = disabled, e.g. 0.03 = 3% TP from entry


class BotState(BaseModel):
    status: str = BotStatus.RUNNING
    position_qty: float = 0.0       # positive = long, negative = short, 0 = flat
    position_entry: float = 0.0
    cash: float = 0.0               # margin wallet balance allocated to this bot
    last_signal: int = 0
    peak_equity: float = 0.0
    daily_open_equity: float = 0.0
    daily_date: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_update: str = ""
    error_message: str = ""
    total_trades: int = 0


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

_INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
    "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
}


def _seconds_until_next_candle(interval: str, buffer: int = 10) -> float:
    """Seconds until the next candle boundary (UTC-aligned) + buffer."""
    secs = _INTERVAL_SECONDS.get(interval, 3600)
    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    elapsed = epoch % secs
    remaining = secs - elapsed + buffer
    return max(remaining, 1)


# ---------------------------------------------------------------------------
# TradingBot  (FUTURES)
# ---------------------------------------------------------------------------

class TradingBot:
    """Async live FUTURES trading bot for a single strategy + symbol."""

    def __init__(
        self,
        config: BotConfig,
        state: Optional[BotState],
        redis_client: redis.Redis,
        binance_client: BinanceClient,
    ):
        self.config = config
        self.state = state or BotState(cash=config.capital, peak_equity=config.capital)
        self.redis = redis_client
        self.binance = binance_client
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._symbol_filters: Optional[Dict] = None  # LOT_SIZE / PRICE_FILTER from exchange info

    # ---- lifecycle ----

    async def run(self):
        """Main loop: sleep -> fetch -> compute -> trade -> persist."""
        self.state.status = BotStatus.RUNNING
        self._log(f"Bot started: {self.config.strategy_name} on {self.config.symbol} "
                   f"({self.config.interval}, {self.config.leverage}x leverage, {self.config.margin_type})")
        self._save_state()

        try:
            # Configure futures: margin type + leverage
            await self._configure_futures()

            # Pre-fetch symbol filters for quantity precision
            await self._load_symbol_filters()

            while not self._stop_event.is_set():
                if self.state.status == BotStatus.PAUSED:
                    await self._interruptible_sleep(30)
                    continue

                # 1. Wait for next candle
                await self._wait_for_next_candle()
                if self._stop_event.is_set():
                    break

                # 2. Fetch candles
                df = await self._fetch_candles()
                if df is None or df.empty:
                    self._log("Failed to fetch candles, skipping cycle")
                    continue

                # 3. Compute signal
                signal = self._compute_signal(df)
                current_price = float(df["close"].iloc[-1])
                bar_high = float(df["high"].iloc[-1])
                bar_low = float(df["low"].iloc[-1])

                # 4. Check SL/TP using the candle's high/low (matches backtest logic)
                sltp_closed = await self._check_sltp(bar_high, bar_low, current_price)

                # 5. Daily tracking reset
                self._update_daily_tracking(current_price)

                # 6. Check risk limits
                if self._check_risk_limits(current_price):
                    break  # risk limit breached, bot stopped

                # 7. Execute trades if signal changed (skip if SL/TP just closed position)
                if not sltp_closed and signal != self.state.last_signal:
                    await self._process_signal(signal, current_price)

                self.state.last_signal = 0 if sltp_closed else signal
                self.state.last_update = datetime.now(timezone.utc).isoformat()

                # 8. Save equity snapshot
                equity = self._current_equity(current_price)
                self._append_equity(equity)
                self._save_state()

        except asyncio.CancelledError:
            self._log("Bot task cancelled")
        except Exception as e:
            self.state.status = BotStatus.ERROR
            self.state.error_message = str(e)
            self._log(f"Bot error: {e}")
            logger.exception(f"Bot {self.config.bot_id} crashed")
        finally:
            self._save_state()

    def stop(self):
        """Signal the bot to stop."""
        self._stop_event.set()
        self.state.status = BotStatus.STOPPED
        self._log("Stop requested")

    def pause(self):
        """Pause trading (bot stays alive but skips cycles)."""
        self.state.status = BotStatus.PAUSED
        self._log("Paused")
        self._save_state()

    def resume(self):
        """Resume from paused state."""
        self.state.status = BotStatus.RUNNING
        self._log("Resumed")
        self._save_state()

    # ---- futures configuration ----

    async def _configure_futures(self):
        """Set margin type and leverage for this symbol on Binance Futures."""
        loop = asyncio.get_event_loop()

        # Set margin type (ignore error if already set)
        try:
            await loop.run_in_executor(
                None,
                lambda: self.binance.futures_change_margin_type(
                    symbol=self.config.symbol,
                    marginType=self.config.margin_type,
                ),
            )
            self._log(f"Margin type set to {self.config.margin_type}")
        except BinanceAPIException as e:
            if e.code == -4046:  # "No need to change margin type"
                pass
            else:
                self._log(f"Warning: margin type change failed: {e}")

        # Set leverage
        try:
            await loop.run_in_executor(
                None,
                lambda: self.binance.futures_change_leverage(
                    symbol=self.config.symbol,
                    leverage=self.config.leverage,
                ),
            )
            self._log(f"Leverage set to {self.config.leverage}x")
        except BinanceAPIException as e:
            self._log(f"Warning: leverage change failed: {e}")

    async def _load_symbol_filters(self):
        """Load LOT_SIZE and other filters from futures exchange info."""
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.binance.futures_exchange_info
            )
            for sym_info in info.get("symbols", []):
                if sym_info["symbol"] == self.config.symbol:
                    self._symbol_filters = {
                        f["filterType"]: f for f in sym_info.get("filters", [])
                    }
                    break
        except Exception as e:
            self._log(f"Warning: could not load symbol filters: {e}")

    # ---- candle waiting ----

    async def _wait_for_next_candle(self):
        """Sleep until next candle boundary in interruptible 30s chunks."""
        remaining = _seconds_until_next_candle(self.config.interval)
        self._log(f"Waiting {remaining:.0f}s for next {self.config.interval} candle")
        await self._interruptible_sleep(remaining)

    async def _interruptible_sleep(self, total_seconds: float):
        """Sleep in 30s chunks so we can respond to stop quickly."""
        chunk = 30
        slept = 0
        while slept < total_seconds and not self._stop_event.is_set():
            to_sleep = min(chunk, total_seconds - slept)
            await asyncio.sleep(to_sleep)
            slept += to_sleep

    # ---- data & signals ----

    async def _fetch_candles(self):
        """Fetch candles with retry."""
        for attempt in range(3):
            try:
                df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: fetch_binance_ohlcv(
                        self.config.symbol,
                        interval=self.config.interval,
                        limit=self.config.candle_lookback,
                    ),
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                self._log(f"Fetch attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(5 * (attempt + 1))
        return None

    def _compute_signal(self, df) -> int:
        """Compute strategy signal from candles. Returns last signal value."""
        signal_func = STRATEGY_MAP.get(self.config.strategy_name)
        if signal_func is None:
            signal_func = STRATEGY_MAP.get("Momentum")
        signals = signal_func(df, **self.config.strategy_params)
        return int(signals.iloc[-1])

    # ---- SL/TP (matches backtest logic exactly) ----

    async def _check_sltp(self, bar_high: float, bar_low: float, current_price: float) -> bool:
        """
        Check stop-loss / take-profit using the candle's high/low.
        Mirrors the exact same logic as crypto_backtest.py and strategy_sweep.py.
        Returns True if position was closed by SL/TP.
        """
        sl = self.config.stop_loss
        tp = self.config.take_profit
        if (sl <= 0 and tp <= 0) or self.state.position_qty == 0:
            return False

        pos = self.state.position_qty
        entry = self.state.position_entry

        if pos > 0:  # LONG
            sl_price = entry * (1 - sl) if sl > 0 else 0
            tp_price = entry * (1 + tp) if tp > 0 else float('inf')
            sl_hit = sl > 0 and bar_low <= sl_price
            tp_hit = tp > 0 and bar_high >= tp_price
        else:  # SHORT
            sl_price = entry * (1 + sl) if sl > 0 else float('inf')
            tp_price = entry * (1 - tp) if tp > 0 else 0
            sl_hit = sl > 0 and bar_high >= sl_price
            tp_hit = tp > 0 and bar_low <= tp_price

        if not sl_hit and not tp_hit:
            return False

        # If both trigger same bar, assume SL hit first (conservative — matches backtest)
        if sl_hit:
            reason = "STOP-LOSS"
            target_price = sl_price
        else:
            reason = "TAKE-PROFIT"
            target_price = tp_price

        side_str = "LONG" if pos > 0 else "SHORT"
        self._log(f"{reason} triggered for {side_str} @ entry={entry:.2f}, "
                  f"target={target_price:.2f}, bar_high={bar_high:.2f}, bar_low={bar_low:.2f}")

        # Close the position via market order (market price, not exact SL/TP price)
        await self._close_position(current_price)
        return True

    # ---- futures order execution ----

    async def _process_signal(self, signal: int, current_price: float):
        """
        Process signal change with full long/short transitions.
          signal  1 → go long   (close short if any, then open long)
          signal -1 → go short  (close long if any, then open short)
          signal  0 → go flat   (close whatever is open)
        """
        pos = self.state.position_qty  # >0 long, <0 short, 0 flat

        if signal == 1:
            if pos < 0:
                # Close short first
                await self._close_position(current_price)
            if self.state.position_qty == 0:
                await self._open_long(current_price)

        elif signal == -1:
            if pos > 0:
                # Close long first
                await self._close_position(current_price)
            if self.state.position_qty == 0:
                await self._open_short(current_price)

        elif signal == 0:
            if pos != 0:
                await self._close_position(current_price)

    async def _open_long(self, current_price: float):
        """Open a LONG position using futures market order."""
        notional = self.state.cash * 0.95 * self.config.leverage
        qty = self._round_qty(notional / current_price)

        if qty <= 0:
            self._log(f"Open LONG skipped: qty {qty} <= 0")
            return

        try:
            order = await self._futures_market_order("BUY", qty)
            fill_price, fill_qty, commission = self._parse_fill(order)
            self.state.position_qty = fill_qty  # positive = long
            self.state.position_entry = fill_price
            self.state.total_trades += 1

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "side": "OPEN_LONG",
                "qty": fill_qty,
                "price": fill_price,
                "commission": commission,
                "leverage": self.config.leverage,
                "order_id": order.get("orderId", ""),
            }
            self._append_trade(trade)
            self._log(f"OPEN LONG {fill_qty} @ {fill_price} ({self.config.leverage}x, fee={commission:.4f})")

        except BinanceAPIException as e:
            self._handle_order_error("Open LONG", e)

    async def _open_short(self, current_price: float):
        """Open a SHORT position using futures market order."""
        notional = self.state.cash * 0.95 * self.config.leverage
        qty = self._round_qty(notional / current_price)

        if qty <= 0:
            self._log(f"Open SHORT skipped: qty {qty} <= 0")
            return

        try:
            order = await self._futures_market_order("SELL", qty)
            fill_price, fill_qty, commission = self._parse_fill(order)
            self.state.position_qty = -fill_qty  # negative = short
            self.state.position_entry = fill_price
            self.state.total_trades += 1

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "side": "OPEN_SHORT",
                "qty": fill_qty,
                "price": fill_price,
                "commission": commission,
                "leverage": self.config.leverage,
                "order_id": order.get("orderId", ""),
            }
            self._append_trade(trade)
            self._log(f"OPEN SHORT {fill_qty} @ {fill_price} ({self.config.leverage}x, fee={commission:.4f})")

        except BinanceAPIException as e:
            self._handle_order_error("Open SHORT", e)

    async def _close_position(self, current_price: float):
        """Close the current position (long or short)."""
        pos = self.state.position_qty
        if pos == 0:
            return

        abs_qty = self._round_qty(abs(pos))
        if abs_qty <= 0:
            return

        # To close a LONG → SELL; to close a SHORT → BUY
        side = "SELL" if pos > 0 else "BUY"
        label = "CLOSE LONG" if pos > 0 else "CLOSE SHORT"

        try:
            order = await self._futures_market_order(side, abs_qty)
            fill_price, fill_qty, commission = self._parse_fill(order)

            # P&L calculation
            if pos > 0:
                pnl = (fill_price - self.state.position_entry) * fill_qty - commission
            else:
                pnl = (self.state.position_entry - fill_price) * fill_qty - commission

            self.state.cash += pnl
            self.state.position_qty = 0.0
            self.state.position_entry = 0.0
            self.state.total_trades += 1

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "side": label.replace(" ", "_"),
                "qty": fill_qty,
                "price": fill_price,
                "commission": commission,
                "pnl": round(pnl, 4),
                "order_id": order.get("orderId", ""),
            }
            self._append_trade(trade)
            self._log(f"{label} {fill_qty} @ {fill_price} pnl={pnl:.4f} (fee={commission:.4f})")

        except BinanceAPIException as e:
            self._handle_order_error(label, e)

    async def _futures_market_order(self, side: str, quantity: float) -> Dict:
        """Place a futures market order."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.binance.futures_create_order(
                symbol=self.config.symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            ),
        )

    def _parse_fill(self, order: Dict) -> tuple:
        """Extract average fill price, total qty, and total commission from futures order response."""
        # Futures orders don't always have 'fills' - use avgPrice + executedQty
        avg_price = float(order.get("avgPrice", 0))
        exec_qty = float(order.get("executedQty", 0))

        # Try fills array if available (more accurate)
        fills = order.get("fills", [])
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
            total_comm = sum(float(f["commission"]) for f in fills)
            avg_price = total_cost / total_qty if total_qty > 0 else avg_price
            return avg_price, total_qty, total_comm

        # Estimate commission (0.04% taker fee for futures)
        commission = exec_qty * avg_price * 0.0004
        return avg_price, exec_qty, commission

    def _round_qty(self, qty: float) -> float:
        """Round quantity to the symbol's step size from futures exchange info."""
        if not self._symbol_filters:
            return round(qty, 3)
        lot_filter = self._symbol_filters.get("LOT_SIZE")
        if lot_filter:
            step = float(lot_filter["stepSize"])
            if step > 0:
                precision = max(0, len(str(step).rstrip("0").split(".")[-1]))
                qty = round(qty - (qty % step), precision)
        return qty

    def _handle_order_error(self, action: str, e: BinanceAPIException):
        """Handle order errors: log and optionally stop bot on permanent errors."""
        self._log(f"{action} failed: {e}")
        if self._is_permanent_error(e):
            self.state.status = BotStatus.ERROR
            self.state.error_message = f"{action} failed: {e}"
            self._stop_event.set()

    @staticmethod
    def _is_permanent_error(e: BinanceAPIException) -> bool:
        """Check if a Binance error is permanent (not worth retrying)."""
        permanent_codes = {-2010, -1013, -1021, -2015, -4003, -4015}
        # -2010 insufficient balance, -1013 invalid qty, -1021 timestamp
        # -2015 invalid api key, -4003 qty less than zero, -4015 invalid leverage
        return e.code in permanent_codes

    # ---- risk management ----

    def _current_equity(self, current_price: float) -> float:
        """Calculate equity: cash + unrealized PnL."""
        if self.state.position_qty == 0:
            return self.state.cash
        pnl = self._unrealized_pnl(current_price)
        return self.state.cash + pnl

    def _unrealized_pnl(self, current_price: float) -> float:
        """Unrealized PnL for current position."""
        pos = self.state.position_qty
        entry = self.state.position_entry
        if pos > 0:
            return (current_price - entry) * pos
        elif pos < 0:
            return (entry - current_price) * abs(pos)
        return 0.0

    def _update_daily_tracking(self, current_price: float):
        """Reset daily tracking at date boundary."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.daily_date != today:
            self.state.daily_date = today
            self.state.daily_open_equity = self._current_equity(current_price)

    def _check_risk_limits(self, current_price: float) -> bool:
        """Check max drawdown and daily loss limit. Returns True if breached."""
        equity = self._current_equity(current_price)

        # Update peak
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        # Max drawdown check
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - equity) / self.state.peak_equity
            if drawdown >= self.config.max_drawdown:
                self._log(f"MAX DRAWDOWN BREACHED: {drawdown:.2%} >= {self.config.max_drawdown:.2%}")
                self.state.status = BotStatus.STOPPED
                self.state.error_message = f"Max drawdown {drawdown:.2%} breached"
                self._stop_event.set()
                return True

        # Daily loss limit check
        if self.state.daily_open_equity > 0:
            daily_loss = (self.state.daily_open_equity - equity) / self.state.daily_open_equity
            if daily_loss >= self.config.daily_loss_limit:
                self._log(f"DAILY LOSS LIMIT BREACHED: {daily_loss:.2%} >= {self.config.daily_loss_limit:.2%}")
                self.state.status = BotStatus.STOPPED
                self.state.error_message = f"Daily loss limit {daily_loss:.2%} breached"
                self._stop_event.set()
                return True

        return False

    # ---- Redis persistence ----

    def _rk(self, suffix: str) -> str:
        """Redis key helper."""
        return f"bot:{suffix}:{self.config.bot_id}"

    def _save_state(self):
        """Persist bot state to Redis."""
        try:
            self.redis.set(self._rk("config"), self.config.model_dump_json())
            self.redis.set(self._rk("state"), self.state.model_dump_json())
        except Exception as e:
            logger.error(f"Redis save failed for {self.config.bot_id}: {e}")

    def _append_trade(self, trade: Dict):
        """Append a trade record to Redis list."""
        try:
            self.redis.rpush(self._rk("trades"), json.dumps(trade, default=str))
        except Exception as e:
            logger.error(f"Redis trade append failed: {e}")

    def _append_equity(self, equity: float):
        """Append an equity snapshot to Redis list."""
        try:
            snap = {"time": datetime.now(timezone.utc).isoformat(), "equity": round(equity, 4)}
            self.redis.rpush(self._rk("equity"), json.dumps(snap))
            # Cap at 10000 entries
            self.redis.ltrim(self._rk("equity"), -10000, -1)
        except Exception as e:
            logger.error(f"Redis equity append failed: {e}")

    def _log(self, message: str):
        """Append log message to Redis + Python logger."""
        logger.info(f"[{self.config.bot_id}] {message}")
        try:
            entry = {"time": datetime.now(timezone.utc).isoformat(), "msg": message}
            self.redis.rpush(self._rk("log"), json.dumps(entry))
            self.redis.ltrim(self._rk("log"), -500, -1)
        except Exception:
            pass

    # ---- public helpers ----

    def get_status(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """Return a status snapshot for the API."""
        equity = self.state.cash
        unrealized_pnl = 0.0
        pos_side = "FLAT"
        if self.state.position_qty != 0 and current_price:
            unrealized_pnl = self._unrealized_pnl(current_price)
            equity = self.state.cash + unrealized_pnl
            pos_side = "LONG" if self.state.position_qty > 0 else "SHORT"

        return {
            "bot_id": self.config.bot_id,
            "strategy": self.config.strategy_name,
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "status": self.state.status,
            "position_side": pos_side,
            "position_qty": abs(self.state.position_qty),
            "position_entry": self.state.position_entry,
            "leverage": self.config.leverage,
            "margin_type": self.config.margin_type,
            "cash": round(self.state.cash, 4),
            "equity": round(equity, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "peak_equity": round(self.state.peak_equity, 4),
            "last_signal": self.state.last_signal,
            "total_trades": self.state.total_trades,
            "capital": self.config.capital,
            "max_drawdown": self.config.max_drawdown,
            "daily_loss_limit": self.config.daily_loss_limit,
            "stop_loss": self.config.stop_loss,
            "take_profit": self.config.take_profit,
            "created_at": self.state.created_at,
            "last_update": self.state.last_update,
            "error_message": self.state.error_message,
        }
