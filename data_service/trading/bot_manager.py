"""
Bot manager: creates, tracks, and restores TradingBot instances.
Initialises shared Redis and Binance clients from environment variables.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any

import redis
from binance.client import Client as BinanceClient

from .bot_engine import BotConfig, BotState, BotStatus, TradingBot

logger = logging.getLogger(__name__)


class BotManager:
    """Manages the lifecycle of all TradingBot instances."""

    def __init__(self):
        # Redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=True,
        )

        # Binance
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_SECRET_KEY", "")
        self.binance = BinanceClient(api_key, api_secret)

        # In-memory registry: bot_id -> TradingBot
        self._bots: Dict[str, TradingBot] = {}
        # asyncio tasks: bot_id -> Task
        self._tasks: Dict[str, asyncio.Task] = {}

        logger.info("BotManager initialised (redis=%s:%s)", redis_host, redis_port)

    # ------------------------------------------------------------------
    # Start / stop / pause / resume
    # ------------------------------------------------------------------

    async def start_bot(self, config: BotConfig) -> Dict[str, Any]:
        """Create a new TradingBot and launch its async task."""
        if config.bot_id in self._bots:
            return {"error": f"Bot {config.bot_id} is already running"}

        state = BotState(cash=config.capital, peak_equity=config.capital)
        bot = TradingBot(config, state, self.redis, self.binance)

        self._bots[config.bot_id] = bot
        task = asyncio.create_task(bot.run(), name=f"bot-{config.bot_id}")
        self._tasks[config.bot_id] = task

        # Register in global bot index
        self._add_to_index(config.bot_id)

        return bot.get_status()

    async def stop_bot(self, bot_id: str, close_position: bool = True) -> Dict[str, Any]:
        """Gracefully stop a bot. Optionally close its open position first."""
        bot = self._bots.get(bot_id)
        if not bot:
            return {"error": f"Bot {bot_id} not found"}

        # Close position before stopping (long or short)
        if close_position and bot.state.position_qty != 0:
            try:
                price = await self._get_current_price(bot.config.symbol)
                await bot._close_position(price)
            except Exception as e:
                logger.warning(f"Failed to close position for {bot_id}: {e}")

        bot.stop()

        # Cancel the asyncio task
        task = self._tasks.pop(bot_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        bot.state.status = BotStatus.STOPPED
        bot._save_state()
        self._bots.pop(bot_id, None)
        return bot.get_status()

    async def pause_bot(self, bot_id: str) -> Dict[str, Any]:
        bot = self._bots.get(bot_id)
        if not bot:
            return {"error": f"Bot {bot_id} not found"}
        bot.pause()
        return bot.get_status()

    async def resume_bot(self, bot_id: str) -> Dict[str, Any]:
        bot = self._bots.get(bot_id)
        if not bot:
            return {"error": f"Bot {bot_id} not found"}
        bot.resume()
        return bot.get_status()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_bot_status(self, bot_id: str) -> Dict[str, Any]:
        """Get live status for a running bot, or load from Redis if stopped."""
        bot = self._bots.get(bot_id)
        if bot:
            price = await self._get_current_price(bot.config.symbol)
            return bot.get_status(current_price=price)

        # Try loading from Redis
        raw_cfg = self.redis.get(f"bot:config:{bot_id}")
        raw_state = self.redis.get(f"bot:state:{bot_id}")
        if raw_cfg and raw_state:
            cfg = BotConfig.model_validate_json(raw_cfg)
            state = BotState.model_validate_json(raw_state)
            pos_side = "FLAT"
            if state.position_qty > 0:
                pos_side = "LONG"
            elif state.position_qty < 0:
                pos_side = "SHORT"
            return {
                "bot_id": cfg.bot_id,
                "strategy": cfg.strategy_name,
                "symbol": cfg.symbol,
                "interval": cfg.interval,
                "status": state.status,
                "position_side": pos_side,
                "position_qty": abs(state.position_qty),
                "position_entry": state.position_entry,
                "leverage": cfg.leverage,
                "margin_type": cfg.margin_type,
                "cash": round(state.cash, 4),
                "equity": round(state.cash, 4),  # no live price available
                "unrealized_pnl": 0.0,
                "peak_equity": round(state.peak_equity, 4),
                "last_signal": state.last_signal,
                "total_trades": state.total_trades,
                "capital": cfg.capital,
                "max_drawdown": cfg.max_drawdown,
                "daily_loss_limit": cfg.daily_loss_limit,
                "stop_loss": cfg.stop_loss,
                "take_profit": cfg.take_profit,
                "created_at": state.created_at,
                "last_update": state.last_update,
                "error_message": state.error_message,
            }

        return {"error": f"Bot {bot_id} not found"}

    def get_bot_trades(self, bot_id: str) -> List[Dict]:
        """Get trade history from Redis."""
        raw = self.redis.lrange(f"bot:trades:{bot_id}", 0, -1)
        return [json.loads(t) for t in raw]

    def get_bot_equity(self, bot_id: str) -> List[Dict]:
        """Get equity curve from Redis."""
        raw = self.redis.lrange(f"bot:equity:{bot_id}", 0, -1)
        return [json.loads(e) for e in raw]

    def get_bot_log(self, bot_id: str, limit: int = 100) -> List[Dict]:
        """Get recent log entries from Redis."""
        raw = self.redis.lrange(f"bot:log:{bot_id}", -limit, -1)
        return [json.loads(l) for l in raw]

    async def list_bots(self) -> List[Dict[str, Any]]:
        """List all known bots (running + historical)."""
        bot_ids = self._get_index()
        results = []
        for bid in bot_ids:
            status = await self.get_bot_status(bid)
            if "error" not in status:
                results.append(status)
        return results

    async def get_account_info(self) -> Dict[str, Any]:
        """Get Binance Futures account balances and positions."""
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None, self.binance.futures_account
            )
            # Non-zero balances
            balances = [
                {
                    "asset": b["asset"],
                    "balance": float(b["walletBalance"]),
                    "available": float(b["availableBalance"]),
                    "unrealized_pnl": float(b.get("crossUnPnl", 0)),
                }
                for b in account.get("assets", [])
                if float(b["walletBalance"]) > 0
            ]
            # Open positions
            positions = [
                {
                    "symbol": p["symbol"],
                    "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                    "qty": abs(float(p["positionAmt"])),
                    "entry_price": float(p["entryPrice"]),
                    "unrealized_pnl": float(p["unrealizedProfit"]),
                    "leverage": int(p["leverage"]),
                    "margin_type": "ISOLATED" if p.get("isolated") else "CROSSED",
                }
                for p in account.get("positions", [])
                if float(p["positionAmt"]) != 0
            ]
            return {"balances": balances, "positions": positions}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Restore on startup
    # ------------------------------------------------------------------

    async def restore_bots(self):
        """Scan Redis for bots that were running and restart them."""
        bot_ids = self._get_index()
        restored = 0
        for bid in bot_ids:
            raw_state = self.redis.get(f"bot:state:{bid}")
            raw_cfg = self.redis.get(f"bot:config:{bid}")
            if not raw_state or not raw_cfg:
                continue
            state = BotState.model_validate_json(raw_state)
            if state.status not in (BotStatus.RUNNING, BotStatus.PAUSED):
                continue

            config = BotConfig.model_validate_json(raw_cfg)
            bot = TradingBot(config, state, self.redis, self.binance)
            self._bots[bid] = bot
            task = asyncio.create_task(bot.run(), name=f"bot-{bid}")
            self._tasks[bid] = task
            restored += 1
            logger.info(f"Restored bot {bid} ({config.strategy_name} on {config.symbol})")

        logger.info(f"Restored {restored} bot(s) from Redis")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_to_index(self, bot_id: str):
        """Add bot_id to the global bot index in Redis."""
        raw = self.redis.get("bot:index")
        ids = json.loads(raw) if raw else []
        if bot_id not in ids:
            ids.append(bot_id)
            self.redis.set("bot:index", json.dumps(ids))

    def _get_index(self) -> List[str]:
        raw = self.redis.get("bot:index")
        return json.loads(raw) if raw else []

    async def _get_current_price(self, symbol: str) -> float:
        """Get latest futures mark price from Binance."""
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.binance.futures_symbol_ticker(symbol=symbol)
            )
            return float(ticker["price"])
        except Exception:
            return 0.0
