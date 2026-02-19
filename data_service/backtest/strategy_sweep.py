#!/usr/bin/env python3
"""
Comprehensive strategy parameter sweep for low-timeframe optimization.
Fetches data once per timeframe, runs 1000+ strategy/param combos,
ranks by Sharpe ratio.

Usage:
    python -m data_service.backtest.strategy_sweep --interval 5m --symbol ETHUSDT --limit 10000
"""

import argparse
import json
import logging
import multiprocessing
import sys
import time
import os
import warnings
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_service.backtest.crypto_backtest import fetch_binance_ohlcv, STRATEGY_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast vectorised backtest (no per-bar Python loop for equity curve)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _backtest_loop(
    closes, opens, highs, lows, sigs,
    initial_capital, total_cost, stop_loss, take_profit,
    leverage, maint_margin_rate,
):
    """
    Numba JIT-compiled inner backtest loop with futures leverage support.
    Returns: (equity_series, final_equity, max_dd, trades, wins, gross_profit, gross_loss, liquidations)
    """
    n = len(closes)
    use_sltp = stop_loss > 0 or take_profit > 0

    cash = initial_capital
    pos_qty = 0.0
    pos_entry = 0.0
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    equity_peak = initial_capital
    max_dd = 0.0
    equity_series = np.empty(n)
    pending_sig = 0
    liquidations = 0

    for i in range(n):
        price = closes[i]
        exec_price = opens[i]
        bar_high = highs[i]
        bar_low = lows[i]

        # --- Liquidation check (before SL/TP) ---
        if leverage > 1.0 and pos_qty != 0.0:
            abs_qty_liq = abs(pos_qty)
            margin_locked = abs_qty_liq * pos_entry / leverage
            if pos_qty > 0.0:
                worst_price = bar_low
                unreal_liq = (worst_price - pos_entry) * abs_qty_liq
            else:
                worst_price = bar_high
                unreal_liq = (pos_entry - worst_price) * abs_qty_liq
            maint_req = abs_qty_liq * worst_price * maint_margin_rate
            if margin_locked + unreal_liq <= maint_req:
                # Liquidated — lose all margin
                trades += 1
                gross_loss += margin_locked
                liquidations += 1
                pos_qty = 0.0
                pos_entry = 0.0
                pending_sig = 0
                equity_series[i] = cash
                if cash > equity_peak:
                    equity_peak = cash
                if equity_peak > 0.0:
                    dd = (equity_peak - cash) / equity_peak
                    if dd > max_dd:
                        max_dd = dd
                pending_sig = sigs[i]
                continue

        # --- Check SL/TP first (intra-bar using high/low) ---
        if use_sltp and pos_qty != 0.0:
            sl_hit = False
            tp_hit = False
            sl_price = 0.0
            tp_price = 0.0

            if pos_qty > 0.0:  # LONG
                if stop_loss > 0:
                    sl_price = pos_entry * (1.0 - stop_loss)
                    sl_hit = bar_low <= sl_price
                if take_profit > 0:
                    tp_price = pos_entry * (1.0 + take_profit)
                    tp_hit = bar_high >= tp_price
            else:  # SHORT
                if stop_loss > 0:
                    sl_price = pos_entry * (1.0 + stop_loss)
                    sl_hit = bar_high >= sl_price
                if take_profit > 0:
                    tp_price = pos_entry * (1.0 - take_profit)
                    tp_hit = bar_low <= tp_price

            if sl_hit or tp_hit:
                # If both trigger same bar, assume SL first (conservative)
                exit_p = sl_price if sl_hit else tp_price
                abs_qty = abs(pos_qty)
                if pos_qty > 0.0:
                    pnl = (exit_p - pos_entry) * abs_qty - total_cost * exit_p * abs_qty
                else:
                    pnl = (pos_entry - exit_p) * abs_qty - total_cost * exit_p * abs_qty
                cash += abs_qty * pos_entry / leverage + pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                pos_qty = 0.0
                pos_entry = 0.0
                pending_sig = 0

        # Mark-to-market
        if pos_qty > 0.0:
            unrealized = (price - pos_entry) * pos_qty
        elif pos_qty < 0.0:
            unrealized = (pos_entry - price) * (-pos_qty)
        else:
            unrealized = 0.0
        equity = cash + abs(pos_qty) * pos_entry / leverage + unrealized
        equity_series[i] = equity
        if equity > equity_peak:
            equity_peak = equity
        if equity_peak > 0.0:
            dd = (equity_peak - equity) / equity_peak
            if dd > max_dd:
                max_dd = dd

        # Execute pending signal at this candle's open
        if pending_sig == 1 and pos_qty <= 0.0:
            # Close short first
            if pos_qty < 0.0:
                abs_qty = -pos_qty
                pnl = (pos_entry - exec_price) * abs_qty - total_cost * exec_price * abs_qty
                cash += abs_qty * pos_entry / leverage + pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                pos_qty = 0.0
                pos_entry = 0.0

            # Open long (margin-based with leverage)
            usable = cash * 0.95
            cost_per_unit = exec_price * (1.0 / leverage + total_cost)
            qty = usable / cost_per_unit
            if qty > 0.0:
                pos_qty = qty
                pos_entry = exec_price
                cash -= qty * exec_price * (1.0 / leverage + total_cost)

        elif pending_sig == -1 and pos_qty >= 0.0:
            # Close long first
            if pos_qty > 0.0:
                pnl = (exec_price - pos_entry) * pos_qty - total_cost * exec_price * pos_qty
                cash += pos_qty * pos_entry / leverage + pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                pos_qty = 0.0
                pos_entry = 0.0

            # Open short (margin-based with leverage)
            usable = cash * 0.95
            cost_per_unit = exec_price * (1.0 / leverage + total_cost)
            qty = usable / cost_per_unit
            if qty > 0.0:
                pos_qty = -qty
                pos_entry = exec_price
                cash -= qty * exec_price * (1.0 / leverage + total_cost)

        pending_sig = sigs[i]

    # Final equity (close any open position at last price)
    last_price = closes[n - 1]
    if pos_qty > 0.0:
        abs_qty = pos_qty
        pnl = (last_price - pos_entry) * abs_qty - total_cost * last_price * abs_qty
        final_equity = cash + abs_qty * pos_entry / leverage + pnl
    elif pos_qty < 0.0:
        abs_qty = -pos_qty
        pnl = (pos_entry - last_price) * abs_qty - total_cost * last_price * abs_qty
        final_equity = cash + abs_qty * pos_entry / leverage + pnl
    else:
        final_equity = cash

    return equity_series, final_equity, max_dd, trades, wins, gross_profit, gross_loss, liquidations


def fast_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0002,
    bars_per_day: int = 96,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    leverage: float = 1.0,
) -> Dict[str, Any]:
    """
    Futures backtest with Numba JIT-compiled inner loop.
    First call incurs ~1s JIT compilation; subsequent calls run at native speed.

    Args:
        leverage: Futures leverage multiplier (1.0 = spot-equivalent).
    """
    total_cost = commission + slippage
    lev = max(1.0, float(leverage))
    maint_margin_rate = 0.005  # 0.5% of notional

    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    sigs = signals.values.astype(np.int64)

    equity_series, final_equity, max_dd, trades, wins, gross_profit, gross_loss, liquidations = \
        _backtest_loop(closes, opens, highs, lows, sigs,
                       initial_capital, total_cost, stop_loss, take_profit,
                       lev, maint_margin_rate)

    total_return = (final_equity - initial_capital) / initial_capital

    # Resample equity to daily for proper Sharpe (pandas, outside JIT)
    eq = pd.Series(equity_series)
    daily_eq = eq.iloc[::bars_per_day]
    if len(daily_eq) < 2:
        daily_eq = eq.iloc[[0, -1]]
    daily_returns = daily_eq.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 1e-10:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        sharpe = max(-50.0, min(50.0, sharpe))
    else:
        sharpe = 0.0

    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 1e-10:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(365)
        sortino = max(-50.0, min(50.0, sortino))
    else:
        sortino = 0.0

    win_rate = wins / trades if trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)

    return {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "trades": trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "final_equity": round(final_equity, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "liquidations": int(liquidations),
        "leverage": float(lev),
    }


# ---------------------------------------------------------------------------
# Parameter grids  (designed by a 30-year pro for LTF)
# ---------------------------------------------------------------------------

def build_param_grid() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Build massive parameter grid for all 29 strategies.
    Target: 17,000+ combinations per timeframe (50,000+ total across 3 TFs).
    """
    combos = []

    # 1. Momentum — 25
    for lb in [3, 5, 7, 10, 14, 20, 25, 30, 40, 50, 60, 75, 100, 120, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600]:
        combos.append(("Momentum", {"lookback": lb}))

    # 2. Mean Reversion — 12×8×8 = 768
    for rsi in [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60]:
        for os in [5, 10, 15, 20, 25, 30, 35, 40]:
            for ob in [55, 60, 65, 70, 75, 80, 85, 90]:
                combos.append(("Mean Reversion", {"rsi_period": rsi, "oversold": os, "overbought": ob}))

    # 3. Breakout — 22
    for lb in [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 40, 50, 60, 75, 100, 120, 150, 200, 250, 300, 350, 400]:
        combos.append(("Breakout", {"lookback": lb}))

    # 4. Multi-Factor — 14×10 = 140
    for mom in [3, 5, 7, 10, 14, 20, 30, 40, 50, 75, 100, 150, 200, 300]:
        for rsi in [3, 5, 7, 10, 14, 20, 25, 30, 40, 50]:
            combos.append(("Multi-Factor", {"mom_lookback": mom, "rsi_period": rsi}))

    # 5. EMA Crossover — ~150
    for fast in [3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 35]:
        for slow in [12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 250]:
            if fast < slow:
                combos.append(("EMA Crossover", {"fast_period": fast, "slow_period": slow}))

    # 6. MACD — ~400
    for fast in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        for slow in [16, 18, 20, 21, 24, 26, 28, 30, 36, 40, 50]:
            for sig in [3, 5, 7, 9, 11, 14]:
                if fast < slow:
                    combos.append(("MACD", {"fast": fast, "slow": slow, "signal_period": sig}))

    # 7. Bollinger Bands — 14×14 = 196
    for period in [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100]:
        for std in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]:
            combos.append(("Bollinger Bands", {"period": period, "num_std": std}))

    # 8. Stochastic — 10×5×7×7 = 2450
    for k in [3, 5, 7, 9, 10, 14, 20, 25, 30, 40]:
        for d in [2, 3, 5, 7, 10]:
            for os in [5, 10, 15, 20, 25, 30, 35]:
                for ob in [65, 70, 75, 80, 85, 90, 95]:
                    combos.append(("Stochastic", {"k_period": k, "d_period": d, "oversold": os, "overbought": ob}))

    # 9. Keltner Channel — 10×8×9 = 720
    for ema in [3, 5, 7, 10, 14, 20, 25, 30, 40, 50]:
        for atr_p in [3, 5, 7, 10, 14, 20, 25, 30]:
            for mult in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                combos.append(("Keltner Channel", {"ema_period": ema, "atr_period": atr_p, "atr_mult": mult}))

    # 10. Williams %R — 10×7×7 = 490
    for period in [5, 7, 10, 14, 20, 25, 30, 40, 50, 60]:
        for os in [-95, -90, -85, -80, -75, -70, -65]:
            for ob in [-35, -30, -25, -20, -15, -10, -5]:
                combos.append(("Williams %R", {"period": period, "oversold": os, "overbought": ob}))

    # 11. CCI — 10×8×8 = 640
    for period in [5, 7, 10, 14, 20, 25, 30, 40, 50, 60]:
        for buy in [-300, -250, -200, -150, -100, -75, -50, -25]:
            for sell in [25, 50, 75, 100, 150, 200, 250, 300]:
                combos.append(("CCI", {"period": period, "buy_level": buy, "sell_level": sell}))

    # 12. Triple EMA — ~350
    for fast in [3, 5, 7, 8, 10, 12, 15]:
        for mid in [10, 12, 15, 18, 20, 25, 30, 35, 40]:
            for slow in [25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150]:
                if fast < mid < slow:
                    combos.append(("Triple EMA", {"fast": fast, "mid": mid, "slow": slow}))

    # 13. RSI+EMA — 9×12×6×7 = 4536
    for rsi in [3, 5, 7, 10, 14, 18, 20, 25, 30]:
        for ema in [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
            for os in [15, 20, 25, 30, 35, 40]:
                for ob in [55, 60, 65, 70, 75, 80, 85]:
                    combos.append(("RSI+EMA", {"rsi_period": rsi, "ema_period": ema, "oversold": os, "overbought": ob}))

    # 14. Supertrend — 12×14 = 168
    for atr_p in [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 40, 50]:
        for mult in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
            combos.append(("Supertrend", {"atr_period": atr_p, "multiplier": mult}))

    # 15. ADX Trend — 7×7×6 = 294
    for adx_p in [5, 7, 10, 14, 20, 25, 30]:
        for thresh in [15, 20, 25, 30, 35, 40, 45]:
            for di_p in [5, 7, 10, 14, 20, 25]:
                combos.append(("ADX Trend", {"adx_period": adx_p, "di_period": di_p, "threshold": thresh}))

    # 16. Parabolic SAR — 6×5×7 = 210
    for af_s in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        for af_i in [0.005, 0.01, 0.015, 0.02, 0.025]:
            for af_m in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
                combos.append(("Parabolic SAR", {"af_start": af_s, "af_increment": af_i, "af_max": af_m}))

    # 17. Hull MA — 22
    for period in [5, 7, 9, 12, 14, 16, 20, 25, 30, 35, 40, 50, 60, 75, 80, 100, 120, 150, 175, 200, 250, 300]:
        combos.append(("Hull MA", {"period": period}))

    # 18. Donchian ATR — 10×7×7 = 490
    for donch in [5, 7, 10, 14, 20, 25, 30, 40, 50, 60]:
        for atr_p in [5, 7, 10, 14, 20, 25, 30]:
            for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
                combos.append(("Donchian ATR", {"donch_period": donch, "atr_period": atr_p, "atr_mult": mult}))

    # 19. Double RSI — 7×7×6×6 = 1764 (filtered ~1400)
    for fast_r in [2, 3, 5, 7, 10, 14, 20]:
        for slow_r in [7, 10, 14, 20, 25, 30, 40]:
            for os in [15, 20, 25, 30, 35, 40]:
                for ob in [60, 65, 70, 75, 80, 85]:
                    if fast_r < slow_r:
                        combos.append(("Double RSI", {"fast_rsi": fast_r, "slow_rsi": slow_r, "oversold": os, "overbought": ob}))

    # 20. VWAP Bands — 12×12 = 144
    for period in [5, 7, 10, 14, 20, 25, 30, 40, 50, 60, 80, 100]:
        for std in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]:
            combos.append(("VWAP Bands", {"period": period, "num_std": std}))

    # --- 10 NEW STRATEGIES ---

    # 21. Ichimoku — 6×6×4 = 144
    for tenkan in [5, 7, 9, 12, 15, 20]:
        for kijun in [15, 20, 26, 30, 40, 52]:
            for senkou in [30, 40, 52, 65]:
                if tenkan < kijun < senkou:
                    combos.append(("Ichimoku", {"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou}))

    # 22. TRIX — 10×8 = 80
    for period in [5, 7, 10, 12, 15, 18, 20, 25, 30, 40]:
        for sig in [3, 5, 7, 9, 11, 13, 15, 20]:
            combos.append(("TRIX", {"period": period, "signal_period": sig}))

    # 23. Aroon — 10×8 = 80
    for period in [7, 10, 14, 20, 25, 30, 40, 50, 60, 75]:
        for thresh in [50, 55, 60, 65, 70, 75, 80, 85]:
            combos.append(("Aroon", {"period": period, "threshold": thresh}))

    # 24. MFI — 10×8×8 = 640
    for period in [3, 5, 7, 10, 14, 20, 25, 30, 40, 50]:
        for os in [5, 10, 15, 20, 25, 30, 35, 40]:
            for ob in [60, 65, 70, 75, 80, 85, 90, 95]:
                combos.append(("MFI", {"period": period, "oversold": os, "overbought": ob}))

    # 25. Squeeze Momentum — 6×6×6×6 = ~800 (filtered)
    for bb_p in [10, 14, 20, 25, 30, 40]:
        for bb_s in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            for kc_p in [10, 14, 20, 25, 30, 40]:
                for kc_m in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
                    combos.append(("Squeeze", {"bb_period": bb_p, "bb_std": bb_s, "kc_period": kc_p, "kc_mult": kc_m}))

    # 26. Elder Ray — 10×5 = 50
    for ema_p in [5, 7, 10, 13, 15, 20, 25, 30, 40, 50]:
        for thresh in [0.0, 0.001, 0.002, 0.005, 0.01]:
            combos.append(("Elder Ray", {"ema_period": ema_p, "threshold": thresh}))

    # 27. CMF — 10×8 = 80
    for period in [5, 7, 10, 14, 20, 25, 30, 40, 50, 60]:
        for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
            combos.append(("CMF", {"period": period, "threshold": thresh}))

    # 28. Fisher Transform — 12
    for period in [5, 7, 8, 10, 12, 14, 18, 20, 25, 30, 40, 50]:
        combos.append(("Fisher Transform", {"period": period}))

    # 29. KST — 6×6×5×5×6 = ~1000 (filtered)
    for r1 in [5, 7, 10, 14, 20, 25]:
        for r2 in [10, 14, 15, 20, 25, 30]:
            for r3 in [15, 20, 25, 30, 40]:
                for r4 in [20, 25, 30, 40, 50]:
                    for sig in [5, 7, 9, 11, 14, 20]:
                        if r1 < r2 < r3 < r4:
                            combos.append(("KST", {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "signal_period": sig}))

    return combos


# ---------------------------------------------------------------------------
# Multiprocessing worker functions
# ---------------------------------------------------------------------------

def _reconstruct_df(arrays):
    """Reconstruct minimal DataFrame from numpy arrays for signal computation."""
    closes, opens, highs, lows, volumes, index_values = arrays
    return pd.DataFrame(
        {"close": closes, "open": opens, "high": highs, "low": lows, "volume": volumes},
        index=pd.DatetimeIndex(index_values),
    )


def _extract_df_arrays(df):
    """Extract numpy arrays from DataFrame for cheap serialization."""
    return (
        df["close"].values,
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["volume"].values,
        df.index.values,
    )


def _sweep_chunk_worker(args):
    """Worker: process a chunk of strategies for a single-symbol sweep."""
    chunk, df_arrays, bpd, stop_loss, take_profit = args
    df = _reconstruct_df(df_arrays)
    results = []
    errors = 0
    for strat_name, params in chunk:
        signal_func = STRATEGY_MAP.get(strat_name)
        if signal_func is None:
            errors += 1
            continue
        try:
            signals = signal_func(df, **params)
            metrics = fast_backtest(df, signals, bars_per_day=bpd,
                                    stop_loss=stop_loss, take_profit=take_profit)
            if metrics["trades"] < 3:
                continue
            results.append({
                "strategy": strat_name,
                "params": params,
                **metrics,
            })
        except Exception:
            errors += 1
    return results, errors


def _wf_chunk_worker(args):
    """Worker: process a chunk of strategies for walk-forward sweep."""
    chunk, df_arrays, windows, bpd, stop_loss, take_profit = args
    df = _reconstruct_df(df_arrays)
    results = []
    errors = 0
    for strat_name, params in chunk:
        signal_func = STRATEGY_MAP.get(strat_name)
        if signal_func is None:
            errors += 1
            continue
        try:
            signals = signal_func(df, **params)
        except Exception:
            errors += 1
            continue

        is_metrics_list = []
        oos_metrics_list = []
        for tr_s, tr_e, te_s, te_e in windows:
            try:
                df_train = df.iloc[tr_s:tr_e]
                sig_train = signals.iloc[tr_s:tr_e]
                is_m = fast_backtest(df_train, sig_train, bars_per_day=bpd,
                                     stop_loss=stop_loss, take_profit=take_profit)
                df_test = df.iloc[te_s:te_e]
                sig_test = signals.iloc[te_s:te_e]
                oos_m = fast_backtest(df_test, sig_test, bars_per_day=bpd,
                                      stop_loss=stop_loss, take_profit=take_profit)
                is_metrics_list.append(is_m)
                oos_metrics_list.append(oos_m)
            except Exception:
                errors += 1

        if len(oos_metrics_list) < 2:
            continue

        oos_sharpes = [m["sharpe"] for m in oos_metrics_list]
        oos_returns = [m["total_return"] for m in oos_metrics_list]
        oos_dds = [m["max_drawdown"] for m in oos_metrics_list]
        oos_trades = [m["trades"] for m in oos_metrics_list]
        oos_wrs = [m["win_rate"] for m in oos_metrics_list]
        oos_pfs = [m["profit_factor"] for m in oos_metrics_list]
        is_sharpes = [m["sharpe"] for m in is_metrics_list]

        avg_oos_trades = np.mean(oos_trades)
        if avg_oos_trades < 2:
            continue

        is_sharpe_mean = float(np.mean(is_sharpes))
        oos_sharpe_mean = float(np.mean(oos_sharpes))
        consistency = float(np.mean([1 if r > 0 else 0 for r in oos_returns]))
        sharpe_decay = 1.0 - (oos_sharpe_mean / is_sharpe_mean) if is_sharpe_mean != 0 else 1.0

        per_window_oos = []
        for i, m in enumerate(oos_metrics_list):
            per_window_oos.append({
                "window": i + 1,
                "sharpe": m["sharpe"],
                "total_return": m["total_return"],
                "max_drawdown": m["max_drawdown"],
                "trades": m["trades"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
            })

        results.append({
            "strategy": strat_name,
            "params": params,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "n_windows": len(oos_metrics_list),
            "is_sharpe_mean": round(is_sharpe_mean, 4),
            "oos_sharpe_mean": round(oos_sharpe_mean, 4),
            "oos_sharpe_std": round(float(np.std(oos_sharpes)), 4),
            "oos_sharpe_min": round(float(np.min(oos_sharpes)), 4),
            "oos_return_mean": round(float(np.mean(oos_returns)), 6),
            "oos_max_dd_mean": round(float(np.mean(oos_dds)), 6),
            "oos_trades_mean": round(float(avg_oos_trades), 1),
            "oos_win_rate_mean": round(float(np.mean(oos_wrs)), 4),
            "oos_profit_factor_mean": round(float(np.mean(oos_pfs)), 4),
            "consistency": round(consistency, 4),
            "sharpe_decay": round(sharpe_decay, 4),
            "per_window_oos": per_window_oos,
        })
    return results, errors


def _mega_sweep_coin_worker(args):
    """Worker: process one coin's full mega sweep (non-walkforward)."""
    (sym, interval, limit, output_dir, sl_tp_levels, top_n) = args
    logger.info(f"\n--- [worker] {sym} {interval} (limit={limit}) ---")
    t_sym = time.time()

    df = fetch_binance_ohlcv(sym, interval=interval, limit=limit)
    if df is None or df.empty:
        logger.error(f"  [worker] Failed to fetch {sym}")
        return None

    logger.info(f"  [worker] {sym}: fetched {len(df)} candles")

    grid = build_param_grid()
    interval_to_bpd = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    bpd = interval_to_bpd.get(interval, 96)

    results = []
    errors = 0
    t0 = time.time()

    for idx, (strat_name, params) in enumerate(grid):
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            done = idx * len(sl_tp_levels)
            total = len(grid) * len(sl_tp_levels)
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(f"  [worker {sym}] {done}/{total} ({rate:.0f}/s)")

        signal_func = STRATEGY_MAP.get(strat_name)
        if signal_func is None:
            errors += 1
            continue

        try:
            signals = signal_func(df, **params)
        except Exception:
            errors += 1
            continue

        for sl, tp in sl_tp_levels:
            try:
                metrics = fast_backtest(df, signals, bars_per_day=bpd,
                                        stop_loss=sl, take_profit=tp)
                if metrics["trades"] < 3:
                    continue
                results.append({
                    "strategy": strat_name,
                    "params": params,
                    "symbol": sym,
                    "interval": interval,
                    "candles": len(df),
                    "stop_loss": sl,
                    "take_profit": tp,
                    **metrics,
                })
            except Exception:
                errors += 1

    sweep_time = time.time() - t_sym
    logger.info(f"  [worker {sym}] Done: {len(results)} valid in {sweep_time:.1f}s")

    if not results:
        return None

    results.sort(key=lambda r: (r["sharpe"], r["profit_factor"], r["total_return"]),
                 reverse=True)

    outfile = os.path.join(output_dir, f"mega_{sym}_{interval}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  [worker {sym}] Saved {len(results)} results to {outfile}")
    return sym


def _mega_wf_coin_worker(args):
    """Worker: process one coin's full walk-forward sweep."""
    (sym, interval, limit, output_dir, sl_tp_levels, top_n,
     n_windows, train_pct) = args
    logger.info(f"\n--- [worker] {sym} {interval} (limit={limit}) ---")
    t_sym = time.time()

    df = fetch_binance_ohlcv(sym, interval=interval, limit=limit)
    if df is None or df.empty:
        logger.error(f"  [worker] Failed to fetch {sym}")
        return None

    logger.info(f"  [worker] {sym}: fetched {len(df)} candles")

    windows = generate_wf_windows(len(df), n_windows, train_pct)
    if not windows:
        logger.error(f"  [worker] Not enough data for walk-forward on {sym}")
        return None

    grid = build_param_grid()
    interval_to_bpd = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    bpd = interval_to_bpd.get(interval, 96)

    results = []
    errors = 0
    t0 = time.time()

    for idx, (strat_name, params) in enumerate(grid):
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  [worker {sym}] {idx+1}/{len(grid)} strategies ({rate:.0f}/s)")

        signal_func = STRATEGY_MAP.get(strat_name)
        if signal_func is None:
            errors += 1
            continue

        try:
            signals = signal_func(df, **params)
        except Exception:
            errors += 1
            continue

        for sl, tp in sl_tp_levels:
            is_metrics_list = []
            oos_metrics_list = []

            for tr_s, tr_e, te_s, te_e in windows:
                try:
                    df_train = df.iloc[tr_s:tr_e]
                    sig_train = signals.iloc[tr_s:tr_e]
                    is_m = fast_backtest(df_train, sig_train, bars_per_day=bpd,
                                         stop_loss=sl, take_profit=tp)
                    df_test = df.iloc[te_s:te_e]
                    sig_test = signals.iloc[te_s:te_e]
                    oos_m = fast_backtest(df_test, sig_test, bars_per_day=bpd,
                                          stop_loss=sl, take_profit=tp)
                    is_metrics_list.append(is_m)
                    oos_metrics_list.append(oos_m)
                except Exception:
                    errors += 1

            if len(oos_metrics_list) < 2:
                continue

            oos_sharpes = [m["sharpe"] for m in oos_metrics_list]
            oos_returns = [m["total_return"] for m in oos_metrics_list]
            oos_dds = [m["max_drawdown"] for m in oos_metrics_list]
            oos_trades = [m["trades"] for m in oos_metrics_list]
            oos_wrs = [m["win_rate"] for m in oos_metrics_list]
            oos_pfs = [m["profit_factor"] for m in oos_metrics_list]
            is_sharpes = [m["sharpe"] for m in is_metrics_list]

            avg_oos_trades = np.mean(oos_trades)
            if avg_oos_trades < 2:
                continue

            is_sharpe_mean = float(np.mean(is_sharpes))
            oos_sharpe_mean = float(np.mean(oos_sharpes))
            consistency = float(np.mean([1 if r > 0 else 0 for r in oos_returns]))
            sharpe_decay = 1.0 - (oos_sharpe_mean / is_sharpe_mean) if is_sharpe_mean != 0 else 1.0

            per_window_oos = []
            for i, m in enumerate(oos_metrics_list):
                per_window_oos.append({
                    "window": i + 1,
                    "sharpe": m["sharpe"],
                    "total_return": m["total_return"],
                    "max_drawdown": m["max_drawdown"],
                    "trades": m["trades"],
                    "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"],
                })

            results.append({
                "strategy": strat_name,
                "params": params,
                "symbol": sym,
                "interval": interval,
                "candles": len(df),
                "stop_loss": sl,
                "take_profit": tp,
                "n_windows": len(oos_metrics_list),
                "is_sharpe_mean": round(is_sharpe_mean, 4),
                "oos_sharpe_mean": round(oos_sharpe_mean, 4),
                "oos_sharpe_std": round(float(np.std(oos_sharpes)), 4),
                "oos_sharpe_min": round(float(np.min(oos_sharpes)), 4),
                "oos_return_mean": round(float(np.mean(oos_returns)), 6),
                "oos_max_dd_mean": round(float(np.mean(oos_dds)), 6),
                "oos_trades_mean": round(float(avg_oos_trades), 1),
                "oos_win_rate_mean": round(float(np.mean(oos_wrs)), 4),
                "oos_profit_factor_mean": round(float(np.mean(oos_pfs)), 4),
                "consistency": round(consistency, 4),
                "sharpe_decay": round(sharpe_decay, 4),
                "per_window_oos": per_window_oos,
            })

    sweep_time = time.time() - t_sym
    logger.info(f"  [worker {sym}] Done: {len(results)} valid in {sweep_time:.1f}s")

    if not results:
        return None

    results.sort(key=lambda r: (r["oos_sharpe_mean"], r["consistency"], r["oos_return_mean"]),
                 reverse=True)

    outfile = os.path.join(output_dir, f"wf_{sym}_{interval}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  [worker {sym}] Saved {len(results)} results to {outfile}")
    return sym


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    symbol: str,
    interval: str,
    limit: int,
    output_file: str,
    top_n: int = 50,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    workers: int = 1,
):
    """Fetch data once, run all combos, save sorted results."""
    sl_tp_str = ""
    if stop_loss > 0 or take_profit > 0:
        sl_tp_str = f" SL={stop_loss:.1%} TP={take_profit:.1%}"
    workers_str = f" workers={workers}" if workers > 1 else ""
    logger.info(f"=== SWEEP: {symbol} {interval} (limit={limit}){sl_tp_str}{workers_str} ===")

    # 1. Fetch data
    t0 = time.time()
    df = fetch_binance_ohlcv(symbol, interval=interval, limit=limit)
    if df is None or df.empty:
        logger.error(f"Failed to fetch data for {symbol} {interval}")
        return
    fetch_time = time.time() - t0
    logger.info(f"Fetched {len(df)} candles in {fetch_time:.1f}s  "
                f"(range: {df.index[0]} → {df.index[-1]})")

    # 2. Build param grid
    grid = build_param_grid()
    logger.info(f"Parameter grid: {len(grid)} combinations")

    # Bars per day for daily return resampling
    interval_to_bpd = {
        "1m": 1440, "5m": 288, "15m": 96,
        "1h": 24, "4h": 6, "1d": 1,
    }
    bpd = interval_to_bpd.get(interval, 96)

    # 3. Run all combos
    results = []
    errors = 0
    t0 = time.time()

    if workers > 1:
        # Parallel: split grid into chunks, send numpy arrays
        df_arrays = _extract_df_arrays(df)
        n_chunks = min(workers, len(grid))
        chunk_size = (len(grid) + n_chunks - 1) // n_chunks
        chunks = [grid[i:i + chunk_size] for i in range(0, len(grid), chunk_size)]
        chunk_args = [(c, df_arrays, bpd, stop_loss, take_profit) for c in chunks]

        logger.info(f"Spawning {len(chunks)} workers ({chunk_size} strategies/chunk)")
        with multiprocessing.Pool(n_chunks) as pool:
            for i, (chunk_results, chunk_errors) in enumerate(
                pool.imap_unordered(_sweep_chunk_worker, chunk_args)
            ):
                results.extend(chunk_results)
                errors += chunk_errors
                logger.info(f"  Chunk done: +{len(chunk_results)} results "
                            f"(total {len(results)})")

        # Add symbol/interval/candles metadata
        for r in results:
            r["symbol"] = symbol
            r["interval"] = interval
            r["candles"] = len(df)
    else:
        for idx, (strat_name, params) in enumerate(grid):
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                logger.info(f"  Progress: {idx+1}/{len(grid)} ({rate:.0f}/s)")

            signal_func = STRATEGY_MAP.get(strat_name)
            if signal_func is None:
                errors += 1
                continue

            try:
                signals = signal_func(df, **params)
                metrics = fast_backtest(df, signals, bars_per_day=bpd,
                                        stop_loss=stop_loss, take_profit=take_profit)

                # Skip if no trades (strategy never triggered)
                if metrics["trades"] < 3:
                    continue

                results.append({
                    "strategy": strat_name,
                    "params": params,
                    "symbol": symbol,
                    "interval": interval,
                    "candles": len(df),
                    **metrics,
                })
            except Exception as e:
                errors += 1

    sweep_time = time.time() - t0
    logger.info(f"Completed {len(grid)} combos in {sweep_time:.1f}s "
                f"({len(grid)/sweep_time:.0f}/s). "
                f"Valid results: {len(results)}, Errors: {errors}")

    if not results:
        logger.warning("No valid results!")
        return

    # 4. Sort by Sharpe ratio (primary), then profit factor, then total return
    results.sort(key=lambda r: (r["sharpe"], r["profit_factor"], r["total_return"]), reverse=True)

    # 5. Save full results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {output_file}")

    # 6. Print top N
    print(f"\n{'='*100}")
    print(f"  TOP {top_n} RESULTS — {symbol} {interval} ({len(df)} candles)")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Strategy':<18} {'Return':>9} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} "
          f"{'Trades':>7} {'WinR':>6} {'PF':>7} {'Params'}")
    print(f"{'-'*100}")

    for i, r in enumerate(results[:top_n]):
        params_str = str(r["params"])
        if len(params_str) > 45:
            params_str = params_str[:42] + "..."
        print(f"{i+1:>3} {r['strategy']:<18} {r['total_return']:>8.2%} {r['sharpe']:>8.2f} "
              f"{r['sortino']:>8.2f} {r['max_drawdown']:>7.2%} {r['trades']:>7} "
              f"{r['win_rate']:>5.1%} {r['profit_factor']:>7.2f} {params_str}")

    print(f"\nTotal combinations tested: {len(grid)}")
    print(f"Combinations with 3+ trades: {len(results)}")
    print(f"Best Sharpe: {results[0]['sharpe']:.4f} ({results[0]['strategy']} {results[0]['params']})")
    best_ret = max(results, key=lambda r: r["total_return"])
    print(f"Best Return: {best_ret['total_return']:.4%} ({best_ret['strategy']} {best_ret['params']})")
    best_pf = max(results, key=lambda r: r["profit_factor"] if r["profit_factor"] < 900 else 0)
    print(f"Best PF:     {best_pf['profit_factor']:.2f} ({best_pf['strategy']} {best_pf['params']})")

    return results


def run_mega_sweep(
    symbols: List[str],
    interval: str,
    limit: int,
    output_dir: str,
    sl_tp_levels: List[Tuple[float, float]],
    top_n: int = 50,
    workers: int = 1,
):
    """
    Run sweep across multiple coins with multiple SL/TP levels.
    Signals computed once per strategy+params; backtested for each SL/TP level.
    When workers > 1, coins are processed in parallel (one worker per coin).
    """
    grid = build_param_grid()
    total_combos = len(grid) * len(sl_tp_levels)
    logger.info(f"=== MEGA SWEEP: {len(symbols)} coins × {interval} × "
                f"{len(grid)} strategies × {len(sl_tp_levels)} SL/TP = "
                f"{total_combos} tests/coin, {total_combos * len(symbols)} total ===")
    if workers > 1:
        logger.info(f"  Using {workers} parallel workers (coin-level parallelism)")

    os.makedirs(output_dir, exist_ok=True)

    if workers > 1:
        coin_args = [
            (sym, interval, limit, output_dir, sl_tp_levels, top_n)
            for sym in symbols
        ]
        n_pool = min(workers, len(symbols))
        with multiprocessing.Pool(n_pool) as pool:
            completed = list(pool.imap_unordered(_mega_sweep_coin_worker, coin_args))
        done = [s for s in completed if s is not None]
        logger.info(f"Mega sweep complete: {len(done)}/{len(symbols)} coins processed")
    else:
        interval_to_bpd = {
            "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
        }
        bpd = interval_to_bpd.get(interval, 96)

        for sym in symbols:
            logger.info(f"\n--- {sym} {interval} (limit={limit}) ---")
            t_sym = time.time()

            df = fetch_binance_ohlcv(sym, interval=interval, limit=limit)
            if df is None or df.empty:
                logger.error(f"  Failed to fetch {sym}")
                continue
            logger.info(f"  Fetched {len(df)} candles "
                        f"({df.index[0]} → {df.index[-1]})")

            results = []
            errors = 0
            t0 = time.time()

            for idx, (strat_name, params) in enumerate(grid):
                if (idx + 1) % 1000 == 0:
                    elapsed = time.time() - t0
                    done = idx * len(sl_tp_levels)
                    total = len(grid) * len(sl_tp_levels)
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(f"  [{sym}] {done}/{total} ({rate:.0f}/s)")

                signal_func = STRATEGY_MAP.get(strat_name)
                if signal_func is None:
                    errors += 1
                    continue

                try:
                    signals = signal_func(df, **params)
                except Exception:
                    errors += 1
                    continue

                for sl, tp in sl_tp_levels:
                    try:
                        metrics = fast_backtest(df, signals, bars_per_day=bpd,
                                                stop_loss=sl, take_profit=tp)
                        if metrics["trades"] < 3:
                            continue
                        results.append({
                            "strategy": strat_name,
                            "params": params,
                            "symbol": sym,
                            "interval": interval,
                            "candles": len(df),
                            "stop_loss": sl,
                            "take_profit": tp,
                            **metrics,
                        })
                    except Exception:
                        errors += 1

            sweep_time = time.time() - t_sym
            logger.info(f"  [{sym}] Done: {len(results)} valid / "
                        f"{len(grid) * len(sl_tp_levels)} tested in {sweep_time:.1f}s "
                        f"({len(grid) * len(sl_tp_levels) / sweep_time:.0f}/s)")

            if not results:
                continue

            results.sort(key=lambda r: (r["sharpe"], r["profit_factor"], r["total_return"]),
                         reverse=True)

            outfile = os.path.join(output_dir, f"mega_{sym}_{interval}.json")
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"  Saved {len(results)} results to {outfile}")

            print(f"\n{'='*110}")
            print(f"  TOP {top_n} — {sym} {interval} ({len(df)} candles)")
            print(f"{'='*110}")
            print(f"{'#':>3} {'Strategy':<18} {'Return':>9} {'Sharpe':>8} {'Sortino':>8} "
                  f"{'MaxDD':>8} {'Trades':>7} {'WinR':>6} {'PF':>7} {'SL/TP':>10} {'Params'}")
            print(f"{'-'*110}")
            for i, r in enumerate(results[:top_n]):
                ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())
                if len(ps) > 40:
                    ps = ps[:37] + "..."
                sl_str = f"{r.get('stop_loss',0)*100:.1f}/{r.get('take_profit',0)*100:.1f}"
                print(f"{i+1:>3} {r['strategy']:<18} {r['total_return']:>8.2%} "
                      f"{r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_drawdown']:>7.2%} "
                      f"{r['trades']:>7} {r['win_rate']:>5.1%} {r['profit_factor']:>7.2f} "
                      f"{sl_str:>10} {ps}")


# ---------------------------------------------------------------------------
# Walk-Forward Testing
# ---------------------------------------------------------------------------

def generate_wf_windows(n_bars, n_windows=5, train_pct=0.70, min_test_bars=100):
    """
    Return list of (train_start, train_end, test_start, test_end) index tuples.
    Windows are anchored so the test slices tile the rightmost portion of the data
    and each train window is a fixed multiple of the test size.
    """
    test_size = int(n_bars * (1 - train_pct) / n_windows)
    test_size = max(test_size, min_test_bars)
    train_size = int(test_size * train_pct / (1 - train_pct))
    windows = []
    for i in range(n_windows):
        test_end = n_bars - (n_windows - 1 - i) * test_size
        test_start = test_end - test_size
        train_start = max(0, test_start - train_size)
        train_end = test_start
        if train_end - train_start >= min_test_bars:
            windows.append((train_start, train_end, test_start, test_end))
    return windows


def run_walkforward_sweep(
    symbol: str,
    interval: str,
    limit: int,
    output_file: str,
    top_n: int = 50,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    n_windows: int = 5,
    train_pct: float = 0.70,
    workers: int = 1,
):
    """Walk-forward sweep: train on one window, validate on the next unseen window."""
    sl_tp_str = ""
    if stop_loss > 0 or take_profit > 0:
        sl_tp_str = f" SL={stop_loss:.1%} TP={take_profit:.1%}"
    workers_str = f" workers={workers}" if workers > 1 else ""
    logger.info(f"=== WALK-FORWARD SWEEP: {symbol} {interval} (limit={limit}){sl_tp_str}{workers_str} "
                f"windows={n_windows} train={train_pct:.0%} ===")

    # 1. Fetch data
    t0 = time.time()
    df = fetch_binance_ohlcv(symbol, interval=interval, limit=limit)
    if df is None or df.empty:
        logger.error(f"Failed to fetch data for {symbol} {interval}")
        return
    fetch_time = time.time() - t0
    logger.info(f"Fetched {len(df)} candles in {fetch_time:.1f}s  "
                f"(range: {df.index[0]} → {df.index[-1]})")

    # 2. Generate windows
    windows = generate_wf_windows(len(df), n_windows, train_pct)
    if not windows:
        logger.error("Not enough data to generate walk-forward windows")
        return
    logger.info(f"Generated {len(windows)} walk-forward windows")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        logger.info(f"  Window {i+1}: train[{tr_s}:{tr_e}] ({tr_e-tr_s} bars) → "
                    f"test[{te_s}:{te_e}] ({te_e-te_s} bars)")

    # 3. Build param grid
    grid = build_param_grid()
    logger.info(f"Parameter grid: {len(grid)} combinations × {len(windows)} windows "
                f"= {len(grid) * len(windows) * 2} backtests")

    interval_to_bpd = {
        "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
    }
    bpd = interval_to_bpd.get(interval, 96)

    # 4. Run all combos
    results = []
    errors = 0
    t0 = time.time()

    if workers > 1:
        # Parallel: split grid into chunks, send numpy arrays
        df_arrays = _extract_df_arrays(df)
        n_chunks = min(workers, len(grid))
        chunk_size = (len(grid) + n_chunks - 1) // n_chunks
        chunks = [grid[i:i + chunk_size] for i in range(0, len(grid), chunk_size)]
        chunk_args = [(c, df_arrays, windows, bpd, stop_loss, take_profit) for c in chunks]

        logger.info(f"Spawning {len(chunks)} workers ({chunk_size} strategies/chunk)")
        with multiprocessing.Pool(n_chunks) as pool:
            for i, (chunk_results, chunk_errors) in enumerate(
                pool.imap_unordered(_wf_chunk_worker, chunk_args)
            ):
                results.extend(chunk_results)
                errors += chunk_errors
                logger.info(f"  Chunk done: +{len(chunk_results)} results "
                            f"(total {len(results)})")

        # Add symbol/interval/candles metadata
        for r in results:
            r["symbol"] = symbol
            r["interval"] = interval
            r["candles"] = len(df)
    else:
        for idx, (strat_name, params) in enumerate(grid):
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                logger.info(f"  Progress: {idx+1}/{len(grid)} ({rate:.0f} strategies/s)")

            signal_func = STRATEGY_MAP.get(strat_name)
            if signal_func is None:
                errors += 1
                continue

            try:
                # Compute signals ONCE on full dataset (no look-ahead: signal[i] uses df[:i+1])
                signals = signal_func(df, **params)
            except Exception:
                errors += 1
                continue

            is_metrics_list = []
            oos_metrics_list = []

            for tr_s, tr_e, te_s, te_e in windows:
                try:
                    # Train (in-sample)
                    df_train = df.iloc[tr_s:tr_e]
                    sig_train = signals.iloc[tr_s:tr_e]
                    is_m = fast_backtest(df_train, sig_train, bars_per_day=bpd,
                                         stop_loss=stop_loss, take_profit=take_profit)

                    # Test (out-of-sample)
                    df_test = df.iloc[te_s:te_e]
                    sig_test = signals.iloc[te_s:te_e]
                    oos_m = fast_backtest(df_test, sig_test, bars_per_day=bpd,
                                          stop_loss=stop_loss, take_profit=take_profit)

                    is_metrics_list.append(is_m)
                    oos_metrics_list.append(oos_m)
                except Exception:
                    errors += 1

            if len(oos_metrics_list) < 2:
                continue

            # Aggregate OOS metrics
            oos_sharpes = [m["sharpe"] for m in oos_metrics_list]
            oos_returns = [m["total_return"] for m in oos_metrics_list]
            oos_dds = [m["max_drawdown"] for m in oos_metrics_list]
            oos_trades = [m["trades"] for m in oos_metrics_list]
            oos_wrs = [m["win_rate"] for m in oos_metrics_list]
            oos_pfs = [m["profit_factor"] for m in oos_metrics_list]
            is_sharpes = [m["sharpe"] for m in is_metrics_list]

            avg_oos_trades = np.mean(oos_trades)
            if avg_oos_trades < 2:
                continue

            is_sharpe_mean = float(np.mean(is_sharpes))
            oos_sharpe_mean = float(np.mean(oos_sharpes))
            consistency = float(np.mean([1 if r > 0 else 0 for r in oos_returns]))
            sharpe_decay = 1.0 - (oos_sharpe_mean / is_sharpe_mean) if is_sharpe_mean != 0 else 1.0

            per_window_oos = []
            for i, m in enumerate(oos_metrics_list):
                per_window_oos.append({
                    "window": i + 1,
                    "sharpe": m["sharpe"],
                    "total_return": m["total_return"],
                    "max_drawdown": m["max_drawdown"],
                    "trades": m["trades"],
                    "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"],
                })

            results.append({
                "strategy": strat_name,
                "params": params,
                "symbol": symbol,
                "interval": interval,
                "candles": len(df),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "n_windows": len(oos_metrics_list),
                "is_sharpe_mean": round(is_sharpe_mean, 4),
                "oos_sharpe_mean": round(oos_sharpe_mean, 4),
                "oos_sharpe_std": round(float(np.std(oos_sharpes)), 4),
                "oos_sharpe_min": round(float(np.min(oos_sharpes)), 4),
                "oos_return_mean": round(float(np.mean(oos_returns)), 6),
                "oos_max_dd_mean": round(float(np.mean(oos_dds)), 6),
                "oos_trades_mean": round(float(avg_oos_trades), 1),
                "oos_win_rate_mean": round(float(np.mean(oos_wrs)), 4),
                "oos_profit_factor_mean": round(float(np.mean(oos_pfs)), 4),
                "consistency": round(consistency, 4),
                "sharpe_decay": round(sharpe_decay, 4),
                "per_window_oos": per_window_oos,
            })

    sweep_time = time.time() - t0
    logger.info(f"Completed {len(grid)} combos in {sweep_time:.1f}s "
                f"({len(grid)/sweep_time:.0f}/s). "
                f"Valid results: {len(results)}, Errors: {errors}")

    if not results:
        logger.warning("No valid walk-forward results!")
        return

    # Sort: OOS Sharpe (primary), consistency, OOS return
    results.sort(key=lambda r: (r["oos_sharpe_mean"], r["consistency"], r["oos_return_mean"]),
                 reverse=True)

    # Save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} walk-forward results to {output_file}")

    # Print top N
    print(f"\n{'='*120}")
    print(f"  WALK-FORWARD TOP {top_n} — {symbol} {interval} ({len(df)} candles, {len(windows)} windows)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'Strategy':<18} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Ret':>9} "
          f"{'OOS MaxDD':>10} {'Consist':>8} {'Decay':>7} {'Trades':>7} {'Params'}")
    print(f"{'-'*120}")

    for i, r in enumerate(results[:top_n]):
        params_str = str(r["params"])
        if len(params_str) > 40:
            params_str = params_str[:37] + "..."
        print(f"{i+1:>3} {r['strategy']:<18} {r['is_sharpe_mean']:>10.2f} "
              f"{r['oos_sharpe_mean']:>11.2f} {r['oos_return_mean']:>8.2%} "
              f"{r['oos_max_dd_mean']:>9.2%} {r['consistency']:>7.0%} "
              f"{r['sharpe_decay']:>7.2f} {r['oos_trades_mean']:>7.1f} {params_str}")

    print(f"\nTotal combinations tested: {len(grid)}")
    print(f"Combinations with valid OOS: {len(results)}")
    if results:
        print(f"Best OOS Sharpe: {results[0]['oos_sharpe_mean']:.4f} "
              f"({results[0]['strategy']} {results[0]['params']})")

    return results


def run_mega_walkforward(
    symbols: List[str],
    interval: str,
    limit: int,
    output_dir: str,
    sl_tp_levels: List[Tuple[float, float]],
    top_n: int = 50,
    n_windows: int = 5,
    train_pct: float = 0.70,
    workers: int = 1,
):
    """
    Walk-forward sweep across multiple coins with multiple SL/TP levels.
    Signals computed once per strategy+params; backtested per SL/TP per window.
    When workers > 1, coins are processed in parallel (one worker per coin).
    """
    grid = build_param_grid()
    total_per_coin = len(grid) * len(sl_tp_levels) * n_windows * 2
    logger.info(f"=== MEGA WALK-FORWARD: {len(symbols)} coins × {interval} × "
                f"{len(grid)} strategies × {len(sl_tp_levels)} SL/TP × "
                f"{n_windows} windows = ~{total_per_coin} backtests/coin ===")
    if workers > 1:
        logger.info(f"  Using {workers} parallel workers (coin-level parallelism)")

    os.makedirs(output_dir, exist_ok=True)

    if workers > 1:
        coin_args = [
            (sym, interval, limit, output_dir, sl_tp_levels, top_n,
             n_windows, train_pct)
            for sym in symbols
        ]
        n_pool = min(workers, len(symbols))
        with multiprocessing.Pool(n_pool) as pool:
            completed = list(pool.imap_unordered(_mega_wf_coin_worker, coin_args))
        done = [s for s in completed if s is not None]
        logger.info(f"Mega walk-forward complete: {len(done)}/{len(symbols)} coins processed")
    else:
        interval_to_bpd = {
            "1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1,
        }
        bpd = interval_to_bpd.get(interval, 96)

        for sym in symbols:
            logger.info(f"\n--- {sym} {interval} (limit={limit}) ---")
            t_sym = time.time()

            df = fetch_binance_ohlcv(sym, interval=interval, limit=limit)
            if df is None or df.empty:
                logger.error(f"  Failed to fetch {sym}")
                continue
            logger.info(f"  Fetched {len(df)} candles "
                        f"({df.index[0]} → {df.index[-1]})")

            windows = generate_wf_windows(len(df), n_windows, train_pct)
            if not windows:
                logger.error(f"  Not enough data for walk-forward windows on {sym}")
                continue
            logger.info(f"  Generated {len(windows)} walk-forward windows")

            results = []
            errors = 0
            t0 = time.time()

            for idx, (strat_name, params) in enumerate(grid):
                if (idx + 1) % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    logger.info(f"  [{sym}] {idx+1}/{len(grid)} strategies ({rate:.0f}/s)")

                signal_func = STRATEGY_MAP.get(strat_name)
                if signal_func is None:
                    errors += 1
                    continue

                try:
                    signals = signal_func(df, **params)
                except Exception:
                    errors += 1
                    continue

                for sl, tp in sl_tp_levels:
                    is_metrics_list = []
                    oos_metrics_list = []

                    for tr_s, tr_e, te_s, te_e in windows:
                        try:
                            df_train = df.iloc[tr_s:tr_e]
                            sig_train = signals.iloc[tr_s:tr_e]
                            is_m = fast_backtest(df_train, sig_train, bars_per_day=bpd,
                                                 stop_loss=sl, take_profit=tp)
                            df_test = df.iloc[te_s:te_e]
                            sig_test = signals.iloc[te_s:te_e]
                            oos_m = fast_backtest(df_test, sig_test, bars_per_day=bpd,
                                                  stop_loss=sl, take_profit=tp)
                            is_metrics_list.append(is_m)
                            oos_metrics_list.append(oos_m)
                        except Exception:
                            errors += 1

                    if len(oos_metrics_list) < 2:
                        continue

                    oos_sharpes = [m["sharpe"] for m in oos_metrics_list]
                    oos_returns = [m["total_return"] for m in oos_metrics_list]
                    oos_dds = [m["max_drawdown"] for m in oos_metrics_list]
                    oos_trades = [m["trades"] for m in oos_metrics_list]
                    oos_wrs = [m["win_rate"] for m in oos_metrics_list]
                    oos_pfs = [m["profit_factor"] for m in oos_metrics_list]
                    is_sharpes = [m["sharpe"] for m in is_metrics_list]

                    avg_oos_trades = np.mean(oos_trades)
                    if avg_oos_trades < 2:
                        continue

                    is_sharpe_mean = float(np.mean(is_sharpes))
                    oos_sharpe_mean = float(np.mean(oos_sharpes))
                    consistency = float(np.mean([1 if r > 0 else 0 for r in oos_returns]))
                    sharpe_decay = 1.0 - (oos_sharpe_mean / is_sharpe_mean) if is_sharpe_mean != 0 else 1.0

                    per_window_oos = []
                    for i, m in enumerate(oos_metrics_list):
                        per_window_oos.append({
                            "window": i + 1,
                            "sharpe": m["sharpe"],
                            "total_return": m["total_return"],
                            "max_drawdown": m["max_drawdown"],
                            "trades": m["trades"],
                            "win_rate": m["win_rate"],
                            "profit_factor": m["profit_factor"],
                        })

                    results.append({
                        "strategy": strat_name,
                        "params": params,
                        "symbol": sym,
                        "interval": interval,
                        "candles": len(df),
                        "stop_loss": sl,
                        "take_profit": tp,
                        "n_windows": len(oos_metrics_list),
                        "is_sharpe_mean": round(is_sharpe_mean, 4),
                        "oos_sharpe_mean": round(oos_sharpe_mean, 4),
                        "oos_sharpe_std": round(float(np.std(oos_sharpes)), 4),
                        "oos_sharpe_min": round(float(np.min(oos_sharpes)), 4),
                        "oos_return_mean": round(float(np.mean(oos_returns)), 6),
                        "oos_max_dd_mean": round(float(np.mean(oos_dds)), 6),
                        "oos_trades_mean": round(float(avg_oos_trades), 1),
                        "oos_win_rate_mean": round(float(np.mean(oos_wrs)), 4),
                        "oos_profit_factor_mean": round(float(np.mean(oos_pfs)), 4),
                        "consistency": round(consistency, 4),
                        "sharpe_decay": round(sharpe_decay, 4),
                        "per_window_oos": per_window_oos,
                    })

            sweep_time = time.time() - t_sym
            logger.info(f"  [{sym}] Done: {len(results)} valid in {sweep_time:.1f}s")

            if not results:
                continue

            results.sort(key=lambda r: (r["oos_sharpe_mean"], r["consistency"], r["oos_return_mean"]),
                         reverse=True)

            outfile = os.path.join(output_dir, f"wf_{sym}_{interval}.json")
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"  Saved {len(results)} results to {outfile}")

            print(f"\n{'='*120}")
            print(f"  WALK-FORWARD TOP {top_n} — {sym} {interval} ({len(df)} candles)")
            print(f"{'='*120}")
            print(f"{'#':>3} {'Strategy':<18} {'IS Shrp':>8} {'OOS Shrp':>9} {'OOS Ret':>9} "
                  f"{'OOS DD':>8} {'Consist':>8} {'Decay':>7} {'SL/TP':>10} {'Params'}")
            print(f"{'-'*120}")
            for i, r in enumerate(results[:top_n]):
                ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())
                if len(ps) > 35:
                    ps = ps[:32] + "..."
                sl_str = f"{r.get('stop_loss',0)*100:.1f}/{r.get('take_profit',0)*100:.1f}"
                print(f"{i+1:>3} {r['strategy']:<18} {r['is_sharpe_mean']:>8.2f} "
                      f"{r['oos_sharpe_mean']:>9.2f} {r['oos_return_mean']:>8.2%} "
                      f"{r['oos_max_dd_mean']:>7.2%} {r['consistency']:>7.0%} "
                      f"{r['sharpe_decay']:>7.2f} {sl_str:>10} {ps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy parameter sweep")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--sl", type=float, default=0.0, help="Stop-loss pct, e.g. 0.01 = 1%")
    parser.add_argument("--tp", type=float, default=0.0, help="Take-profit pct, e.g. 0.03 = 3%")
    parser.add_argument("--mega", action="store_true", help="Run mega multi-coin sweep")
    parser.add_argument("--walkforward", action="store_true", help="Use walk-forward testing")
    parser.add_argument("--wf-windows", type=int, default=5, help="Number of walk-forward windows")
    parser.add_argument("--wf-train-pct", type=float, default=0.70, help="Train fraction per window")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of CPU workers for parallel execution (default: 1 = sequential)")
    args = parser.parse_args()

    if args.mega:
        COINS = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
        ]
        SL_TP = [
            (0.005, 0.015),  # 0.5% SL / 1.5% TP
            (0.01,  0.03),   # 1% SL / 3% TP
            (0.02,  0.06),   # 2% SL / 6% TP
        ]
        outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data", "sweeps")
        if args.walkforward:
            run_mega_walkforward(COINS, args.interval, args.limit, outdir, SL_TP, args.top,
                                n_windows=args.wf_windows, train_pct=args.wf_train_pct,
                                workers=args.workers)
        else:
            run_mega_sweep(COINS, args.interval, args.limit, outdir, SL_TP, args.top,
                          workers=args.workers)
    else:
        if args.walkforward:
            if args.output is None:
                suffix = ""
                if args.sl > 0 or args.tp > 0:
                    suffix = f"_sl{int(args.sl*100)}tp{int(args.tp*100)}"
                args.output = f"/tmp/wf_{args.symbol}_{args.interval}{suffix}.json"
            run_walkforward_sweep(args.symbol, args.interval, args.limit, args.output, args.top,
                                  stop_loss=args.sl, take_profit=args.tp,
                                  n_windows=args.wf_windows, train_pct=args.wf_train_pct,
                                  workers=args.workers)
        else:
            if args.output is None:
                suffix = ""
                if args.sl > 0 or args.tp > 0:
                    suffix = f"_sl{int(args.sl*100)}tp{int(args.tp*100)}"
                args.output = f"/tmp/sweep_{args.symbol}_{args.interval}{suffix}.json"
            run_sweep(args.symbol, args.interval, args.limit, args.output, args.top,
                      stop_loss=args.sl, take_profit=args.tp, workers=args.workers)
