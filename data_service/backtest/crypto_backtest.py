"""
Real crypto backtesting with Binance data.
Provides strategy functions and a backtest runner with proper mark-to-market equity tracking.
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def fetch_binance_ohlcv(symbol: str, interval: str = "1d", limit: int = 365) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from Binance public API. Supports up to ~5 years via pagination."""
    pair = symbol.upper()
    if not pair.endswith(("USDT", "BTC", "ETH", "BUSD")):
        pair += "USDT"

    try:
        all_klines = []
        remaining = limit
        end_time = None  # None = latest

        while remaining > 0:
            batch = min(remaining, 1000)
            url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit={batch}"
            if end_time is not None:
                url += f"&endTime={end_time}"

            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.error(f"Binance API error {resp.status_code} for {pair}")
                break
            klines = resp.json()
            if not klines:
                break

            all_klines = klines + all_klines  # prepend older data
            remaining -= len(klines)

            if len(klines) < batch:
                break  # no more data available

            # Next batch ends just before the earliest candle we got
            end_time = klines[0][0] - 1

        if not all_klines:
            return None

        rows = []
        for k in all_klines:
            dt = datetime.utcfromtimestamp(k[0] / 1000)
            rows.append({
                "date": dt,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(rows).set_index("date")
        df = df[~df.index.duplicated(keep='first')].sort_index()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch Binance data for {pair}: {e}")
        return None


# ---------------------------------------------------------------------------
# Strategy signal generators
# Each returns a Series of signals: 1 = buy, -1 = sell, 0 = hold
# ---------------------------------------------------------------------------

def _momentum_signals(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Buy when close > SMA, sell when close < SMA."""
    sma = df["close"].rolling(lookback).mean()
    signal = pd.Series(0, index=df.index)
    signal[df["close"] > sma] = 1
    signal[df["close"] < sma] = -1
    signal.iloc[:lookback] = 0
    return signal


def _mean_reversion_signals(df: pd.DataFrame, rsi_period: int = 14,
                            oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    """Buy when RSI < oversold, sell when RSI > overbought."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signal = pd.Series(0, index=df.index)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    signal.iloc[:rsi_period] = 0
    return signal


def _breakout_signals(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Buy on new high breakout, sell on new low breakdown (Donchian channel)."""
    # Shift by 1 so we compare today's close to yesterday's channel
    high_ch = df["high"].rolling(lookback).max().shift(1)
    low_ch = df["low"].rolling(lookback).min().shift(1)

    signal = pd.Series(0, index=df.index)
    signal[df["close"] > high_ch] = 1
    signal[df["close"] < low_ch] = -1
    signal.iloc[:lookback + 1] = 0
    return signal


def _multi_factor_signals(df: pd.DataFrame, mom_lookback: int = 20,
                          rsi_period: int = 14) -> pd.Series:
    """Combine momentum + mean-reversion: buy when momentum agrees or RSI oversold."""
    mom = _momentum_signals(df, mom_lookback)
    mr = _mean_reversion_signals(df, rsi_period)
    combined = mom + mr  # range -2..+2
    signal = pd.Series(0, index=df.index)
    signal[combined >= 1] = 1   # at least one factor says buy
    signal[combined <= -2] = -1  # both say sell
    # Also sell when momentum turns negative
    signal[(mom == -1) & (mr != 1)] = -1
    return signal


# ---------------------------------------------------------------------------
# Additional strategy signal generators for low-timeframe optimization
# ---------------------------------------------------------------------------

def _ema_crossover_signals(df: pd.DataFrame, fast_period: int = 8, slow_period: int = 21) -> pd.Series:
    """Buy when fast EMA crosses above slow EMA, sell on cross below."""
    fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow = df["close"].ewm(span=slow_period, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[fast > slow] = 1
    signal[fast < slow] = -1
    warmup = max(fast_period, slow_period)
    signal.iloc[:warmup] = 0
    return signal


def _macd_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.Series:
    """Buy when MACD line crosses above signal line, sell on cross below."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[macd_line > signal_line] = 1
    signal[macd_line < signal_line] = -1
    signal.iloc[:slow + signal_period] = 0
    return signal


def _bollinger_signals(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Mean reversion: buy below lower band, sell above upper band."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower] = 1
    signal[df["close"] > upper] = -1
    signal.iloc[:period] = 0
    return signal


def _stochastic_signals(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                         oversold: float = 20.0, overbought: float = 80.0) -> pd.Series:
    """Stochastic oscillator: buy when %K crosses up from oversold, sell from overbought."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    signal = pd.Series(0, index=df.index)
    signal[(k < oversold) & (k > d)] = 1
    signal[(k > overbought) & (k < d)] = -1
    signal.iloc[:k_period + d_period] = 0
    return signal


def _keltner_signals(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14,
                      atr_mult: float = 2.0) -> pd.Series:
    """Keltner Channel breakout: buy above upper band, sell below lower band."""
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    signal = pd.Series(0, index=df.index)
    signal[df["close"] > upper] = 1
    signal[df["close"] < lower] = -1
    warmup = max(ema_period, atr_period)
    signal.iloc[:warmup] = 0
    return signal


def _williams_r_signals(df: pd.DataFrame, period: int = 14,
                         oversold: float = -80.0, overbought: float = -20.0) -> pd.Series:
    """Williams %R: buy when oversold, sell when overbought."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    wr = -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
    signal = pd.Series(0, index=df.index)
    signal[wr < oversold] = 1
    signal[wr > overbought] = -1
    signal.iloc[:period] = 0
    return signal


def _cci_signals(df: pd.DataFrame, period: int = 20,
                  buy_level: float = -100.0, sell_level: float = 100.0) -> pd.Series:
    """Commodity Channel Index: buy below buy_level, sell above sell_level."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad).replace(0, np.nan)
    signal = pd.Series(0, index=df.index)
    signal[cci < buy_level] = 1
    signal[cci > sell_level] = -1
    signal.iloc[:period] = 0
    return signal


def _triple_ema_signals(df: pd.DataFrame, fast: int = 5, mid: int = 15, slow: int = 30) -> pd.Series:
    """Triple EMA alignment: buy when fast > mid > slow, sell when fast < mid < slow."""
    ema_f = df["close"].ewm(span=fast, adjust=False).mean()
    ema_m = df["close"].ewm(span=mid, adjust=False).mean()
    ema_s = df["close"].ewm(span=slow, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[(ema_f > ema_m) & (ema_m > ema_s)] = 1
    signal[(ema_f < ema_m) & (ema_m < ema_s)] = -1
    signal.iloc[:slow] = 0
    return signal


def _rsi_ema_signals(df: pd.DataFrame, rsi_period: int = 14, ema_period: int = 50,
                      oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    """RSI for entry signal, EMA for trend filter. Buy: RSI oversold + above EMA. Sell: RSI overbought + below EMA."""
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # EMA trend filter
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[(rsi < oversold) & (df["close"] > ema)] = 1
    signal[(rsi > overbought) & (df["close"] < ema)] = -1
    warmup = max(rsi_period, ema_period)
    signal.iloc[:warmup] = 0
    return signal


def _supertrend_signals(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Supertrend indicator: trend-following using ATR bands."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    hl2 = (df["high"] + df["low"]) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 = up, -1 = down

    for i in range(atr_period, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

    signal = pd.Series(0, index=df.index)
    signal[direction == 1] = 1
    signal[direction == -1] = -1
    signal.iloc[:atr_period] = 0
    return signal


# ---------------------------------------------------------------------------
# Additional strategies for expanded sweep
# ---------------------------------------------------------------------------

def _adx_trend_signals(df: pd.DataFrame, adx_period: int = 14, di_period: int = 14, threshold: float = 25.0) -> pd.Series:
    """ADX trend strength + DI crossover. Buy when +DI > -DI and ADX > threshold."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out where other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(di_period).mean()
    plus_di = 100 * (plus_dm.rolling(di_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(di_period).mean() / atr)
    dx = (100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)))
    adx = dx.rolling(adx_period).mean()
    signal = pd.Series(0, index=df.index)
    signal[(plus_di > minus_di) & (adx > threshold)] = 1
    signal[(minus_di > plus_di) & (adx > threshold)] = -1
    signal.iloc[:di_period + adx_period] = 0
    return signal


def _parabolic_sar_signals(df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR stop-and-reverse."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)
    sar = np.zeros(n)
    direction = np.ones(n)  # 1=long, -1=short
    af = af_start
    ep = low[0]
    sar[0] = high[0]
    for i in range(1, n):
        if direction[i - 1] == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], low[i - 1], low[max(0, i - 2)])
            if low[i] < sar[i]:
                direction[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                direction[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_increment, af_max)
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], high[i - 1], high[max(0, i - 2)])
            if high[i] > sar[i]:
                direction[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                direction[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_increment, af_max)
    signal = pd.Series(0, index=df.index)
    signal[pd.Series(direction, index=df.index) == 1] = 1
    signal[pd.Series(direction, index=df.index) == -1] = -1
    return signal


def _hull_ma_signals(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Hull Moving Average direction. HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    close = df["close"]
    half_n = max(int(period / 2), 1)
    sqrt_n = max(int(np.sqrt(period)), 1)
    wma_half = close.rolling(half_n).mean()
    wma_full = close.rolling(period).mean()
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(sqrt_n).mean()
    signal = pd.Series(0, index=df.index)
    signal[hma > hma.shift(1)] = 1
    signal[hma < hma.shift(1)] = -1
    signal.iloc[:period + sqrt_n] = 0
    return signal


def _donchian_atr_signals(df: pd.DataFrame, donch_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0) -> pd.Series:
    """Donchian breakout with ATR-based trailing stop."""
    high, low, close = df["high"], df["low"], df["close"]
    upper = high.rolling(donch_period).max()
    lower = low.rolling(donch_period).min()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    signal = pd.Series(0, index=df.index)
    signal[close > upper.shift(1)] = 1
    signal[close < (upper.shift(1) - atr_mult * atr)] = -1
    signal.iloc[:max(donch_period, atr_period)] = 0
    return signal


def _double_rsi_signals(df: pd.DataFrame, fast_rsi: int = 5, slow_rsi: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    """Double RSI: fast RSI for entries, slow RSI for trend filter."""
    close = df["close"]
    def _rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - 100 / (1 + rs)
    rsi_fast = _rsi(close, fast_rsi)
    rsi_slow = _rsi(close, slow_rsi)
    signal = pd.Series(0, index=df.index)
    signal[(rsi_fast < oversold) & (rsi_slow < 50)] = 1
    signal[(rsi_fast > overbought) & (rsi_slow > 50)] = -1
    signal.iloc[:slow_rsi + 1] = 0
    return signal


def _vwap_signals(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """VWAP bands mean reversion. Buy below lower band, sell above upper band."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    cum_tp_vol = (tp * vol).rolling(period).sum()
    cum_vol = vol.rolling(period).sum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    vwap_std = tp.rolling(period).std()
    upper = vwap + num_std * vwap_std
    lower = vwap - num_std * vwap_std
    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower] = 1
    signal[df["close"] > upper] = -1
    signal.iloc[:period] = 0
    return signal


def _ichimoku_signals(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.Series:
    """Ichimoku Cloud: tenkan/kijun cross with cloud confirmation."""
    high, low, close = df["high"], df["low"], df["close"]
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = (tenkan_sen + kijun_sen) / 2
    senkou_b_line = (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    cloud_top = pd.concat([senkou_a, senkou_b_line], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b_line], axis=1).min(axis=1)
    signal = pd.Series(0, index=df.index)
    signal[(tenkan_sen > kijun_sen) & (close > cloud_top)] = 1
    signal[(tenkan_sen < kijun_sen) & (close < cloud_bot)] = -1
    signal.iloc[:senkou_b] = 0
    return signal


def _trix_signals(df: pd.DataFrame, period: int = 15, signal_period: int = 9) -> pd.Series:
    """TRIX: triple-smoothed EMA rate of change with signal line."""
    close = df["close"]
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ema3.pct_change() * 100
    trix_signal = trix.ewm(span=signal_period, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[trix > trix_signal] = 1
    signal[trix < trix_signal] = -1
    signal.iloc[:period * 3] = 0
    return signal


def _aroon_signals(df: pd.DataFrame, period: int = 25, threshold: float = 70.0) -> pd.Series:
    """Aroon oscillator: trend detection via high/low position within lookback."""
    high, low = df["high"], df["low"]
    aroon_up = high.rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    aroon_down = low.rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    signal = pd.Series(0, index=df.index)
    signal[(aroon_up > threshold) & (aroon_down < (100 - threshold))] = 1
    signal[(aroon_down > threshold) & (aroon_up < (100 - threshold))] = -1
    signal.iloc[:period + 1] = 0
    return signal


def _mfi_signals(df: pd.DataFrame, period: int = 14, oversold: float = 20.0, overbought: float = 80.0) -> pd.Series:
    """Money Flow Index: volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0).rolling(period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    signal = pd.Series(0, index=df.index)
    signal[mfi < oversold] = 1
    signal[mfi > overbought] = -1
    signal.iloc[:period + 1] = 0
    return signal


def _squeeze_signals(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """Squeeze Momentum: detects Bollinger inside Keltner, trades breakout direction."""
    close, high, low = df["close"], df["high"], df["low"]
    bb_mid = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_mid = close.ewm(span=kc_period, adjust=False).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_off = ~squeeze_on
    mom = close - close.rolling(bb_period).mean()
    signal = pd.Series(0, index=df.index)
    signal[squeeze_off & (mom > 0) & (mom > mom.shift())] = 1
    signal[squeeze_off & (mom < 0) & (mom < mom.shift())] = -1
    signal.iloc[:max(bb_period, kc_period)] = 0
    return signal


def _elder_ray_signals(df: pd.DataFrame, ema_period: int = 13, threshold: float = 0.0) -> pd.Series:
    """Elder Ray: bull/bear power with EMA trend filter."""
    close = df["close"]
    ema = close.ewm(span=ema_period, adjust=False).mean()
    bull_power = df["high"] - ema
    bear_power = df["low"] - ema
    signal = pd.Series(0, index=df.index)
    signal[(ema > ema.shift()) & (bear_power < 0) & (bear_power > bear_power.shift())] = 1
    signal[(ema < ema.shift()) & (bull_power > 0) & (bull_power < bull_power.shift())] = -1
    signal.iloc[:ema_period] = 0
    return signal


def _cmf_signals(df: pd.DataFrame, period: int = 20, threshold: float = 0.05) -> pd.Series:
    """Chaikin Money Flow: volume accumulation/distribution oscillator."""
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    hl_range = high - low
    clv = ((close - low) - (high - close)) / (hl_range + 1e-10)
    mf_vol = clv * vol
    cmf = mf_vol.rolling(period).sum() / (vol.rolling(period).sum() + 1e-10)
    signal = pd.Series(0, index=df.index)
    signal[cmf > threshold] = 1
    signal[cmf < -threshold] = -1
    signal.iloc[:period] = 0
    return signal


def _fisher_signals(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Fisher Transform: normalizes price to Gaussian, signal on direction change."""
    hl2 = (df["high"] + df["low"]) / 2
    highest = hl2.rolling(period).max()
    lowest = hl2.rolling(period).min()
    raw = 2 * ((hl2 - lowest) / (highest - lowest + 1e-10)) - 1
    raw = raw.clip(-0.999, 0.999)
    # Smoothed
    val = raw.ewm(span=5, adjust=False).mean().clip(-0.999, 0.999)
    fisher = (np.log((1 + val) / (1 - val))) / 2
    signal = pd.Series(0, index=df.index)
    signal[(fisher > fisher.shift()) & (fisher.shift() <= fisher.shift(2))] = 1
    signal[(fisher < fisher.shift()) & (fisher.shift() >= fisher.shift(2))] = -1
    signal.iloc[:period + 2] = 0
    return signal


def _kst_signals(df: pd.DataFrame, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30, signal_period: int = 9) -> pd.Series:
    """KST (Know Sure Thing): weighted sum of ROC at different periods."""
    close = df["close"]
    roc1 = close.pct_change(r1).rolling(10).mean()
    roc2 = close.pct_change(r2).rolling(10).mean()
    roc3 = close.pct_change(r3).rolling(10).mean()
    roc4 = close.pct_change(r4).rolling(15).mean()
    kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
    kst_signal = kst.rolling(signal_period).mean()
    signal = pd.Series(0, index=df.index)
    signal[kst > kst_signal] = 1
    signal[kst < kst_signal] = -1
    signal.iloc[:r4 + 15] = 0
    return signal


STRATEGY_MAP = {
    "Momentum Strategy": _momentum_signals,
    "Momentum": _momentum_signals,
    "Mean Reversion": _mean_reversion_signals,
    "Breakout": _breakout_signals,
    "Multi-Factor": _multi_factor_signals,
    "EMA Crossover": _ema_crossover_signals,
    "MACD": _macd_signals,
    "Bollinger Bands": _bollinger_signals,
    "Stochastic": _stochastic_signals,
    "Keltner Channel": _keltner_signals,
    "Williams %R": _williams_r_signals,
    "CCI": _cci_signals,
    "Triple EMA": _triple_ema_signals,
    "RSI+EMA": _rsi_ema_signals,
    "Supertrend": _supertrend_signals,
    "ADX Trend": _adx_trend_signals,
    "Parabolic SAR": _parabolic_sar_signals,
    "Hull MA": _hull_ma_signals,
    "Donchian ATR": _donchian_atr_signals,
    "Double RSI": _double_rsi_signals,
    "VWAP Bands": _vwap_signals,
    "Ichimoku": _ichimoku_signals,
    "TRIX": _trix_signals,
    "Aroon": _aroon_signals,
    "MFI": _mfi_signals,
    "Squeeze": _squeeze_signals,
    "Elder Ray": _elder_ray_signals,
    "CMF": _cmf_signals,
    "Fisher Transform": _fisher_signals,
    "KST": _kst_signals,
}


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_crypto_backtest(
    symbols: List[str],
    strategy_name: str = "Momentum Strategy",
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001,
    interval: str = "1d",
    limit: int = 365,
    strategy_params: Optional[Dict[str, Any]] = None,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    leverage: int = 1,
) -> Dict[str, Any]:
    """
    Run a real backtest on Binance data.

    Args:
        leverage: Futures leverage multiplier (1 = spot-equivalent, >1 = futures).
            Position sizing uses margin * leverage. Equity tracks margin, not notional.
            At leverage=1 all divisions are no-ops — fully backward compatible.

    Returns dict with: equity_curve (list of {date, value}), trades, metrics, etc.
    """
    if strategy_params is None:
        strategy_params = {}
    leverage = max(1, leverage)  # safety clamp
    maint_margin_rate = 0.005    # 0.5% of notional (Binance-like)

    # 1. Fetch data for all symbols
    all_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_binance_ohlcv(sym, interval=interval, limit=limit)
        if df is not None and not df.empty:
            all_data[sym] = df

    if not all_data:
        return {"error": "Could not fetch data for any symbol"}

    # Use the first symbol with data as primary (single-asset backtest per symbol, then combine)
    # For simplicity, run single-asset backtest on first symbol if only one, else equal-weight
    n_assets = len(all_data)
    capital_per_asset = initial_capital / n_assets

    # Get signal function
    signal_func = STRATEGY_MAP.get(strategy_name, _momentum_signals)

    # Track combined portfolio
    combined_equity = {}
    all_trades = []

    for sym, df in all_data.items():
        signals = signal_func(df, **strategy_params)
        cash = capital_per_asset
        position_qty = 0.0   # positive = long, negative = short
        position_entry = 0.0
        trades = []
        pending_signal = 0  # signal from previous candle, to execute this candle
        use_sltp = stop_loss > 0 or take_profit > 0
        liquidation_count = 0

        for i in range(len(df)):
            ts = df.index[i]
            price = df["close"].iloc[i]
            exec_price = df["open"].iloc[i]  # trades execute at this candle's open
            bar_high = df["high"].iloc[i]
            bar_low = df["low"].iloc[i]

            # --- Liquidation check (before SL/TP) ---
            if leverage > 1 and position_qty != 0:
                abs_qty = abs(position_qty)
                margin_locked = abs_qty * position_entry / leverage
                if position_qty > 0:
                    worst_price = bar_low
                    unrealized_liq = (worst_price - position_entry) * abs_qty
                else:
                    worst_price = bar_high
                    unrealized_liq = (position_entry - worst_price) * abs_qty
                maintenance_margin = abs_qty * worst_price * maint_margin_rate
                if margin_locked + unrealized_liq <= maintenance_margin:
                    # Liquidated — lose all margin
                    trades.append({
                        "timestamp": ts.isoformat(), "symbol": sym,
                        "side": "liquidation", "quantity": round(abs_qty, 6),
                        "price": round(worst_price, 2), "commission": 0.0,
                        "pnl": round(-margin_locked, 2),
                    })
                    cash += 0.0  # margin is lost
                    position_qty = 0.0
                    position_entry = 0.0
                    pending_signal = 0
                    liquidation_count += 1
                    # Skip SL/TP and signal execution for this bar
                    equity_val = cash
                    ts_str = ts.strftime("%Y-%m-%d %H:%M")
                    combined_equity.setdefault(ts_str, 0.0)
                    combined_equity[ts_str] += equity_val
                    pending_signal = signals.iloc[i]
                    continue

            # --- Check Stop-Loss / Take-Profit using candle high/low ---
            if use_sltp and position_qty != 0:
                if position_qty > 0:  # LONG position
                    sl_price = position_entry * (1 - stop_loss)
                    tp_price = position_entry * (1 + take_profit)
                    sl_hit = bar_low <= sl_price
                    tp_hit = bar_high >= tp_price
                else:  # SHORT position
                    sl_price = position_entry * (1 + stop_loss)
                    tp_price = position_entry * (1 - take_profit)
                    sl_hit = bar_high >= sl_price
                    tp_hit = bar_low <= tp_price

                if sl_hit or tp_hit:
                    # If both trigger same bar, assume SL hit first (conservative)
                    exit_price = sl_price if sl_hit else tp_price
                    side_label = "sl_close_long" if position_qty > 0 else "sl_close_short"
                    if not sl_hit and tp_hit:
                        side_label = "tp_close_long" if position_qty > 0 else "tp_close_short"
                    qty_abs = abs(position_qty)
                    if position_qty > 0:
                        pnl = (exit_price - position_entry) * qty_abs
                    else:
                        pnl = (position_entry - exit_price) * qty_abs
                    comm = exit_price * qty_abs * commission_rate
                    pnl -= comm
                    cash += qty_abs * position_entry / leverage + pnl
                    trades.append({
                        "timestamp": ts.isoformat(), "symbol": sym,
                        "side": side_label, "quantity": round(qty_abs, 6),
                        "price": round(exit_price, 2), "commission": round(comm, 2),
                        "pnl": round(pnl, 2),
                    })
                    position_qty = 0.0
                    position_entry = 0.0
                    pending_signal = 0  # cancel any pending signal after SL/TP exit

            # Mark-to-market (futures: margin + unrealized PnL)
            if position_qty > 0:
                unrealized = (price - position_entry) * position_qty
                equity_val = cash + position_qty * position_entry / leverage + unrealized
            elif position_qty < 0:
                unrealized = (position_entry - price) * abs(position_qty)
                equity_val = cash + abs(position_qty) * position_entry / leverage + unrealized
            else:
                equity_val = cash

            # Record equity
            ts_str = ts.strftime("%Y-%m-%d %H:%M")
            combined_equity.setdefault(ts_str, 0.0)
            combined_equity[ts_str] += equity_val

            # Execute pending signal at this candle's open (avoids look-ahead bias)
            # --- Go LONG ---
            if pending_signal == 1 and position_qty <= 0:
                # Close short first
                if position_qty < 0:
                    qty_abs = abs(position_qty)
                    pnl = (position_entry - exec_price) * qty_abs
                    comm = exec_price * qty_abs * commission_rate
                    pnl -= comm
                    cash += qty_abs * position_entry / leverage + pnl
                    trades.append({
                        "timestamp": ts.isoformat(), "symbol": sym,
                        "side": "close_short", "quantity": round(qty_abs, 6),
                        "price": round(exec_price, 2), "commission": round(comm, 2),
                        "pnl": round(pnl, 2),
                    })
                    position_qty = 0.0
                    position_entry = 0.0

                # Open long (margin-based: deduct margin, not notional)
                usable = cash * 0.95
                comm = usable * commission_rate / (1 + commission_rate)
                margin = usable - comm
                qty = margin * leverage / exec_price
                if qty > 0:
                    position_qty = qty
                    position_entry = exec_price
                    cash -= (margin + comm)
                    trades.append({
                        "timestamp": ts.isoformat(), "symbol": sym,
                        "side": "open_long", "quantity": round(qty, 6),
                        "price": round(exec_price, 2), "commission": round(comm, 2),
                    })

            # --- Go SHORT ---
            elif pending_signal == -1 and position_qty >= 0:
                # Close long first
                if position_qty > 0:
                    pnl = (exec_price - position_entry) * position_qty
                    comm = exec_price * position_qty * commission_rate
                    pnl -= comm
                    cash += position_qty * position_entry / leverage + pnl
                    trades.append({
                        "timestamp": ts.isoformat(), "symbol": sym,
                        "side": "close_long", "quantity": round(position_qty, 6),
                        "price": round(exec_price, 2), "commission": round(comm, 2),
                        "pnl": round(pnl, 2),
                    })
                    position_qty = 0.0
                    position_entry = 0.0

                # Open short (margin-based: deduct margin, not notional)
                usable = cash * 0.95
                comm = usable * commission_rate / (1 + commission_rate)
                margin = usable - comm
                qty = margin * leverage / exec_price
                if qty > 0:
                    position_qty = -qty
                    position_entry = exec_price
                    cash -= (margin + comm)
                    trades.append({
                        "timestamp": ts.isoformat(), "symbol": sym,
                        "side": "open_short", "quantity": round(qty, 6),
                        "price": round(exec_price, 2), "commission": round(comm, 2),
                    })

            # Store this candle's signal to execute on next candle's open
            pending_signal = signals.iloc[i]

        all_trades.extend(trades)

    # Build equity curve
    equity_dates = sorted(combined_equity.keys())
    equity_values = [combined_equity[d] for d in equity_dates]

    if len(equity_values) < 2:
        return {"error": "Not enough data points for backtest"}

    equity_series = pd.Series(equity_values, index=pd.to_datetime(equity_dates))

    # Calculate metrics
    total_return = (equity_values[-1] - initial_capital) / initial_capital

    # Resample to true daily equity for proper volatility/Sharpe calculation
    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    total_days = (equity_series.index[-1] - equity_series.index[0]).days
    annualized_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1

    # Standard Sharpe: mean(daily_ret) / std(daily_ret) * sqrt(365)
    # Both numerator and denominator scale consistently
    volatility = daily_returns.std() * np.sqrt(365) if len(daily_returns) > 1 else 0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0.0

    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = float(drawdown.min())

    total_trades = len(all_trades)
    close_trades = [t for t in all_trades if t["side"] in ("sell", "close_long", "close_short", "sl_close_long", "sl_close_short", "tp_close_long", "tp_close_short", "liquidation")]
    winning_trades = [t for t in close_trades if t.get("pnl", 0) > 0]
    win_rate = len(winning_trades) / len(close_trades) if close_trades else 0.0

    avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
    losing_trades = [t for t in close_trades if t.get("pnl", 0) <= 0]
    avg_loss = abs(np.mean([t["pnl"] for t in losing_trades])) if losing_trades else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0

    # Sortino: mean(daily_ret) / downside_std * sqrt(365)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino_ratio = (daily_returns.mean() / downside.std()) * np.sqrt(365)
    else:
        sortino_ratio = 0.0

    # Calmar: annualized return / max drawdown
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Aggregate liquidation counts across all symbols
    total_liquidations = sum(
        1 for t in all_trades if t.get("side") == "liquidation"
    )

    return {
        "strategy_name": strategy_name,
        "symbols": symbols,
        "initial_capital": initial_capital,
        "leverage": leverage,
        "liquidation_count": total_liquidations,
        "final_value": round(equity_values[-1], 2),
        "total_return": round(total_return, 4),
        "annualized_return": round(annualized_return, 4),
        "volatility": round(volatility, 4),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar_ratio, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "equity_curve": [
            {"date": d, "value": round(v, 2)}
            for d, v in zip(equity_dates, equity_values)
        ],
        "trades": all_trades,
        "drawdown_series": [
            {"date": d.strftime("%Y-%m-%d"), "drawdown": round(float(dd), 4)}
            for d, dd in zip(drawdown.index, drawdown.values)
        ],
    }
