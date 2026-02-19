#!/usr/bin/env python3
"""Sweep 1h and 4h timeframes with rate limiting."""
import sys, time
sys.path.insert(0, '/app')
from data_service.backtest.crypto_backtest import run_crypto_backtest

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

def run_sweep(tf_name, limit, configs):
    print(f'\n{"="*100}')
    print(f'TIMEFRAME: {tf_name} | {limit} candles')
    print(f'{"="*100}')

    results = []
    for i, (strat, sym, params, label) in enumerate(configs):
        if i > 0 and i % 5 == 0:
            time.sleep(2)  # rate limit
        try:
            r = run_crypto_backtest(symbols=[sym], strategy_name=strat, initial_capital=100000,
                commission_rate=0.001, interval=tf_name, limit=limit, strategy_params=params)
            if 'error' not in r and r['total_trades'] > 0:
                results.append((strat[:9], sym, label, r))
        except Exception as e:
            pass

    profitable = [(s, sym, p, r) for s, sym, p, r in results if r['total_return'] > 0]
    profitable.sort(key=lambda x: x[3]['total_return'], reverse=True)

    print(f'Tested {len(results)} | {len(profitable)} profitable')
    print(f'{"Strategy":>10s} {"Symbol":>8s} {"Params":>25s} | {"Return":>8s} {"MaxDD":>8s} {"WR":>6s} {"PF":>6s} {"Trades":>6s}')
    print('-' * 90)

    show = profitable[:20] if profitable else sorted(results, key=lambda x: x[3]['total_return'], reverse=True)[:10]
    for strat, sym, params, r in show:
        print(f"{strat:>10s} {sym:>8s} {params:>25s} | {r['total_return']:>+7.2%} {r['max_drawdown']:>7.2%} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} {r['total_trades']:>6d}")

# ---- 1H ----
configs_1h = []
for sym in symbols:
    for lb in [6, 12, 24, 48, 72, 120, 168]:
        configs_1h.append(('Momentum', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for rsi in [6, 12, 24, 48]:
        for os_val, ob_val in [(20,80),(25,75),(30,70)]:
            configs_1h.append(('Mean Reversion', sym, {'rsi_period': rsi, 'oversold': float(os_val), 'overbought': float(ob_val)}, f'rsi={rsi} os={os_val} ob={ob_val}'))
for sym in symbols:
    for lb in [12, 24, 48, 72, 120, 168]:
        configs_1h.append(('Breakout', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for mom in [24, 48, 72, 120, 168]:
        for rsi in [6, 12, 24, 48]:
            configs_1h.append(('Multi-Factor', sym, {'mom_lookback': mom, 'rsi_period': rsi}, f'mom={mom} rsi={rsi}'))

run_sweep('1h', 2000, configs_1h)  # ~83 days, less pagination

# ---- 4H ----
configs_4h = []
for sym in symbols:
    for lb in [6, 12, 18, 24, 42, 84, 126, 168]:
        configs_4h.append(('Momentum', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for rsi in [6, 12, 18, 24, 42]:
        for os_val, ob_val in [(20,80),(25,75),(30,70)]:
            configs_4h.append(('Mean Reversion', sym, {'rsi_period': rsi, 'oversold': float(os_val), 'overbought': float(ob_val)}, f'rsi={rsi} os={os_val} ob={ob_val}'))
for sym in symbols:
    for lb in [6, 12, 24, 42, 84, 126]:
        configs_4h.append(('Breakout', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for mom in [12, 24, 42, 84, 126]:
        for rsi in [6, 12, 18, 24]:
            configs_4h.append(('Multi-Factor', sym, {'mom_lookback': mom, 'rsi_period': rsi}, f'mom={mom} rsi={rsi}'))

run_sweep('4h', 2000, configs_4h)  # ~333 days

print('\nDONE')
