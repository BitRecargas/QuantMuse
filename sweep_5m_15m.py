#!/usr/bin/env python3
"""Sweep 5m and 15m timeframes with rate limiting."""
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
        if i > 0 and i % 4 == 0:
            time.sleep(1.5)
        try:
            r = run_crypto_backtest(symbols=[sym], strategy_name=strat, initial_capital=100000,
                commission_rate=0.001, interval=tf_name, limit=limit, strategy_params=params)
            if 'error' not in r and r['total_trades'] > 0:
                results.append((strat[:9], sym, label, r))
        except:
            pass

    profitable = [(s, sym, p, r) for s, sym, p, r in results if r['total_return'] > 0]
    profitable.sort(key=lambda x: x[3]['total_return'], reverse=True)

    print(f'Tested {len(results)} | {len(profitable)} profitable')
    print(f'{"Strategy":>10s} {"Symbol":>8s} {"Params":>25s} | {"Return":>8s} {"MaxDD":>8s} {"WR":>6s} {"PF":>6s} {"Trades":>6s}')
    print('-' * 90)

    show = profitable[:20] if profitable else sorted(results, key=lambda x: x[3]['total_return'], reverse=True)[:10]
    for strat, sym, params, r in show:
        print(f"{strat:>10s} {sym:>8s} {params:>25s} | {r['total_return']:>+7.2%} {r['max_drawdown']:>7.2%} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} {r['total_trades']:>6d}")

# ---- 5M (use 1000 candles = ~3.5 days, no pagination needed) ----
configs_5m = []
for sym in symbols:
    for lb in [12, 24, 48, 96, 144, 288]:
        configs_5m.append(('Momentum', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for rsi in [12, 24, 48, 96]:
        for os_val, ob_val in [(20,80),(25,75),(30,70),(15,85)]:
            configs_5m.append(('Mean Reversion', sym, {'rsi_period': rsi, 'oversold': float(os_val), 'overbought': float(ob_val)}, f'rsi={rsi} os={os_val} ob={ob_val}'))
for sym in symbols:
    for lb in [24, 48, 96, 144, 288]:
        configs_5m.append(('Breakout', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for mom in [48, 96, 144, 288]:
        for rsi in [12, 24, 48]:
            configs_5m.append(('Multi-Factor', sym, {'mom_lookback': mom, 'rsi_period': rsi}, f'mom={mom} rsi={rsi}'))

run_sweep('5m', 1000, configs_5m)

# ---- 15M (use 1000 candles = ~10.4 days, no pagination needed) ----
configs_15m = []
for sym in symbols:
    for lb in [8, 16, 24, 48, 96, 192]:
        configs_15m.append(('Momentum', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for rsi in [8, 16, 24, 48, 96]:
        for os_val, ob_val in [(20,80),(25,75),(30,70),(15,85)]:
            configs_15m.append(('Mean Reversion', sym, {'rsi_period': rsi, 'oversold': float(os_val), 'overbought': float(ob_val)}, f'rsi={rsi} os={os_val} ob={ob_val}'))
for sym in symbols:
    for lb in [16, 24, 48, 96, 192]:
        configs_15m.append(('Breakout', sym, {'lookback': lb}, f'lb={lb}'))
for sym in symbols:
    for mom in [24, 48, 96, 192]:
        for rsi in [8, 16, 24, 48]:
            configs_15m.append(('Multi-Factor', sym, {'mom_lookback': mom, 'rsi_period': rsi}, f'mom={mom} rsi={rsi}'))

run_sweep('15m', 1000, configs_15m)

print('\nDONE')
