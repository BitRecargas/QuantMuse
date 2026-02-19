#!/usr/bin/env python3
"""Sweep intraday timeframes for profitable strategies."""
import sys
sys.path.insert(0, '/app')
from data_service.backtest.crypto_backtest import run_crypto_backtest

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

timeframes = [
    ('5m',  5000, [12,24,48,96,144,288], [12,24,48,96], [48,96,144,288], [48,96,144,288], [12,24,48]),
    ('15m', 5000, [8,16,24,48,96,192],   [8,16,24,48,96], [16,24,48,96,192], [24,48,96,192], [8,16,24,48]),
    ('1h',  4000, [6,12,24,48,72,120,168], [6,12,24,48,72], [12,24,48,72,120,168], [24,48,72,120,168], [6,12,24,48]),
    ('4h',  4000, [6,12,18,24,42,84,126,168], [6,12,18,24,42], [6,12,24,42,84,126], [12,24,42,84,126], [6,12,18,24]),
]

for tf_name, limit, mom_lbs, mr_rsis, bo_lbs, mf_moms, mf_rsis in timeframes:
    print(f'\n{"="*100}')
    print(f'TIMEFRAME: {tf_name} | {limit} candles')
    print(f'{"="*100}')

    results = []

    # Momentum
    for sym in symbols:
        for lb in mom_lbs:
            try:
                r = run_crypto_backtest(symbols=[sym], strategy_name='Momentum', initial_capital=100000,
                    commission_rate=0.001, interval=tf_name, limit=limit, strategy_params={'lookback': lb})
                if 'error' not in r and r['total_trades'] > 0:
                    results.append(('Momentum', sym, f'lb={lb}', r))
            except: pass

    # Mean Reversion
    for sym in symbols:
        for rsi in mr_rsis:
            for os_val, ob_val in [(20,80),(25,75),(30,70)]:
                try:
                    r = run_crypto_backtest(symbols=[sym], strategy_name='Mean Reversion', initial_capital=100000,
                        commission_rate=0.001, interval=tf_name, limit=limit,
                        strategy_params={'rsi_period': rsi, 'oversold': float(os_val), 'overbought': float(ob_val)})
                    if 'error' not in r and r['total_trades'] > 0:
                        results.append(('MeanRev', sym, f'rsi={rsi} os={os_val} ob={ob_val}', r))
                except: pass

    # Breakout
    for sym in symbols:
        for lb in bo_lbs:
            try:
                r = run_crypto_backtest(symbols=[sym], strategy_name='Breakout', initial_capital=100000,
                    commission_rate=0.001, interval=tf_name, limit=limit, strategy_params={'lookback': lb})
                if 'error' not in r and r['total_trades'] > 0:
                    results.append(('Breakout', sym, f'lb={lb}', r))
            except: pass

    # Multi-Factor
    for sym in symbols:
        for mom in mf_moms:
            for rsi in mf_rsis:
                try:
                    r = run_crypto_backtest(symbols=[sym], strategy_name='Multi-Factor', initial_capital=100000,
                        commission_rate=0.001, interval=tf_name, limit=limit,
                        strategy_params={'mom_lookback': mom, 'rsi_period': rsi})
                    if 'error' not in r and r['total_trades'] > 0:
                        results.append(('MultiFact', sym, f'mom={mom} rsi={rsi}', r))
                except: pass

    profitable = [(s, sym, p, r) for s, sym, p, r in results if r['total_return'] > 0]
    profitable.sort(key=lambda x: x[3]['total_return'], reverse=True)

    print(f'Tested {len(results)} configs | {len(profitable)} profitable')
    print(f'{"Strategy":>10s} {"Symbol":>8s} {"Params":>25s} | {"Return":>8s} {"MaxDD":>8s} {"WR":>6s} {"PF":>6s} {"Trades":>6s}')
    print('-' * 90)

    show = profitable[:15] if profitable else sorted(results, key=lambda x: x[3]['total_return'], reverse=True)[:10]
    for strat, sym, params, r in show:
        print(f"{strat:>10s} {sym:>8s} {params:>25s} | {r['total_return']:>+7.2%} {r['max_drawdown']:>7.2%} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} {r['total_trades']:>6d}")

print('\n' + '='*100)
print('DONE')
