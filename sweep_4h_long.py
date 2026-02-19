#!/usr/bin/env python3
import sys, time
sys.path.insert(0, '/app')
from data_service.backtest.crypto_backtest import run_crypto_backtest

top_configs = [
    ('Momentum', 'ETHUSDT', {'lookback': 168}, 'lb=168'),
    ('Momentum', 'ETHUSDT', {'lookback': 126}, 'lb=126'),
    ('Breakout', 'ETHUSDT', {'lookback': 6}, 'lb=6'),
    ('Multi-Factor', 'BNBUSDT', {'mom_lookback': 126, 'rsi_period': 6}, 'mom=126 rsi=6'),
    ('Multi-Factor', 'BNBUSDT', {'mom_lookback': 126, 'rsi_period': 12}, 'mom=126 rsi=12'),
    ('Breakout', 'BNBUSDT', {'lookback': 126}, 'lb=126'),
    ('Breakout', 'BNBUSDT', {'lookback': 6}, 'lb=6'),
    ('Momentum', 'BNBUSDT', {'lookback': 126}, 'lb=126'),
    ('Momentum', 'BNBUSDT', {'lookback': 168}, 'lb=168'),
    ('Momentum', 'BTCUSDT', {'lookback': 168}, 'lb=168'),
    ('Momentum', 'BTCUSDT', {'lookback': 126}, 'lb=126'),
    ('Momentum', 'SOLUSDT', {'lookback': 168}, 'lb=168'),
    ('Momentum', 'SOLUSDT', {'lookback': 126}, 'lb=126'),
]

periods = [
    ('1Y', 2190),
    ('2Y', 4380),
    ('3Y', 6570),
]

out = open('/tmp/4h_results.txt', 'w')

def p(s):
    print(s)
    out.write(s + '\n')
    out.flush()

p('=' * 105)
p('4-HOUR CANDLES: LONG-TERM ROBUSTNESS TEST (1Y, 2Y, 3Y)')
p('=' * 105)

for strat, sym, params, label in top_configs:
    p(f'\n--- {strat} | {sym} | {label} ---')
    p(f'{"Period":>6s} {"Return":>10s} {"Ann.Ret":>10s} {"MaxDD":>8s} {"WR":>6s} {"PF":>7s} {"Trades":>7s} {"Final":>12s}')

    all_positive = True
    for pname, limit in periods:
        time.sleep(2)
        try:
            r = run_crypto_backtest(
                symbols=[sym], strategy_name=strat,
                initial_capital=100000, commission_rate=0.001,
                interval='4h', limit=limit, strategy_params=params
            )
            if 'error' not in r:
                ret = r['total_return']
                if ret < 0:
                    all_positive = False
                p(f"{pname:>6s} {ret:>+9.2%} {r['annualized_return']:>+9.2%} {r['max_drawdown']:>7.2%} {r['win_rate']:>5.1%} {r['profit_factor']:>7.2f} {r['total_trades']:>7d} ${r['final_value']:>10,.0f}")
            else:
                p(f"{pname:>6s} ERROR: {r['error']}")
        except Exception as e:
            p(f"{pname:>6s} FAILED: {e}")

    if all_positive:
        p('  >>> PROFITABLE ACROSS ALL PERIODS <<<')

p('\nDONE')
out.close()
