# QuantMuse — Bellas Server Context

## What This Project Is
QuantMuse is a crypto trading system with backtesting, strategy sweeps, live futures trading, and a Streamlit dashboard. You are on the **bellas server** (Ryzen 9 5950X, 64GB RAM, RTX 5090, Ubuntu 24.04).

## Current State on Bellas
- **Repo**: `~/QuantMuse` cloned from `https://github.com/BitRecargas/QuantMuse.git`
- **Python venv**: `~/QuantMuse/venv` with numba, pandas, numpy, requests installed
- **Docker**: `docker-compose.bellas.yml` for full stack (API :9000, Dashboard :9501, Redis)
- **Sweep engine**: `data_service/backtest/strategy_sweep.py` with `--workers N` multiprocessing

## Architecture
```
QuantMuse/
├── data_service/backtest/
│   ├── strategy_sweep.py    # Numba JIT mega-sweep (17k+ combos, 29 strategies, --workers N)
│   ├── crypto_backtest.py   # Single-asset backtest engine (Binance OHLCV)
│   └── performance_analyzer.py  # (lazy-loaded, needs matplotlib/seaborn)
├── data_service/trading/
│   ├── bot_engine.py        # Live futures bot (Binance Futures API)
│   └── bot_manager.py       # Multi-bot orchestrator
├── data_service/dashboard/
│   └── dashboard_app.py     # Streamlit dashboard (6 tabs)
├── data_service/web/
│   └── api_server.py        # FastAPI REST API
├── data/sweeps/             # Sweep result JSONs (wf_*.json, mega_*.json)
├── docker-compose.bellas.yml  # Full stack: ports 9000/9501
├── docker-compose.sweep.yml   # Sweep-only Docker (not recommended, use venv)
├── Dockerfile               # Full image with torch/spacy/nltk
├── Dockerfile.sweep         # Lightweight sweep-only image
├── setup_bellas.sh          # One-shot setup script
└── .env                     # API keys (Binance, OpenAI)
```

## How to Run Sweeps
The sweep engine tests 17,273 strategy/param combinations across 10 coins with walk-forward validation. Uses `--workers N` for CPU multiprocessing.

```bash
source ~/QuantMuse/venv/bin/activate && cd ~/QuantMuse

# Single symbol test (quick, ~15 seconds)
python -m data_service.backtest.strategy_sweep \
  --symbol ETHUSDT --interval 1h --limit 500 --workers 4 --top 5

# Full mega walk-forward (10 coins, all SL/TP combos)
# Run 2 at a time with --workers 8 each = 16 cores fully used
nohup python -m data_service.backtest.strategy_sweep \
  --interval 1h --limit 15000 --mega --walkforward --workers 8 \
  > /tmp/sweep_1h.log 2>&1 &

nohup python -m data_service.backtest.strategy_sweep \
  --interval 15m --limit 35000 --mega --walkforward --workers 8 \
  > /tmp/sweep_15m.log 2>&1 &

# After those finish:
nohup python -m data_service.backtest.strategy_sweep \
  --interval 5m --limit 50000 --mega --walkforward --workers 8 \
  > /tmp/sweep_5m.log 2>&1 &

nohup python -m data_service.backtest.strategy_sweep \
  --interval 1m --limit 50000 --mega --walkforward --workers 8 \
  > /tmp/sweep_1m.log 2>&1 &

# Check progress
tail -5 /tmp/sweep_1h.log
tail -5 /tmp/sweep_15m.log
```

### Worker Guidelines
- `--workers 8` per job, max 2 concurrent jobs = 16 cores (Ryzen 9 5950X)
- Mega sweeps use **coin-level parallelism** (1 worker per coin)
- Single-symbol sweeps split the strategy grid into N chunks
- `--workers 1` = sequential (backward compatible)
- Each worker holds ~50MB. 8 workers × 2 jobs = ~800MB — fine on 64GB

### Sweep Output
- Results go to `data/sweeps/wf_<SYMBOL>_<TF>.json` (walk-forward)
- Results go to `data/sweeps/mega_<SYMBOL>_<TF>.json` (regular)
- Each file is ~76-80MB, ~48k strategy results per coin
- Dashboard auto-detects these files

### Timeframe Limits (minimum candles for walk-forward)
- 1h: `--limit 15000` (625 days)
- 15m: `--limit 35000` (365 days)
- 5m: `--limit 50000` (175 days)
- 1m: `--limit 50000` (35 days)

## How to Run Full Docker Stack
```bash
cd ~/QuantMuse

# Build and start (API + Dashboard + Redis)
docker compose -f docker-compose.bellas.yml up --build -d

# Check logs
docker compose -f docker-compose.bellas.yml logs -f

# Stop
docker compose -f docker-compose.bellas.yml down
```

- **Dashboard**: http://localhost:9501
- **API**: http://localhost:9000
- **Redis**: internal only (no exposed port)
- Uses isolated `quantmuse-net` Docker network, ports 9000/9501 to avoid conflicts

## Key Design Details

### Backtest Engine
- Numba `@njit(cache=True)` inner loop for native-speed backtesting
- Signals execute on **next candle's open** (no look-ahead bias)
- Bar order: Liquidation check → SL/TP check → Mark-to-market → Execute pending signal
- 29 strategies in `STRATEGY_MAP`
- Supports `leverage` parameter (1 = spot, >1 = futures margin-based)

### Walk-Forward Testing
- `generate_wf_windows(n_bars, n_windows, train_pct)` — sliding train/test windows
- Signals computed on FULL dataset once (no look-ahead), then sliced per window
- Capital resets fresh per window
- Key metrics: `oos_sharpe_mean`, `consistency` (% profitable windows), `sharpe_decay`
- Sorted by OOS Sharpe (primary), consistency, OOS return

### Futures Leverage Model
- Position sizing: `qty = margin * leverage / exec_price`
- Equity: `cash + abs(pos_qty) * pos_entry / leverage + unrealized`
- Liquidation: if `margin_locked + unrealized <= maintenance_margin` (0.5% of notional)

### After Changing `_backtest_loop` Signature
Delete `__pycache__` to force Numba recompilation:
```bash
find ~/QuantMuse -name "__pycache__" -path "*/backtest/*" -exec rm -rf {} +
```

## Analysis Results (from Mac sweeps)

### Top Strategy: RSI+EMA
- Best params (15m): `rsi=5, ema=5, overbought=60, oversold=15, SL=2%, TP=6%`
- OOS Sharpe: 2.19, works on all 10 coins, 74% consistency, 16 trades/window
- RSI+EMA dominates across all timeframes

### Best by Timeframe
- **1h**: Keltner Channel (ema=3, atr=14, mult=1.5, SL=1%/TP=3%) — perfect 1.0 consistency
- **15m**: RSI+EMA (rsi=5, ema=5, ob=60, os=15, SL=2%/TP=6%) — best balance
- **5m/1m**: RSI+EMA variants dominate, higher Sharpe but fewer trades

### Best SL/TP: 2% / 6% (clear winner, tight stops kill most strategies)

### Dead Strategies (avoid)
MACD, Parabolic SAR, Fisher Transform, Squeeze, Multi-Factor, Elder Ray, KST, Hull MA

## Coins
BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, DOTUSDT

## SL/TP Levels Tested
- 0.5% SL / 1.5% TP
- 1% SL / 3% TP
- 2% SL / 6% TP

## Conventions
- Strategy signals: `1` = buy, `-1` = sell, `0` = hold (pd.Series)
- Commission: 0.1% (0.001)
- Slippage: 0.02% (0.0002)
- Capital: $10,000 default
- Minimum 3 trades required for valid result

## Git
- Remote: `https://github.com/BitRecargas/QuantMuse.git`
- Branch: `main`
- Push changes: `git push origin main`
- Pull from Mac: `git pull`

## User Preferences
- Prefers autonomous execution ("do it") over asking for confirmation
- Runs long sweeps as background processes with nohup
- Wants backtest numbers to match live futures trading
- Use `--workers 8` for sweeps (physical cores, not hyperthreads)
