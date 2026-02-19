# CLAUDE.md — QuantMuse Project Guide

## Project Overview
QuantMuse is a full-stack crypto/equities trading system with backtesting, strategy sweeps, live futures trading, AI/NLP analysis, and a Streamlit dashboard. Deployed via Docker Compose.

## Architecture

```
QuantMuse/
├── data_service/                # Core Python package
│   ├── backtest/
│   │   ├── crypto_backtest.py   # Single-asset backtest engine (Binance OHLCV)
│   │   ├── strategy_sweep.py    # Numba JIT mega-sweep (17k+ combos, 29 strategies)
│   │   ├── backtest_engine.py   # Legacy/generic backtest engine
│   │   └── performance_analyzer.py
│   ├── trading/
│   │   ├── bot_engine.py        # Live futures bot (Binance Futures API)
│   │   └── bot_manager.py       # Multi-bot orchestrator
│   ├── dashboard/
│   │   └── dashboard_app.py     # Streamlit dashboard (6 tabs)
│   ├── web/
│   │   └── api_server.py        # FastAPI REST API
│   ├── ai/                      # NLP, sentiment, LLM integration
│   ├── fetchers/                # Binance, Yahoo, Alpha Vantage
│   ├── factors/                 # Factor calculator, screener, backtest
│   ├── strategies/              # Strategy registry, optimizer
│   ├── storage/                 # DB, cache, file storage
│   ├── vector_db/               # Vector store for embeddings
│   ├── visualization/           # Plotly chart generators
│   └── utils/                   # Logger, exceptions
├── sweep_*.py                   # CLI sweep runners (1h/4h, 5m/15m, intraday)
├── docker-compose.yml           # 3 services: web, dashboard, redis
├── Dockerfile                   # Python 3.11-slim, pip install
└── data/sweeps/                 # Sweep result JSONs (~26MB each)
```

## Docker Services
| Service     | Container             | Port  | Command                                  |
|-------------|----------------------|-------|------------------------------------------|
| web         | quantmuse-web        | 8000  | `uvicorn data_service.web.api_server:app` |
| dashboard   | quantmuse-dashboard  | 8501  | `streamlit run data_service/dashboard/dashboard_app.py` |
| redis       | quantmuse-redis      | 6379  | Redis 7 Alpine                           |

**Rebuild & restart:** `docker-compose up --build -d`

## Key Design Decisions

### Backtest Engine (`crypto_backtest.py`)
- Fetches OHLCV from Binance public API (paginated, up to ~5 years)
- Signals execute on **next candle's open** (no look-ahead bias)
- Bar processing order: Liquidation check -> SL/TP check -> Mark-to-market -> Execute pending signal
- 29 strategies in `STRATEGY_MAP` (Momentum, Mean Reversion, Breakout, MACD, Keltner, RSI+EMA, Supertrend, etc.)
- Supports `leverage` parameter (1 = spot, >1 = futures margin-based)

### Sweep Engine (`strategy_sweep.py`)
- Numba `@njit(cache=True)` inner loop for native-speed backtesting
- `build_param_grid()` generates 17,000+ strategy/param combinations
- `fast_backtest()` wraps the JIT loop with Pandas pre/post-processing
- `run_mega_sweep()` tests multiple coins x multiple SL/TP levels
- **`--workers N`**: CPU multiprocessing for parallel execution
  - Single-symbol sweeps: splits strategy grid into N chunks processed in parallel
  - Mega sweeps: processes N coins in parallel (one worker per coin)
  - Recommended: `--workers` = number of physical CPU cores (not hyperthreads)
  - `--workers 1` (default): sequential execution, fully backward compatible
- **After changing `_backtest_loop` signature:** delete `__pycache__` to force Numba recompilation

### Futures Leverage Model (added 2025-02-18)
- `leverage` parameter in both `crypto_backtest.py` and `strategy_sweep.py`
- Position sizing: `qty = margin * leverage / exec_price`
- Cash deduction: deducts margin only (not notional)
- Equity: `cash + abs(pos_qty) * pos_entry / leverage + unrealized`
- Close position: returns `margin + pnl` (not `notional + pnl`)
- Liquidation: if `margin_locked + unrealized <= maintenance_margin` (0.5% of notional)
- At `leverage=1`: all `/leverage` divisions are no-ops, fully backward compatible

### Live Trading (`bot_engine.py`)
- Uses Binance Futures API with real leverage, margin types (ISOLATED/CROSSED)
- Configurable SL/TP, max drawdown, daily loss limits
- Bot state persisted to Redis for crash recovery

### Dashboard (`dashboard_app.py`)
- 6 tabs: Performance Analysis, Strategy Backtest, Market Data, AI Analysis, System Status, Live Trading
- Performance Analysis loads sweep JSONs from `data/sweeps/`
- Strategy Backtest has leverage slider (1-50x), SL/TP controls, strategy presets
- Live Trading tab manages futures bots via API

### API Server (`api_server.py`)
- `BacktestRequest` accepts `leverage` parameter
- `BotStartRequest` configures live bot (strategy, leverage, margin type, SL/TP)
- Bot management: start/stop/pause/resume/status/trades/equity endpoints

### Walk-Forward Testing (`strategy_sweep.py`)
- `generate_wf_windows(n_bars, n_windows, train_pct)` — generates sliding train/test index windows
- `run_walkforward_sweep()` — single-symbol walk-forward: compute signals once on full data, slice per window, aggregate OOS metrics
- `run_mega_walkforward()` — multi-coin walk-forward with multiple SL/TP levels
- Signals computed on FULL dataset once (no look-ahead: signal[i] depends only on df[:i+1]), then sliced per window
- Capital resets fresh per window — each fold is an independent backtest
- Results sorted by OOS Sharpe (primary), consistency, OOS return
- Key metrics: `oos_sharpe_mean`, `consistency` (% profitable windows), `sharpe_decay` (0 = generalizes, 1 = overfitted)
- Dashboard auto-detects `wf_*.json` files and shows WF leaderboard, IS vs OOS scatter, per-window detail

**CLI usage:**
```bash
# Basic walk-forward
python -m data_service.backtest.strategy_sweep \
  --symbol ETHUSDT --interval 1h --limit 15000 \
  --walkforward --wf-windows 5 --wf-train-pct 0.70

# Walk-forward with SL/TP
python -m data_service.backtest.strategy_sweep \
  --symbol ETHUSDT --interval 15m --limit 35000 \
  --walkforward --wf-windows 5 --sl 0.02 --tp 0.06

# Mega walk-forward (10 coins)
python -m data_service.backtest.strategy_sweep \
  --interval 1h --limit 15000 --mega \
  --walkforward --wf-windows 5

# Parallel: single symbol with 8 workers
python -m data_service.backtest.strategy_sweep \
  --symbol ETHUSDT --interval 1h --limit 15000 \
  --walkforward --workers 8

# Parallel: mega walk-forward with 10 workers (1 per coin)
python -m data_service.backtest.strategy_sweep \
  --interval 1h --limit 15000 --mega --walkforward --workers 10

# Parallel: mega sweep with 16 workers
python -m data_service.backtest.strategy_sweep \
  --interval 5m --limit 50000 --mega --workers 16
```

**Recommended `limit` values per timeframe for walk-forward (need enough bars for train+test):**
- 1m: 50000+ (35 days)
- 5m: 50000+ (175 days)
- 15m: 35000+ (365 days)
- 1h: 15000+ (625 days)
- 4h: 5000+ (833 days)

## Sweep Results
- Regular sweeps: `data/sweeps/mega_<SYMBOL>_<TF>.json` (~26MB each)
- Walk-forward: `data/sweeps/wf_<SYMBOL>_<TF>.json`
Completed timeframes: 1h (10 coins), 15m (10 coins), 5m (in progress), 1m (in progress).
Coins: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, DOTUSDT.

## Environment
- `.env` file contains `BINANCE_API_KEY`, `BINANCE_API_SECRET` (for live trading only)
- Backtesting uses public Binance API (no keys needed)
- Redis used for bot state persistence

## Common Commands
```bash
# Rebuild and restart
docker-compose up --build -d

# Check logs
docker-compose logs -f dashboard
docker-compose logs -f web

# Run a sweep from CLI
python -m data_service.backtest.strategy_sweep --symbol ETHUSDT --interval 15m --limit 35000

# Run mega sweep (10 coins, multiple SL/TP)
python -m data_service.backtest.strategy_sweep --symbol ETHUSDT --interval 5m --limit 50000 --mega
```

## Conventions
- Strategy signals: `1` = buy, `-1` = sell, `0` = hold (pd.Series)
- Commission: 0.1% default (0.001)
- Slippage: 0.02% in sweep engine (0.0002)
- Capital: $10,000 default in sweeps, $100,000 in dashboard
- Minimum 3 trades required for a sweep result to be valid
- Sweep results sorted by Sharpe ratio (primary), then profit factor, then total return
