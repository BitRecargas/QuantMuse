from .backtest_engine import BacktestEngine
from .crypto_backtest import run_crypto_backtest, fetch_binance_ohlcv

# Lazy import — PerformanceAnalyzer needs matplotlib/seaborn which aren't
# required for the sweep engine. Import it only when accessed.
def __getattr__(name):
    if name == "PerformanceAnalyzer":
        from .performance_analyzer import PerformanceAnalyzer
        return PerformanceAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BacktestEngine', 'PerformanceAnalyzer', 'run_crypto_backtest', 'fetch_binance_ohlcv'] 