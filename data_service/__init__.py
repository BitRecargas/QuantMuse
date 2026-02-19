# Core modules
try:
    from .fetchers import AlphaVantageFetcher, YahooFetcher, BinanceFetcher
except ImportError:
    # Handle missing dependencies gracefully
    AlphaVantageFetcher = None
    YahooFetcher = None
    BinanceFetcher = None

try:
    from .processors import DataProcessor
except ImportError:
    DataProcessor = None

try:
    from .storage import DatabaseManager, FileStorage, CacheManager
except ImportError:
    DatabaseManager = None
    FileStorage = None
    CacheManager = None

try:
    from .utils import Logger, TradingException
except ImportError:
    Logger = None
    TradingException = None

# AI modules
try:
    from .ai import SentimentAnalyzer, NewsProcessor, SocialMediaMonitor, LLMIntegration, NLPProcessor, SentimentFactorCalculator, LangChainAgent
except ImportError:
    SentimentAnalyzer = None
    NewsProcessor = None
    SocialMediaMonitor = None
    LLMIntegration = None
    NLPProcessor = None
    SentimentFactorCalculator = None
    LangChainAgent = None

# Backtesting modules
try:
    from .backtest import BacktestEngine, PerformanceAnalyzer
except ImportError:
    BacktestEngine = None
    PerformanceAnalyzer = None

# Factor analysis modules
try:
    from .factors import FactorCalculator, FactorScreener, FactorBacktest, StockSelector, FactorOptimizer
except ImportError:
    FactorCalculator = None
    FactorScreener = None
    FactorBacktest = None
    StockSelector = None
    FactorOptimizer = None

# Live trading
try:
    from .trading import TradingBot, BotConfig, BotState, BotManager
except ImportError:
    TradingBot = None
    BotConfig = None
    BotState = None
    BotManager = None

__version__ = "0.1.0"

__all__ = [
    # Data fetchers
    'AlphaVantageFetcher',
    'YahooFetcher', 
    'BinanceFetcher',
    
    # Data processors
    'DataProcessor',
    
    # Storage
    'DatabaseManager',
    'FileStorage',
    'CacheManager',
    
    # Utilities
    'Logger',
    'TradingException',
    
    # AI modules
    'SentimentAnalyzer',
    'NewsProcessor',
    'SocialMediaMonitor',
    'LLMIntegration',
    'NLPProcessor',
    'SentimentFactorCalculator',
    'LangChainAgent',
    
    # Backtesting
    'BacktestEngine',
    'PerformanceAnalyzer',
    
    # Quantitative factors
    'FactorCalculator',
    'FactorScreener',
    'FactorBacktest',
    'StockSelector',
    'FactorOptimizer',

    # Live trading
    'TradingBot',
    'BotConfig',
    'BotState',
    'BotManager',
] 