#!/usr/bin/env python3
"""
FastAPI Web Server for Trading System
Provides RESTful API endpoints for web management interface
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import logging
from datetime import datetime, timedelta
import asyncio
import json
import os

# Import our trading system modules
try:
    from ..backtest import BacktestEngine, PerformanceAnalyzer, run_crypto_backtest
    from ..factors import FactorCalculator, FactorScreener, FactorBacktest
    from ..strategies import StrategyRegistry
    from ..ai import LLMIntegration, NLPProcessor, SentimentFactorCalculator
    from ..fetchers import YahooFetcher, BinanceFetcher
    from ..storage import DatabaseManager
    from ..utils import Logger
except ImportError as e:
    logging.error(f"Failed to import trading modules: {e}")

try:
    from ..trading import BotManager, BotConfig
except ImportError as e:
    logging.warning(f"Trading module not available: {e}")
    BotManager = None
    BotConfig = None

# Pydantic models for API requests/responses
class StrategyRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    parameters: Dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0

class BacktestRequest(BaseModel):
    strategy_config: StrategyRequest
    commission_rate: float = 0.001
    rebalance_frequency: str = "daily"
    leverage: int = 1

class FactorAnalysisRequest(BaseModel):
    symbols: List[str]
    factors: List[str]
    start_date: str
    end_date: str

class AIAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "sentiment"  # sentiment, news, market_analysis

class BotStartRequest(BaseModel):
    strategy_name: str = "Momentum"
    symbol: str = "ETHUSDT"
    interval: str = "4h"
    strategy_params: Dict[str, Any] = {}
    capital: float = 100.0
    leverage: int = 5
    margin_type: str = "ISOLATED"
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.05
    candle_lookback: int = 300
    stop_loss: float = 0.0      # 0 = disabled, e.g. 0.01 = 1%
    take_profit: float = 0.0    # 0 = disabled, e.g. 0.03 = 3%

class SystemStatusResponse(BaseModel):
    status: str
    uptime: str
    active_strategies: int
    total_trades: int
    system_metrics: Dict[str, Any]

class APIServer:
    """FastAPI server for trading system web interface"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Trading System API",
            description="RESTful API for trading system management",
            version="1.0.0"
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize trading system components
        self._initialize_components()
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify actual origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Mount static files for frontend
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    def _initialize_components(self):
        """Initialize trading system components independently"""
        components = {
            'backtest_engine': lambda: BacktestEngine(),
            'performance_analyzer': lambda: PerformanceAnalyzer(),
            'factor_calculator': lambda: FactorCalculator(),
            'factor_screener': lambda: FactorScreener(),
            'factor_backtest': lambda: FactorBacktest(),
            'strategy_registry': lambda: StrategyRegistry(),
            'nlp_processor': lambda: NLPProcessor(use_transformers=False),
            'sentiment_calculator': lambda: SentimentFactorCalculator(),
        }
        for name, factory in components.items():
            try:
                setattr(self, name, factory())
            except Exception as e:
                self.logger.warning(f"Could not init {name}: {e}")
                setattr(self, name, None)

        # Optional components that need API keys
        try:
            self.llm_integration = LLMIntegration()
        except Exception:
            self.llm_integration = None

        try:
            self.yahoo_fetcher = YahooFetcher()
        except Exception:
            self.yahoo_fetcher = None

        try:
            self.binance_fetcher = BinanceFetcher()
        except Exception:
            self.binance_fetcher = None

        # Live trading bot manager
        try:
            if BotManager is not None:
                self.bot_manager = BotManager()
            else:
                self.bot_manager = None
        except Exception as e:
            self.logger.warning(f"Could not init BotManager: {e}")
            self.bot_manager = None

        self.logger.info("Trading system components initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main dashboard page"""
            index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "static", "index.html")
            if not os.path.exists(index_path):
                # Fallback: try relative to working directory
                index_path = os.path.join("static", "index.html")
            return FileResponse(index_path, media_type="text/html")
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/system/status")
        async def get_system_status():
            """Get system status and metrics"""
            try:
                # Get system metrics
                metrics = {
                    "cpu_usage": 45.2,
                    "memory_usage": 2.3,
                    "active_connections": 12,
                    "api_calls_per_min": 156
                }
                
                return SystemStatusResponse(
                    status="running",
                    uptime="2 days, 5 hours, 30 minutes",
                    active_strategies=3,
                    total_trades=1250,
                    system_metrics=metrics
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/strategies")
        async def get_available_strategies():
            """Get list of available strategies"""
            try:
                strategies = [
                    {"name": "Momentum Strategy", "description": "Price momentum based strategy"},
                    {"name": "Value Strategy", "description": "Value investing strategy"},
                    {"name": "Mean Reversion", "description": "Mean reversion strategy"},
                    {"name": "Multi-Factor", "description": "Multi-factor strategy"},
                    {"name": "Risk Parity", "description": "Risk parity strategy"},
                    {"name": "Sector Rotation", "description": "Sector rotation strategy"}
                ]
                return {"strategies": strategies}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/backtest/run")
        async def run_backtest(request: BacktestRequest):
            """Run real strategy backtest with Binance data"""
            try:
                cfg = request.strategy_config
                self.logger.info(f"Running backtest: {cfg.strategy_name} on {cfg.symbols}")

                # Parse limit from rebalance_frequency (sent as numeric string from frontend)
                try:
                    limit = int(request.rebalance_frequency)
                except (ValueError, TypeError):
                    period_map = {"daily": 365, "weekly": 365, "monthly": 365, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
                    limit = period_map.get(request.rebalance_frequency, 365)

                # Extract interval from parameters (default to 1d)
                interval = cfg.parameters.pop("interval", "1d")
                if interval not in ("5m", "15m", "1h", "4h", "1d"):
                    interval = "1d"

                # Run real backtest in a thread to not block the event loop
                lev = max(1, request.leverage)
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: run_crypto_backtest(
                        symbols=cfg.symbols,
                        strategy_name=cfg.strategy_name,
                        initial_capital=cfg.initial_capital,
                        commission_rate=request.commission_rate,
                        interval=interval,
                        limit=limit,
                        strategy_params=cfg.parameters,
                        leverage=lev,
                    )
                )

                if "error" in results:
                    raise HTTPException(status_code=400, detail=results["error"])

                return {"status": "success", "results": results}

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Backtest failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/factors/analyze")
        async def analyze_factors(request: FactorAnalysisRequest):
            """Analyze factor performance"""
            try:
                self.logger.info(f"Analyzing factors: {request.factors}")
                
                # Generate sample factor analysis results
                results = {
                    "factors": request.factors,
                    "performance": {
                        "momentum": {"ic": 0.15, "ir": 1.2, "win_rate": 0.58},
                        "value": {"ic": 0.12, "ir": 1.0, "win_rate": 0.52},
                        "quality": {"ic": 0.10, "ir": 0.8, "win_rate": 0.48}
                    },
                    "correlation_matrix": [
                        [1.0, 0.2, 0.1],
                        [0.2, 1.0, 0.3],
                        [0.1, 0.3, 1.0]
                    ]
                }
                
                return {"status": "success", "results": results}
                
            except Exception as e:
                self.logger.error(f"Factor analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/ai/analyze")
        async def analyze_with_ai(request: AIAnalysisRequest):
            """Analyze text using AI"""
            try:
                self.logger.info(f"AI analysis request: {request.analysis_type}")
                
                if request.analysis_type == "sentiment":
                    # Use NLP processor for sentiment analysis
                    processed = self.nlp_processor.preprocess_text(request.text)
                    result = {
                        "sentiment": processed.sentiment_label,
                        "confidence": processed.sentiment_score,
                        "keywords": processed.keywords[:5],
                        "topics": processed.topics
                    }
                else:
                    # Use LLM for other analysis types
                    result = {
                        "analysis": "AI analysis result",
                        "confidence": 0.85,
                        "recommendations": ["Sample recommendation 1", "Sample recommendation 2"]
                    }
                
                return {"status": "success", "results": result}
                
            except Exception as e:
                self.logger.error(f"AI analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/market/data/{symbol}")
        async def get_market_data(symbol: str, period: str = "1y", interval: str = "1d"):
            """Get real OHLCV market data from Binance"""
            import aiohttp

            # Map period to Binance limit
            period_map = {"1d": 24, "1w": 168, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
            limit = period_map.get(period, 365)

            # Map interval for Binance
            interval_map = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
            binance_interval = interval_map.get(interval, "1d")

            # Normalize symbol for Binance (add USDT if missing)
            pair = symbol.upper()
            if not pair.endswith(("USDT", "BTC", "ETH", "BUSD")):
                pair = pair + "USDT"

            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={binance_interval}&limit={limit}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail=f"Binance API error for {pair}")
                        klines = await resp.json()

                data = []
                for k in klines:
                    ts_ms = k[0]
                    dt = datetime.utcfromtimestamp(ts_ms / 1000)
                    data.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "time": dt.isoformat(),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "price": float(k[4])  # backwards compat
                    })

                return {"symbol": pair, "data": data}

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get market data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/portfolio/status")
        async def get_portfolio_status():
            """Get current portfolio status"""
            try:
                return {
                    "total_value": 125000.0,
                    "cash": 25000.0,
                    "positions": [
                        {"symbol": "AAPL", "quantity": 100, "value": 15000.0, "pnl": 1500.0},
                        {"symbol": "GOOGL", "quantity": 50, "value": 5000.0, "pnl": 500.0},
                        {"symbol": "MSFT", "quantity": 75, "value": 20000.0, "pnl": 2000.0}
                    ],
                    "daily_pnl": 1250.0,
                    "total_pnl": 4000.0
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get portfolio status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/trades/recent")
        async def get_recent_trades(limit: int = 20):
            """Get recent trades"""
            try:
                # Generate sample trade data
                trades = []
                for i in range(limit):
                    trades.append({
                        "id": f"trade_{i}",
                        "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                        "symbol": ["AAPL", "GOOGL", "MSFT"][i % 3],
                        "side": "buy" if i % 2 == 0 else "sell",
                        "quantity": 100 + i * 10,
                        "price": 150.0 + i * 0.5,
                        "status": "filled"
                    })

                return {"trades": trades}

            except Exception as e:
                self.logger.error(f"Failed to get recent trades: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # -----------------------------------------------------------
        # Live Trading Bot endpoints
        # -----------------------------------------------------------

        @self.app.on_event("startup")
        async def _restore_bots_on_startup():
            if self.bot_manager:
                try:
                    await self.bot_manager.restore_bots()
                except Exception as e:
                    self.logger.error(f"Failed to restore bots: {e}")

        @self.app.post("/api/bot/start")
        async def start_bot(request: BotStartRequest):
            """Start a new trading bot."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            try:
                config = BotConfig(
                    strategy_name=request.strategy_name,
                    symbol=request.symbol,
                    interval=request.interval,
                    strategy_params=request.strategy_params,
                    capital=request.capital,
                    leverage=request.leverage,
                    margin_type=request.margin_type,
                    max_drawdown=request.max_drawdown,
                    daily_loss_limit=request.daily_loss_limit,
                    candle_lookback=request.candle_lookback,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit,
                )
                result = await self.bot_manager.start_bot(config)
                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])
                return {"status": "started", "bot": result}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to start bot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/bot/{bot_id}/stop")
        async def stop_bot(bot_id: str):
            """Stop a bot and close its position."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            result = await self.bot_manager.stop_bot(bot_id, close_position=True)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return {"status": "stopped", "bot": result}

        @self.app.post("/api/bot/{bot_id}/pause")
        async def pause_bot(bot_id: str):
            """Pause a running bot."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            result = await self.bot_manager.pause_bot(bot_id)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return {"status": "paused", "bot": result}

        @self.app.post("/api/bot/{bot_id}/resume")
        async def resume_bot(bot_id: str):
            """Resume a paused bot."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            result = await self.bot_manager.resume_bot(bot_id)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return {"status": "resumed", "bot": result}

        @self.app.get("/api/bot/{bot_id}/status")
        async def get_bot_status(bot_id: str):
            """Get bot status, position, P&L."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            result = await self.bot_manager.get_bot_status(bot_id)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result

        @self.app.get("/api/bot/{bot_id}/trades")
        async def get_bot_trades(bot_id: str):
            """Get trade history for a bot."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            trades = self.bot_manager.get_bot_trades(bot_id)
            return {"bot_id": bot_id, "trades": trades}

        @self.app.get("/api/bot/{bot_id}/equity")
        async def get_bot_equity(bot_id: str):
            """Get equity curve for a bot."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            equity = self.bot_manager.get_bot_equity(bot_id)
            return {"bot_id": bot_id, "equity": equity}

        @self.app.get("/api/bot/list")
        async def list_bots():
            """List all bots."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            bots = await self.bot_manager.list_bots()
            return {"bots": bots}

        @self.app.get("/api/bot/account")
        async def get_account():
            """Get Binance account balances."""
            if not self.bot_manager:
                raise HTTPException(status_code=503, detail="Trading module not available")
            info = await self.bot_manager.get_account_info()
            return info
    
    def run(self, debug: bool = False):
        """Run the FastAPI server"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            log_level="info"
        )

def main():
    """Main function to run the API server"""
    server = APIServer()
    server.run(debug=True)


# Module-level app instance for uvicorn (e.g. uvicorn data_service.web.api_server:app)
app = APIServer().app

if __name__ == "__main__":
    main() 