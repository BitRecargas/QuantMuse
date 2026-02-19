#!/usr/bin/env python3
"""
Trading System Dashboard
A comprehensive Streamlit dashboard for trading system visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import logging
import sys
import os
import requests
from typing import Any, Dict

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from data_service.backtest import BacktestEngine, PerformanceAnalyzer
    from data_service.dashboard import ChartGenerator, DashboardWidgets
    from data_service.factors import FactorCalculator, FactorBacktest
    from data_service.strategies import StrategyRegistry
    from data_service.ai import NLPProcessor, SentimentFactorCalculator
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.info("Please install required dependencies: pip install -e .[ai,visualization]")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDashboard:
    """Main trading dashboard application"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
        self.widgets = DashboardWidgets()
        self.performance_analyzer = PerformanceAnalyzer()
        self.backtest_engine = BacktestEngine()
        self.factor_calculator = FactorCalculator()
        self.factor_backtest = FactorBacktest()
        self.nlp_processor = NLPProcessor(use_transformers=False)
        self.sentiment_calculator = SentimentFactorCalculator()
        
    def run(self):
        """Run the dashboard application"""
        st.set_page_config(
            page_title="Trading System Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">📈 Trading System Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Performance Analysis",
            "🎯 Strategy Backtest",
            "📈 Market Data",
            "🤖 AI Analysis",
            "⚙️ System Status",
            "🔴 Live Trading",
        ])

        with tab1:
            self._show_performance_analysis()

        with tab2:
            self._show_strategy_backtest()

        with tab3:
            self._show_market_data()

        with tab4:
            self._show_ai_analysis()

        with tab5:
            self._show_system_status()

        with tab6:
            self._show_live_trading()
    
    def _create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Date range selector
        st.sidebar.subheader("📅 Date Range")
        start_date = st.sidebar.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Strategy selector
        st.sidebar.subheader("🎯 Strategy")
        strategy_options = ["Momentum Strategy", "Value Strategy", "Mean Reversion", "Custom"]
        selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)
        
        # Symbols selector
        st.sidebar.subheader("📈 Symbols")
        symbols = st.sidebar.multiselect(
            "Select Symbols",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
             "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"],
            default=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        )
        
        # Initial capital
        st.sidebar.subheader("💰 Capital")
        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=10000
        )
        
        # Store in session state
        st.session_state.update({
            'start_date': start_date,
            'end_date': end_date,
            'selected_strategy': selected_strategy,
            'symbols': symbols,
            'initial_capital': initial_capital
        })
    
    def _show_performance_analysis(self):
        """Show performance analysis tab — loads real sweep results and runs live backtests."""
        from data_service.backtest.crypto_backtest import run_crypto_backtest

        st.header("📊 Performance Analysis")

        # --- Load sweep results ---
        sweep_dir = os.environ.get("SWEEP_DIR", "/app/data/sweeps")
        sweep_files = {
            "1m": os.path.join(sweep_dir, "sweep_futures_ETHUSDT_1m.json"),
            "5m": os.path.join(sweep_dir, "sweep_futures_ETHUSDT_5m.json"),
            "15m": os.path.join(sweep_dir, "sweep_futures_ETHUSDT_15m.json"),
        }
        all_results = []
        for tf, path in sweep_files.items():
            try:
                with open(path) as f:
                    data = json.load(f)
                for r in data:
                    r["_tf"] = tf
                all_results.extend(data)
            except FileNotFoundError:
                pass

        if not all_results:
            st.warning("No sweep results found. Run a strategy sweep first from the command line.")
            return

        # Build leaderboard dataframe
        lb = pd.DataFrame(all_results)
        lb = lb[lb["trades"] >= 3].sort_values("sharpe", ascending=False).reset_index(drop=True)

        # --- Timeframe + strategy selector ---
        col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 2])
        with col_sel1:
            tf_choice = st.selectbox("Timeframe", ["All", "1m", "5m", "15m"], key="pa_tf")
        with col_sel2:
            min_trades = st.slider("Min Trades", 3, 50, 10, key="pa_min_trades")
        with col_sel3:
            sort_by = st.selectbox("Rank By", ["sharpe", "total_return", "profit_factor", "win_rate"], key="pa_sort")

        filtered = lb[lb["trades"] >= min_trades]
        if tf_choice != "All":
            filtered = filtered[filtered["_tf"] == tf_choice]
        filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)

        if filtered.empty:
            st.warning("No results match the filter criteria.")
            return

        # --- Top 20 leaderboard ---
        st.subheader(f"🏆 Top 20 Strategies ({len(filtered):,} total)")
        display_cols = ["_tf", "strategy", "total_return", "sharpe", "sortino", "max_drawdown",
                        "trades", "win_rate", "profit_factor", "params"]
        top_df = filtered.head(20)[display_cols].copy()
        top_df.columns = ["TF", "Strategy", "Return", "Sharpe", "Sortino", "MaxDD",
                          "Trades", "WinRate", "PF", "Params"]
        top_df["Return"] = top_df["Return"].apply(lambda x: f"{x:.2%}")
        top_df["Sharpe"] = top_df["Sharpe"].apply(lambda x: f"{x:.2f}")
        top_df["Sortino"] = top_df["Sortino"].apply(lambda x: f"{x:.2f}")
        top_df["MaxDD"] = top_df["MaxDD"].apply(lambda x: f"{x:.2%}")
        top_df["WinRate"] = top_df["WinRate"].apply(lambda x: f"{x:.1%}")
        top_df["PF"] = top_df["PF"].apply(lambda x: f"{x:.2f}")
        def _fmt_params(p):
            if isinstance(p, dict):
                return ", ".join(f"{k}={v}" for k, v in p.items())
            return str(p)
        top_df["Params"] = top_df["Params"].apply(_fmt_params)
        top_df.index = range(1, len(top_df) + 1)
        st.dataframe(
            top_df,
            use_container_width=True,
            column_config={"Params": st.column_config.TextColumn("Params", width="large")},
        )

        # --- Pick strategy to deep-dive ---
        st.divider()
        st.subheader("🔍 Strategy Deep Dive")

        options = []
        for i, row in filtered.head(20).iterrows():
            params_str = ", ".join(f"{k}={v}" for k, v in row["params"].items()) if isinstance(row["params"], dict) else str(row["params"])
            label = f"#{i+1} {row['_tf']} {row['strategy']} — {row['total_return']:.2%} Sharpe {row['sharpe']:.2f} ({params_str})"
            options.append((label, i))

        selected_label = st.selectbox("Select a strategy to analyze:", [o[0] for o in options], key="pa_pick")
        selected_idx = [o[1] for o in options if o[0] == selected_label][0]
        selected = filtered.iloc[selected_idx]

        # Summary metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{selected['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{selected['sharpe']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{selected['max_drawdown']:.2%}")
        with col4:
            st.metric("Win Rate", f"{selected['win_rate']:.1%}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Sortino", f"{selected['sortino']:.2f}")
        with col6:
            st.metric("Profit Factor", f"{selected['profit_factor']:.2f}")
        with col7:
            st.metric("Trades", str(selected["trades"]))
        with col8:
            st.metric("Final Equity", f"${selected['final_equity']:,.2f}")

        # Run full backtest for equity curve
        if st.button("📈 Run Full Backtest (Equity Curve)", type="primary", key="pa_run"):
            strat_name = selected["strategy"]
            symbol = selected.get("symbol", "ETHUSDT")
            interval = selected["_tf"]
            params = selected["params"] if isinstance(selected["params"], dict) else {}
            candles = int(selected.get("candles", 15000))

            with st.spinner(f"Running {strat_name} on {symbol} {interval}..."):
                results = run_crypto_backtest(
                    symbols=[symbol],
                    strategy_name=strat_name,
                    initial_capital=10000.0,
                    commission_rate=0.001,
                    interval=interval,
                    limit=candles,
                    strategy_params=params,
                )

            if "error" in results:
                st.error(f"Backtest failed: {results['error']}")
            else:
                self._display_backtest_results(results)

        # --- Aggregate stats ---
        st.divider()
        st.subheader("📊 Sweep Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Strategy Frequency in Top 100**")
            top100 = filtered.head(100)
            freq = top100["strategy"].value_counts().reset_index()
            freq.columns = ["Strategy", "Count"]
            fig_bar = px.bar(freq.head(10), x="Strategy", y="Count",
                             title="Most Frequent Strategies in Top 100")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.write("**Return vs Max Drawdown (Top 100)**")
            scatter_df = top100[["strategy", "total_return", "max_drawdown", "sharpe", "trades"]].copy()
            scatter_df["total_return_pct"] = scatter_df["total_return"] * 100
            scatter_df["max_drawdown_pct"] = scatter_df["max_drawdown"] * 100
            fig_scatter = px.scatter(
                scatter_df, x="max_drawdown_pct", y="total_return_pct",
                color="strategy", size="trades", hover_data=["sharpe"],
                title="Return vs Drawdown",
                labels={"max_drawdown_pct": "Max Drawdown %", "total_return_pct": "Total Return %"},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Sharpe distribution
        st.write("**Sharpe Ratio Distribution (all valid results)**")
        fig_hist = px.histogram(filtered[filtered["sharpe"].between(-3, 5)], x="sharpe",
                                nbins=100, title="Sharpe Ratio Distribution",
                                color_discrete_sequence=["royalblue"])
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="Sharpe=1")
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- Walk-Forward Analysis ---
        self._show_walkforward_analysis(sweep_dir)
    
    def _show_walkforward_analysis(self, sweep_dir: str):
        """Show walk-forward analysis section if WF results exist."""
        import glob as glob_mod

        wf_files = sorted(glob_mod.glob(os.path.join(sweep_dir, "wf_*.json")))
        if not wf_files:
            return

        st.divider()
        st.subheader("Walk-Forward Analysis (Out-of-Sample)")

        # Parse available symbols/timeframes from filenames (no loading yet)
        wf_index = {}
        for path in wf_files:
            fname = os.path.basename(path)
            parts = fname.replace("wf_", "").replace(".json", "").rsplit("_", 1)
            if len(parts) == 2:
                wf_index[(parts[0], parts[1])] = path

        all_symbols = sorted(set(k[0] for k in wf_index))
        all_intervals = sorted(set(k[1] for k in wf_index))

        # Filter controls — select BEFORE loading
        wf_c1, wf_c2, wf_c3, wf_c4 = st.columns(4)
        with wf_c1:
            wf_sym = st.selectbox("Symbol", all_symbols, key="wf_sym")
        with wf_c2:
            wf_tf = st.selectbox("Timeframe", all_intervals, key="wf_tf")
        with wf_c3:
            wf_min_consist = st.slider("Min Consistency %", 0, 100, 60, 10, key="wf_min_consist") / 100
        with wf_c4:
            wf_sort = st.selectbox("Sort By", ["oos_sharpe_mean", "consistency", "oos_return_mean",
                                                "oos_sharpe_min", "sharpe_decay"], key="wf_sort")

        # Load only the selected file
        selected_path = wf_index.get((wf_sym, wf_tf))
        if not selected_path:
            st.warning(f"No walk-forward results for {wf_sym} {wf_tf}.")
            return

        @st.cache_data(ttl=600)
        def _load_wf_top(path, top_n=500):
            with open(path) as f:
                data = json.load(f)
            # Only keep top N to avoid memory issues (files can be 80MB+)
            return data[:top_n]

        wf_data = _load_wf_top(selected_path)
        if not wf_data:
            st.warning("Walk-forward file is empty.")
            return

        wf_df = pd.DataFrame(wf_data)
        wf_df["symbol"] = wf_sym
        wf_df["interval"] = wf_tf

        # Filter out degenerate Sharpe values and apply consistency filter
        wf_filt = wf_df[
            (wf_df["oos_sharpe_mean"].between(-50, 50)) &
            (wf_df["is_sharpe_mean"].between(-50, 50)) &
            (wf_df["consistency"] >= wf_min_consist) &
            (wf_df["oos_trades_mean"] >= 3)
        ].copy()

        ascending = True if wf_sort == "sharpe_decay" else False
        wf_filt = wf_filt.sort_values(wf_sort, ascending=ascending).reset_index(drop=True)

        if wf_filt.empty:
            st.warning("No walk-forward results match the filters.")
            return

        # Leaderboard table
        st.write(f"**Top 20 Walk-Forward Results** ({len(wf_filt):,} total)")
        disp_cols = ["strategy", "symbol", "interval", "is_sharpe_mean", "oos_sharpe_mean",
                     "oos_return_mean", "oos_max_dd_mean", "consistency", "sharpe_decay", "params"]
        top_wf = wf_filt.head(20)[disp_cols].copy()
        top_wf.columns = ["Strategy", "Symbol", "TF", "IS Sharpe", "OOS Sharpe",
                          "OOS Return", "OOS MaxDD", "Consistency", "Sharpe Decay", "Params"]
        top_wf["IS Sharpe"] = top_wf["IS Sharpe"].apply(lambda x: f"{x:.2f}")
        top_wf["OOS Sharpe"] = top_wf["OOS Sharpe"].apply(lambda x: f"{x:.2f}")
        top_wf["OOS Return"] = top_wf["OOS Return"].apply(lambda x: f"{x:.2%}")
        top_wf["OOS MaxDD"] = top_wf["OOS MaxDD"].apply(lambda x: f"{x:.2%}")
        top_wf["Consistency"] = top_wf["Consistency"].apply(lambda x: f"{x:.0%}")
        top_wf["Sharpe Decay"] = top_wf["Sharpe Decay"].apply(lambda x: f"{x:.2f}")
        def _fmt_p(p):
            if isinstance(p, dict):
                return ", ".join(f"{k}={v}" for k, v in p.items())
            return str(p)
        top_wf["Params"] = top_wf["Params"].apply(_fmt_p)
        top_wf.index = range(1, len(top_wf) + 1)
        st.dataframe(top_wf, use_container_width=True,
                     column_config={"Params": st.column_config.TextColumn("Params", width="large")})

        # IS vs OOS Sharpe scatter
        wf_c_left, wf_c_right = st.columns(2)
        with wf_c_left:
            st.write("**IS vs OOS Sharpe** (diagonal = perfect generalization)")
            scatter_wf = wf_filt.head(200)[["strategy", "is_sharpe_mean", "oos_sharpe_mean", "consistency"]].copy()
            fig_wf_scatter = px.scatter(
                scatter_wf, x="is_sharpe_mean", y="oos_sharpe_mean",
                color="strategy", size="consistency",
                hover_data=["consistency"],
                title="In-Sample vs Out-of-Sample Sharpe",
                labels={"is_sharpe_mean": "IS Sharpe (mean)", "oos_sharpe_mean": "OOS Sharpe (mean)"},
            )
            # Diagonal line
            max_val = max(scatter_wf["is_sharpe_mean"].max(), scatter_wf["oos_sharpe_mean"].max(), 1)
            min_val = min(scatter_wf["is_sharpe_mean"].min(), scatter_wf["oos_sharpe_mean"].min(), -1)
            fig_wf_scatter.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                     line=dict(color="gray", dash="dash"))
            fig_wf_scatter.update_layout(height=450)
            st.plotly_chart(fig_wf_scatter, use_container_width=True)

        with wf_c_right:
            st.write("**Sharpe Decay Distribution**")
            decay_data = wf_filt[wf_filt["sharpe_decay"].between(-2, 2)]
            fig_decay = px.histogram(decay_data, x="sharpe_decay", nbins=50,
                                     title="Sharpe Decay (0 = generalizes, 1 = overfitted)",
                                     color_discrete_sequence=["coral"])
            fig_decay.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="Perfect")
            fig_decay.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Overfitted")
            fig_decay.update_layout(height=450)
            st.plotly_chart(fig_decay, use_container_width=True)

        # Per-window detail for selected strategy
        st.write("**Per-Window OOS Detail**")
        wf_options = []
        for i, row in wf_filt.head(20).iterrows():
            ps = _fmt_p(row["params"])
            label = f"#{i+1} {row['symbol']} {row['interval']} {row['strategy']} — OOS Sharpe {row['oos_sharpe_mean']:.2f} ({ps})"
            wf_options.append((label, i))

        if wf_options:
            sel_label = st.selectbox("Select strategy:", [o[0] for o in wf_options], key="wf_detail_pick")
            sel_idx = [o[1] for o in wf_options if o[0] == sel_label][0]
            sel_row = wf_filt.iloc[sel_idx]

            per_window = sel_row.get("per_window_oos", [])
            if per_window and isinstance(per_window, list):
                pw_df = pd.DataFrame(per_window)
                pw_c1, pw_c2 = st.columns(2)
                with pw_c1:
                    fig_pw_ret = px.bar(pw_df, x="window", y="total_return",
                                        title="OOS Return per Window",
                                        color="total_return",
                                        color_continuous_scale=["red", "gray", "green"],
                                        labels={"total_return": "Return", "window": "Window"})
                    fig_pw_ret.update_layout(height=350)
                    st.plotly_chart(fig_pw_ret, use_container_width=True)
                with pw_c2:
                    fig_pw_shrp = px.bar(pw_df, x="window", y="sharpe",
                                         title="OOS Sharpe per Window",
                                         color="sharpe",
                                         color_continuous_scale=["red", "gray", "green"],
                                         labels={"sharpe": "Sharpe", "window": "Window"})
                    fig_pw_shrp.update_layout(height=350)
                    st.plotly_chart(fig_pw_shrp, use_container_width=True)

                st.dataframe(pw_df, use_container_width=True)

    def _show_strategy_backtest(self):
        """Show strategy backtest tab with real Binance data"""
        from data_service.backtest.crypto_backtest import run_crypto_backtest

        st.header("🎯 Strategy Backtest")

        # ---- Presets ----
        st.subheader("⚡ Strategy Presets")
        st.caption("Backtested on 1 year of ETHUSDT 15m data with 1% SL / 3% TP")

        def _apply_bt_preset(strategy, interval, period, params, sl=1.0, tp=3.0):
            """Set widget session_state keys directly so values persist across reruns."""
            st.session_state["bt_strategy"] = strategy
            st.session_state["bt_interval"] = interval
            st.session_state["bt_period"] = period
            st.session_state["bt_sl"] = sl
            st.session_state["bt_tp"] = tp
            # Strategy-specific param keys
            for k, v in params.items():
                st.session_state[f"bt_{k}"] = v

        bp1, bp2 = st.columns(2)
        with bp1:
            if st.button("📈 BEST GROWTH — Keltner 15m\n51.6% return | 7.18% MaxDD | 54 trades/yr", key="bt_preset_growth", use_container_width=True):
                _apply_bt_preset("Keltner Channel", "15m", "1 Year",
                                 {"ema_period": 10, "atr_period": 14, "atr_mult": 3.5})
                st.rerun()
        with bp2:
            if st.button("🚀 MAX RETURN — Keltner 15m\n63.9% return | 7.75% MaxDD | 92 trades/yr", key="bt_preset_maxret", use_container_width=True):
                _apply_bt_preset("Keltner Channel", "15m", "1 Year",
                                 {"ema_period": 25, "atr_period": 10, "atr_mult": 4.0})
                st.rerun()

        # Strategy configuration — initialise defaults via session_state
        # (avoids Streamlit warning when presets also set these keys)
        st.session_state.setdefault("bt_strategy", "Momentum")
        st.session_state.setdefault("bt_interval", "1d")
        st.session_state.setdefault("bt_period", "1 Year")
        st.session_state.setdefault("bt_sl", 1.0)
        st.session_state.setdefault("bt_tp", 3.0)

        col1, col2 = st.columns(2)

        ALL_STRATEGIES = [
            "Momentum", "Mean Reversion", "Breakout", "Multi-Factor",
            "EMA Crossover", "MACD", "Bollinger Bands", "Stochastic",
            "Keltner Channel", "Williams %R", "CCI", "Triple EMA",
            "RSI+EMA", "Supertrend", "ADX Trend", "Parabolic SAR",
            "Hull MA", "Donchian ATR", "Double RSI", "VWAP Bands",
        ]

        with col1:
            st.subheader("⚙️ Strategy Parameters")

            strategy_type = st.selectbox(
                "Strategy Type",
                ALL_STRATEGIES,
                key="bt_strategy",
            )

            strategy_params = {}
            if strategy_type == "Momentum":
                strategy_params["lookback"] = st.slider("SMA Lookback Period", 5, 300, 50, key="bt_lookback")
            elif strategy_type == "Mean Reversion":
                strategy_params["rsi_period"] = st.slider("RSI Period", 5, 100, 14, key="bt_mr_rsi")
                strategy_params["oversold"] = st.slider("Oversold Threshold", 15.0, 40.0, 30.0, 1.0, key="bt_mr_os")
                strategy_params["overbought"] = st.slider("Overbought Threshold", 60.0, 85.0, 70.0, 1.0, key="bt_mr_ob")
            elif strategy_type == "Breakout":
                strategy_params["lookback"] = st.slider("Channel Lookback", 5, 300, 20, key="bt_br_lb")
            elif strategy_type == "Multi-Factor":
                strategy_params["mom_lookback"] = st.slider("Momentum Lookback", 5, 300, 50, key="bt_mf_mom")
                strategy_params["rsi_period"] = st.slider("RSI Period", 5, 100, 14, key="bt_mf_rsi")
            elif strategy_type == "EMA Crossover":
                strategy_params["fast_period"] = st.slider("Fast EMA", 3, 50, 8, key="bt_ema_fast")
                strategy_params["slow_period"] = st.slider("Slow EMA", 10, 200, 21, key="bt_ema_slow")
            elif strategy_type == "MACD":
                strategy_params["fast"] = st.slider("MACD Fast", 4, 20, 12, key="bt_macd_f")
                strategy_params["slow"] = st.slider("MACD Slow", 15, 50, 26, key="bt_macd_s")
                strategy_params["signal_period"] = st.slider("Signal Period", 3, 15, 9, key="bt_macd_sig")
            elif strategy_type == "Bollinger Bands":
                strategy_params["period"] = st.slider("BB Period", 5, 60, 20, key="bt_bb_p")
                strategy_params["num_std"] = st.slider("Num Std Dev", 1.0, 4.0, 2.0, 0.25, key="bt_bb_std")
            elif strategy_type == "Stochastic":
                strategy_params["k_period"] = st.slider("%K Period", 3, 30, 14, key="bt_sto_k")
                strategy_params["d_period"] = st.slider("%D Period", 2, 10, 3, key="bt_sto_d")
                strategy_params["oversold"] = st.slider("Oversold", 5.0, 40.0, 20.0, 5.0, key="bt_sto_os")
                strategy_params["overbought"] = st.slider("Overbought", 60.0, 95.0, 80.0, 5.0, key="bt_sto_ob")
            elif strategy_type == "Keltner Channel":
                st.session_state.setdefault("bt_ema_period", 20)
                st.session_state.setdefault("bt_atr_period", 14)
                st.session_state.setdefault("bt_atr_mult", 2.0)
                strategy_params["ema_period"] = st.slider("EMA Period", 3, 50, key="bt_ema_period")
                strategy_params["atr_period"] = st.slider("ATR Period", 3, 30, key="bt_atr_period")
                strategy_params["atr_mult"] = st.slider("ATR Multiplier", 0.5, 5.0, step=0.5, key="bt_atr_mult")
            elif strategy_type == "Williams %R":
                strategy_params["period"] = st.slider("Period", 5, 50, 14, key="bt_wr_p")
                strategy_params["oversold"] = st.slider("Oversold", -95.0, -60.0, -80.0, 5.0, key="bt_wr_os")
                strategy_params["overbought"] = st.slider("Overbought", -40.0, -5.0, -20.0, 5.0, key="bt_wr_ob")
            elif strategy_type == "CCI":
                strategy_params["period"] = st.slider("CCI Period", 5, 60, 20, key="bt_cci_p")
                strategy_params["buy_level"] = st.slider("Buy Level", -300.0, 0.0, -100.0, 25.0, key="bt_cci_buy")
                strategy_params["sell_level"] = st.slider("Sell Level", 0.0, 300.0, 100.0, 25.0, key="bt_cci_sell")
            elif strategy_type == "Triple EMA":
                strategy_params["fast"] = st.slider("Fast EMA", 3, 15, 5, key="bt_tema_f")
                strategy_params["mid"] = st.slider("Mid EMA", 10, 40, 15, key="bt_tema_m")
                strategy_params["slow"] = st.slider("Slow EMA", 20, 120, 30, key="bt_tema_s")
            elif strategy_type == "RSI+EMA":
                strategy_params["rsi_period"] = st.slider("RSI Period", 5, 30, 14, key="bt_re_rsi")
                strategy_params["ema_period"] = st.slider("EMA Period", 3, 100, 5, key="bt_re_ema")
                strategy_params["oversold"] = st.slider("Oversold", 15.0, 40.0, 30.0, 5.0, key="bt_re_os")
                strategy_params["overbought"] = st.slider("Overbought", 60.0, 85.0, 70.0, 5.0, key="bt_re_ob")
            elif strategy_type == "Supertrend":
                strategy_params["atr_period"] = st.slider("ATR Period", 5, 30, 10, key="bt_st_atr")
                strategy_params["multiplier"] = st.slider("Multiplier", 1.0, 5.0, 3.0, 0.5, key="bt_st_mult")
            elif strategy_type == "ADX Trend":
                strategy_params["adx_period"] = st.slider("ADX Period", 5, 50, 14, key="bt_adx_p")
                strategy_params["adx_threshold"] = st.slider("ADX Threshold", 15.0, 40.0, 25.0, 5.0, key="bt_adx_thr")
                strategy_params["di_period"] = st.slider("DI Period", 5, 50, 14, key="bt_adx_di")
            elif strategy_type == "Parabolic SAR":
                strategy_params["af_start"] = st.slider("AF Start", 0.01, 0.05, 0.02, 0.01, key="bt_psar_s")
                strategy_params["af_max"] = st.slider("AF Max", 0.1, 0.4, 0.2, 0.05, key="bt_psar_m")
            elif strategy_type == "Hull MA":
                strategy_params["period"] = st.slider("Hull Period", 5, 100, 20, key="bt_hull_p")
            elif strategy_type == "Donchian ATR":
                strategy_params["donchian_period"] = st.slider("Donchian Period", 10, 100, 20, key="bt_don_p")
                strategy_params["atr_period"] = st.slider("ATR Period", 5, 30, 14, key="bt_don_atr")
                strategy_params["atr_mult"] = st.slider("ATR Multiplier", 0.5, 4.0, 2.0, 0.5, key="bt_don_mult")
            elif strategy_type == "Double RSI":
                strategy_params["fast_rsi"] = st.slider("Fast RSI", 2, 10, 5, key="bt_drsi_f")
                strategy_params["slow_rsi"] = st.slider("Slow RSI", 10, 30, 14, key="bt_drsi_s")
                strategy_params["oversold"] = st.slider("Oversold", 15.0, 40.0, 30.0, 5.0, key="bt_drsi_os")
                strategy_params["overbought"] = st.slider("Overbought", 60.0, 85.0, 70.0, 5.0, key="bt_drsi_ob")
            elif strategy_type == "VWAP Bands":
                strategy_params["period"] = st.slider("VWAP Period", 10, 100, 20, key="bt_vwap_p")
                strategy_params["num_std"] = st.slider("Num Std Dev", 1.0, 4.0, 2.0, 0.25, key="bt_vwap_std")

        with col2:
            st.subheader("📊 Backtest Settings")

            commission_rate = st.slider("Commission Rate (%)", 0.0, 1.0, 0.1, 0.01, key="bt_comm") / 100

            _bt_intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
            interval = st.selectbox("Candle Interval", _bt_intervals, key="bt_interval")

            _bt_periods = ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3 Years"]
            period = st.selectbox("Backtest Period", _bt_periods, key="bt_period")
            period_days_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "3 Years": 1095}
            candles_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
            limit = period_days_map[period] * candles_per_day[interval]

            initial_capital = st.session_state.get("initial_capital", 100000)

            st.markdown("---")
            st.subheader("Futures Leverage")
            bt_leverage = st.slider("Leverage", 1, 50, 1, key="bt_leverage", help="1x = spot-equivalent. Higher = futures leverage.")
            if bt_leverage > 1:
                st.info(f"Futures mode: **{bt_leverage}x** leverage")

            st.markdown("---")
            st.subheader("Risk Management (SL/TP)")
            sl_pct = st.slider("Stop Loss %", 0.0, 10.0, step=0.5, key="bt_sl", help="0 = disabled")
            tp_pct = st.slider("Take Profit %", 0.0, 20.0, step=0.5, key="bt_tp", help="0 = disabled")

        symbols = st.session_state.get("symbols", ["BTCUSDT"])

        if st.button("🚀 Run Backtest", type="primary"):
            if not symbols:
                st.warning("Select at least one symbol in the sidebar.")
                return

            with st.spinner(f"Running {strategy_type} backtest on {', '.join(symbols)} ({interval} candles, {period}){f' {bt_leverage}x leverage' if bt_leverage > 1 else ''} with real Binance data..."):
                results = run_crypto_backtest(
                    symbols=symbols,
                    strategy_name=strategy_type,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    interval=interval,
                    limit=limit,
                    strategy_params=strategy_params,
                    stop_loss=sl_pct / 100.0,
                    take_profit=tp_pct / 100.0,
                    leverage=bt_leverage,
                )

            if "error" in results:
                st.error(f"Backtest failed: {results['error']}")
                return

            self._display_backtest_results(results)
    
    def _show_market_data(self):
        """Show market data tab with real Binance data"""
        st.header("📈 Market Data")

        # Symbol selector
        symbols = st.session_state.get('symbols', ['BTCUSDT'])
        symbol = st.selectbox("Select Symbol", symbols)

        # Timeframe selector
        timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M", "3M", "6M", "1Y"])

        # Interval selector
        interval = st.selectbox("Candle Interval", ["1h", "4h", "1d", "1w"], index=2)

        # Fetch real market data from Binance
        market_data = self._fetch_binance_data(symbol, timeframe, interval)

        if market_data is None or market_data.empty:
            st.warning(f"Could not fetch data for {symbol}. Make sure it's a valid Binance pair (e.g. BTCUSDT).")
            return

        # Price chart - candlestick
        st.subheader(f"📊 {symbol} Price Chart (Live Binance Data)")
        price_fig = go.Figure(data=[go.Candlestick(
            x=market_data.index,
            open=market_data['open'],
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close'],
            name=symbol
        )])
        price_fig.update_layout(
            title=f"{symbol} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(price_fig, use_container_width=True)

        # Technical indicators
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Technical Indicators")

            # RSI
            rsi_data = self._calculate_rsi(market_data['close'])
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=market_data.index, y=rsi_data, name='RSI', line=dict(color='purple')))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            rsi_fig.update_layout(title="RSI (14)", height=300)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with col2:
            st.subheader("📊 Volume Analysis")

            # Color volume bars based on price direction
            colors = ['green' if c >= o else 'red' for c, o in zip(market_data['close'], market_data['open'])]
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(
                x=market_data.index,
                y=market_data['volume'],
                name='Volume',
                marker_color=colors
            ))
            volume_fig.update_layout(title="Trading Volume", height=300)
            st.plotly_chart(volume_fig, use_container_width=True)

        # Market statistics
        st.subheader("📋 Market Statistics")

        col1, col2, col3, col4 = st.columns(4)

        current_price = market_data['close'].iloc[-1]
        prev_price = market_data['close'].iloc[-2] if len(market_data) > 1 else current_price

        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")

        with col2:
            daily_return = (current_price / prev_price - 1) * 100
            st.metric("Daily Return", f"{daily_return:+.2f}%")

        with col3:
            volatility = market_data['close'].pct_change().std() * np.sqrt(365) * 100
            st.metric("Annualized Vol", f"{volatility:.1f}%")

        with col4:
            avg_volume = market_data['volume'].mean()
            st.metric("Avg Volume", f"{avg_volume:,.0f}")

        # Moving averages overlay
        st.subheader("📉 Moving Averages")
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=market_data.index, y=market_data['close'], name='Close', line=dict(color='blue', width=1)))
        if len(market_data) >= 7:
            ma7 = market_data['close'].rolling(7).mean()
            ma_fig.add_trace(go.Scatter(x=market_data.index, y=ma7, name='MA7', line=dict(color='orange', width=1)))
        if len(market_data) >= 25:
            ma25 = market_data['close'].rolling(25).mean()
            ma_fig.add_trace(go.Scatter(x=market_data.index, y=ma25, name='MA25', line=dict(color='green', width=1)))
        if len(market_data) >= 99:
            ma99 = market_data['close'].rolling(99).mean()
            ma_fig.add_trace(go.Scatter(x=market_data.index, y=ma99, name='MA99', line=dict(color='red', width=1)))
        ma_fig.update_layout(title=f"{symbol} Moving Averages", height=400, yaxis_title="Price (USDT)")
        st.plotly_chart(ma_fig, use_container_width=True)
    
    def _show_ai_analysis(self):
        """Show AI analysis tab"""
        st.header("🤖 AI Analysis")
        
        # NLP Analysis
        st.subheader("📝 Sentiment Analysis")
        
        # Text input for analysis
        text_input = st.text_area(
            "Enter financial news or text for sentiment analysis:",
            value="Apple's quarterly earnings exceeded expectations, driving stock price higher by 5%! 🚀",
            height=100
        )
        
        if st.button("🔍 Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                # Process text
                processed = self.nlp_processor.preprocess_text(text_input)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", processed.sentiment_label)
                
                with col2:
                    st.metric("Confidence", f"{processed.sentiment_score:.3f}")
                
                with col3:
                    st.metric("Keywords", ", ".join(processed.keywords[:3]))
                
                # Show detailed analysis
                st.subheader("📊 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Cleaned Text:**")
                    st.write(processed.cleaned_text)
                    
                    st.write("**Topics:**")
                    st.write(", ".join(processed.topics))
                
                with col2:
                    st.write("**Language:**")
                    st.write(processed.language)
                    
                    st.write("**All Keywords:**")
                    st.write(", ".join(processed.keywords))
        
        # Factor Analysis
        st.subheader("📈 Factor Analysis")
        
        # Factor selection
        factors = st.multiselect(
            "Select Factors to Analyze",
            ["Momentum", "Value", "Quality", "Size", "Volatility", "Sentiment"],
            default=["Momentum", "Value"]
        )
        
        if st.button("📊 Analyze Factors"):
            with st.spinner("Analyzing factors..."):
                # Generate sample factor data
                factor_data = self._generate_sample_factor_data()
                
                # Display factor performance
                st.subheader("📊 Factor Performance")
                
                # Factor performance table
                factor_perf_df = pd.DataFrame([
                    ["Momentum", 0.15, 0.08, 1.88, 0.65],
                    ["Value", 0.12, 0.06, 2.00, 0.58],
                    ["Quality", 0.10, 0.05, 2.00, 0.52],
                    ["Size", 0.08, 0.07, 1.14, 0.45],
                ], columns=["Factor", "Return", "Volatility", "Sharpe", "IC"])
                
                st.dataframe(factor_perf_df, use_container_width=True)
                
                # Factor correlation heatmap
                st.subheader("🔥 Factor Correlation")
                correlation_data = np.random.rand(4, 4)
                correlation_data = (correlation_data + correlation_data.T) / 2
                np.fill_diagonal(correlation_data, 1)
                
                corr_fig = px.imshow(
                    correlation_data,
                    labels=dict(x="Factor", y="Factor", color="Correlation"),
                    x=["Momentum", "Value", "Quality", "Size"],
                    y=["Momentum", "Value", "Quality", "Size"],
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                st.plotly_chart(corr_fig, use_container_width=True)
    
    def _show_system_status(self):
        """Show system status tab"""
        st.header("⚙️ System Status")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CPU Usage", "45%", "5%")
        
        with col2:
            st.metric("Memory Usage", "2.3 GB", "0.2 GB")
        
        with col3:
            st.metric("Active Connections", "12", "2")
        
        with col4:
            st.metric("API Calls/min", "156", "23")
        
        # System health
        st.subheader("🏥 System Health")
        
        # Health indicators
        health_data = {
            "Database Connection": "✅ Healthy",
            "API Services": "✅ Healthy", 
            "Data Feeds": "✅ Healthy",
            "Strategy Engine": "✅ Healthy",
            "Risk Management": "✅ Healthy",
            "Order Execution": "⚠️ Warning",
            "Cache System": "✅ Healthy"
        }
        
        for service, status in health_data.items():
            if "✅" in status:
                st.success(f"{service}: {status}")
            elif "⚠️" in status:
                st.warning(f"{service}: {status}")
            else:
                st.error(f"{service}: {status}")
        
        # Recent logs
        st.subheader("📋 Recent Logs")
        
        logs = [
            ("2024-01-15 10:30:15", "INFO", "Strategy execution completed successfully"),
            ("2024-01-15 10:29:45", "INFO", "Market data updated for AAPL, GOOGL, MSFT"),
            ("2024-01-15 10:29:30", "WARNING", "High latency detected in order execution"),
            ("2024-01-15 10:28:15", "INFO", "Risk check passed for new order"),
            ("2024-01-15 10:27:30", "INFO", "Sentiment analysis completed for 50 news articles")
        ]
        
        log_df = pd.DataFrame(logs, columns=["Timestamp", "Level", "Message"])
        st.dataframe(log_df, use_container_width=True)
    
    # ------------------------------------------------------------------
    # Live Trading tab
    # ------------------------------------------------------------------

    @staticmethod
    def _api_base() -> str:
        """Base URL for the FastAPI web service."""
        host = os.getenv("API_HOST", "web")
        port = os.getenv("API_PORT", "8000")
        return f"http://{host}:{port}"

    def _show_live_trading(self):
        """Show live trading bot management tab."""
        st.header("🔴 Live Trading")

        # ---- Presets ----
        st.subheader("⚡ Strategy Presets")
        st.caption("Backtested on 1 year of ETHUSDT 15m data with 1% SL / 3% TP")

        def _apply_lt_preset(strategy, symbol, interval, params, max_dd=8, daily=5, sl=1.0, tp=3.0):
            """Set Live Trading widget session_state keys directly."""
            st.session_state["lt_strategy"] = strategy
            st.session_state["lt_symbol"] = symbol
            st.session_state["lt_interval"] = interval
            st.session_state["lt_max_dd"] = max_dd
            st.session_state["lt_daily"] = daily
            st.session_state["lt_sl"] = sl
            st.session_state["lt_tp"] = tp
            for k, v in params.items():
                st.session_state[f"lt_kc_{k}"] = v

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if st.button("📈 BEST GROWTH — Keltner 15m\n51.6% return | 7.18% MaxDD | 54 trades/yr", key="preset_growth", use_container_width=True):
                _apply_lt_preset("Keltner Channel", "ETHUSDT", "15m",
                                 {"ema": 10, "atr": 14, "mult": 3.5})
                st.rerun()
        with pcol2:
            if st.button("🚀 MAX RETURN — Keltner 15m\n63.9% return | 7.75% MaxDD | 92 trades/yr", key="preset_maxret", use_container_width=True):
                _apply_lt_preset("Keltner Channel", "ETHUSDT", "15m",
                                 {"ema": 25, "atr": 10, "mult": 4.0})
                st.rerun()

        # ---- Bot configuration panel ----
        st.subheader("🛠️ Start a New Bot")

        col1, col2, col3 = st.columns(3)

        LT_STRATEGIES = [
            "Momentum", "Mean Reversion", "Breakout", "Multi-Factor",
            "EMA Crossover", "MACD", "Bollinger Bands", "Stochastic",
            "Keltner Channel", "Williams %R", "CCI", "Triple EMA",
            "RSI+EMA", "Supertrend", "ADX Trend", "Parabolic SAR",
            "Hull MA", "Donchian ATR", "Double RSI", "VWAP Bands",
        ]

        with col1:
            lt_strategy = st.selectbox(
                "Strategy",
                LT_STRATEGIES,
                key="lt_strategy",
            )
            lt_symbol = st.text_input("Symbol", value="ETHUSDT", key="lt_symbol")
            _intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
            lt_interval = st.selectbox(
                "Candle Interval",
                _intervals,
                index=4,
                key="lt_interval",
            )

        with col2:
            lt_params: Dict[str, Any] = {}
            if lt_strategy == "Momentum":
                lt_params["lookback"] = st.slider("SMA Lookback", 5, 300, 168, key="lt_lookback")
            elif lt_strategy == "Mean Reversion":
                lt_params["rsi_period"] = st.slider("RSI Period", 5, 100, 14, key="lt_rsi")
                lt_params["oversold"] = st.slider("Oversold", 15.0, 40.0, 30.0, 1.0, key="lt_os")
                lt_params["overbought"] = st.slider("Overbought", 60.0, 85.0, 70.0, 1.0, key="lt_ob")
            elif lt_strategy == "Breakout":
                lt_params["lookback"] = st.slider("Channel Lookback", 5, 300, 20, key="lt_br_lb")
            elif lt_strategy == "Multi-Factor":
                lt_params["mom_lookback"] = st.slider("Momentum Lookback", 5, 300, 50, key="lt_mf_mom")
                lt_params["rsi_period"] = st.slider("RSI Period", 5, 100, 14, key="lt_mf_rsi")
            elif lt_strategy == "EMA Crossover":
                lt_params["fast_period"] = st.slider("Fast EMA", 3, 50, 8, key="lt_ema_fast")
                lt_params["slow_period"] = st.slider("Slow EMA", 10, 200, 21, key="lt_ema_slow")
            elif lt_strategy == "MACD":
                lt_params["fast"] = st.slider("MACD Fast", 4, 20, 12, key="lt_macd_f")
                lt_params["slow"] = st.slider("MACD Slow", 15, 50, 26, key="lt_macd_s")
                lt_params["signal_period"] = st.slider("Signal Period", 3, 15, 9, key="lt_macd_sig")
            elif lt_strategy == "Bollinger Bands":
                lt_params["period"] = st.slider("BB Period", 5, 60, 20, key="lt_bb_p")
                lt_params["num_std"] = st.slider("Num Std Dev", 1.0, 4.0, 2.0, 0.25, key="lt_bb_std")
            elif lt_strategy == "Stochastic":
                lt_params["k_period"] = st.slider("%K Period", 3, 30, 14, key="lt_sto_k")
                lt_params["d_period"] = st.slider("%D Period", 2, 10, 3, key="lt_sto_d")
                lt_params["oversold"] = st.slider("Oversold", 5.0, 40.0, 20.0, 5.0, key="lt_sto_os")
                lt_params["overbought"] = st.slider("Overbought", 60.0, 95.0, 80.0, 5.0, key="lt_sto_ob")
            elif lt_strategy == "Keltner Channel":
                lt_params["ema_period"] = st.slider("EMA Period", 3, 50, 20, key="lt_kc_ema")
                lt_params["atr_period"] = st.slider("ATR Period", 3, 30, 14, key="lt_kc_atr")
                lt_params["atr_mult"] = st.slider("ATR Multiplier", 0.5, 5.0, 2.0, 0.5, key="lt_kc_mult")
            elif lt_strategy == "Williams %R":
                lt_params["period"] = st.slider("Period", 5, 50, 14, key="lt_wr_p")
                lt_params["oversold"] = st.slider("Oversold", -95.0, -60.0, -80.0, 5.0, key="lt_wr_os")
                lt_params["overbought"] = st.slider("Overbought", -40.0, -5.0, -20.0, 5.0, key="lt_wr_ob")
            elif lt_strategy == "CCI":
                lt_params["period"] = st.slider("CCI Period", 5, 60, 20, key="lt_cci_p")
                lt_params["buy_level"] = st.slider("Buy Level", -300.0, 0.0, -100.0, 25.0, key="lt_cci_buy")
                lt_params["sell_level"] = st.slider("Sell Level", 0.0, 300.0, 100.0, 25.0, key="lt_cci_sell")
            elif lt_strategy == "Triple EMA":
                lt_params["fast"] = st.slider("Fast EMA", 3, 15, 5, key="lt_tema_f")
                lt_params["mid"] = st.slider("Mid EMA", 10, 40, 15, key="lt_tema_m")
                lt_params["slow"] = st.slider("Slow EMA", 20, 120, 30, key="lt_tema_s")
            elif lt_strategy == "RSI+EMA":
                lt_params["rsi_period"] = st.slider("RSI Period", 5, 30, 14, key="lt_re_rsi")
                lt_params["ema_period"] = st.slider("EMA Period", 3, 100, 5, key="lt_re_ema")
                lt_params["oversold"] = st.slider("Oversold", 15.0, 40.0, 30.0, 5.0, key="lt_re_os")
                lt_params["overbought"] = st.slider("Overbought", 60.0, 85.0, 70.0, 5.0, key="lt_re_ob")
            elif lt_strategy == "Supertrend":
                lt_params["atr_period"] = st.slider("ATR Period", 5, 30, 10, key="lt_st_atr")
                lt_params["multiplier"] = st.slider("Multiplier", 1.0, 5.0, 3.0, 0.5, key="lt_st_mult")
            elif lt_strategy == "ADX Trend":
                lt_params["adx_period"] = st.slider("ADX Period", 5, 50, 14, key="lt_adx_p")
                lt_params["adx_threshold"] = st.slider("ADX Threshold", 15.0, 40.0, 25.0, 5.0, key="lt_adx_thr")
                lt_params["di_period"] = st.slider("DI Period", 5, 50, 14, key="lt_adx_di")
            elif lt_strategy == "Parabolic SAR":
                lt_params["af_start"] = st.slider("AF Start", 0.01, 0.05, 0.02, 0.01, key="lt_psar_s")
                lt_params["af_max"] = st.slider("AF Max", 0.1, 0.4, 0.2, 0.05, key="lt_psar_m")
            elif lt_strategy == "Hull MA":
                lt_params["period"] = st.slider("Hull Period", 5, 100, 20, key="lt_hull_p")
            elif lt_strategy == "Donchian ATR":
                lt_params["donchian_period"] = st.slider("Donchian Period", 10, 100, 20, key="lt_don_p")
                lt_params["atr_period"] = st.slider("ATR Period", 5, 30, 14, key="lt_don_atr")
                lt_params["atr_mult"] = st.slider("ATR Multiplier", 0.5, 4.0, 2.0, 0.5, key="lt_don_mult")
            elif lt_strategy == "Double RSI":
                lt_params["fast_rsi"] = st.slider("Fast RSI", 2, 10, 5, key="lt_drsi_f")
                lt_params["slow_rsi"] = st.slider("Slow RSI", 10, 30, 14, key="lt_drsi_s")
                lt_params["oversold"] = st.slider("Oversold", 15.0, 40.0, 30.0, 5.0, key="lt_drsi_os")
                lt_params["overbought"] = st.slider("Overbought", 60.0, 85.0, 70.0, 5.0, key="lt_drsi_ob")
            elif lt_strategy == "VWAP Bands":
                lt_params["period"] = st.slider("VWAP Period", 10, 100, 20, key="lt_vwap_p")
                lt_params["num_std"] = st.slider("Num Std Dev", 1.0, 4.0, 2.0, 0.25, key="lt_vwap_std")

        with col3:
            lt_capital = st.number_input("Margin (USDT)", min_value=10.0, value=100.0, step=10.0, key="lt_capital")
            lt_leverage = st.slider("Leverage", 1, 50, 5, key="lt_leverage")
            lt_margin_type = st.selectbox("Margin Type", ["ISOLATED", "CROSSED"], key="lt_margin")
            lt_max_dd = st.slider("Max Drawdown %", 1, 50, 15, key="lt_max_dd") / 100.0
            lt_daily = st.slider("Daily Loss Limit %", 1, 20, 5, key="lt_daily") / 100.0

            st.markdown("---")
            st.markdown("**Risk Management (SL/TP)**")
            lt_sl = st.slider("Stop Loss %", 0.0, 10.0, 1.0, 0.5, key="lt_sl", help="0 = disabled. Per-trade SL from entry price.")
            lt_tp = st.slider("Take Profit %", 0.0, 20.0, 3.0, 0.5, key="lt_tp", help="0 = disabled. Per-trade TP from entry price.")

        st.info(f"Effective position size: **${lt_capital * lt_leverage:,.0f}** ({lt_leverage}x leverage on ${lt_capital:,.0f} margin)")

        if st.button("🚀 Start Futures Bot", type="primary", key="lt_start"):
            payload = {
                "strategy_name": lt_strategy,
                "symbol": lt_symbol.upper(),
                "interval": lt_interval,
                "strategy_params": lt_params,
                "capital": lt_capital,
                "leverage": lt_leverage,
                "margin_type": lt_margin_type,
                "max_drawdown": lt_max_dd,
                "daily_loss_limit": lt_daily,
                "stop_loss": lt_sl / 100.0,
                "take_profit": lt_tp / 100.0,
            }
            try:
                resp = requests.post(f"{self._api_base()}/api/bot/start", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Bot started: {data['bot']['bot_id']}")
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")

        st.divider()

        # ---- Active bots list ----
        st.subheader("📋 Active Bots")

        if st.button("🔄 Refresh", key="lt_refresh"):
            pass  # triggers rerun

        try:
            resp = requests.get(f"{self._api_base()}/api/bot/list", timeout=10)
            if resp.status_code == 200:
                bots = resp.json().get("bots", [])
            else:
                bots = []
                st.warning(f"API returned {resp.status_code}")
        except Exception as e:
            bots = []
            st.info(f"API not reachable ({e}). Start the web service first.")

        if bots:
            for bot in bots:
                status_icon = {"running": "🟢", "paused": "🟡", "stopped": "🔴", "error": "❌"}.get(bot.get("status", ""), "⚪")
                lev = bot.get("leverage", 1)
                pos_side = bot.get("position_side", "FLAT")
                with st.expander(f"{status_icon} {bot['bot_id']} — {bot['strategy']} {bot['symbol']} {lev}x ({bot['status']})"):
                    # Metrics row
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Equity", f"${bot.get('equity', 0):,.2f}")
                    m2.metric("Margin", f"${bot.get('cash', 0):,.2f}")
                    side_icon = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "FLAT": "⚪ FLAT"}.get(pos_side, pos_side)
                    m3.metric("Side", side_icon)
                    m4.metric("Size", f"{bot.get('position_qty', 0):.4f}")
                    m5.metric("Unrealized P&L", f"${bot.get('unrealized_pnl', 0):,.2f}")
                    m6.metric("Trades", str(bot.get("total_trades", 0)))

                    # Control buttons
                    bc1, bc2, bc3 = st.columns(3)
                    bid = bot["bot_id"]
                    if bot["status"] == "running":
                        if bc1.button("⏸ Pause", key=f"pause_{bid}"):
                            requests.post(f"{self._api_base()}/api/bot/{bid}/pause", timeout=10)
                            st.rerun()
                        if bc2.button("⏹ Stop", key=f"stop_{bid}"):
                            requests.post(f"{self._api_base()}/api/bot/{bid}/stop", timeout=10)
                            st.rerun()
                    elif bot["status"] == "paused":
                        if bc1.button("▶ Resume", key=f"resume_{bid}"):
                            requests.post(f"{self._api_base()}/api/bot/{bid}/resume", timeout=10)
                            st.rerun()
                        if bc2.button("⏹ Stop", key=f"stopP_{bid}"):
                            requests.post(f"{self._api_base()}/api/bot/{bid}/stop", timeout=10)
                            st.rerun()

                    if bot.get("error_message"):
                        st.error(f"Error: {bot['error_message']}")

                    _sl_pct = bot.get('stop_loss', 0) * 100
                    _tp_pct = bot.get('take_profit', 0) * 100
                    _sltp_str = f"SL: {_sl_pct:.1f}% / TP: {_tp_pct:.1f}%" if _sl_pct > 0 or _tp_pct > 0 else "SL/TP: disabled"
                    st.caption(f"{_sltp_str}  |  Created: {bot.get('created_at', 'N/A')}  |  Last update: {bot.get('last_update', 'N/A')}")

                    # Trade history
                    try:
                        tr = requests.get(f"{self._api_base()}/api/bot/{bid}/trades", timeout=10)
                        trades = tr.json().get("trades", []) if tr.status_code == 200 else []
                    except Exception:
                        trades = []
                    if trades:
                        st.write("**Trade History**")
                        st.dataframe(pd.DataFrame(trades), use_container_width=True)

                    # Equity curve
                    try:
                        eq = requests.get(f"{self._api_base()}/api/bot/{bid}/equity", timeout=10)
                        eq_data = eq.json().get("equity", []) if eq.status_code == 200 else []
                    except Exception:
                        eq_data = []
                    if len(eq_data) >= 2:
                        st.write("**Equity Curve**")
                        eq_df = pd.DataFrame(eq_data)
                        eq_df["time"] = pd.to_datetime(eq_df["time"])
                        eq_fig = go.Figure()
                        eq_fig.add_trace(go.Scatter(
                            x=eq_df["time"], y=eq_df["equity"],
                            mode="lines", name="Equity",
                            line=dict(color="royalblue", width=2),
                        ))
                        eq_fig.update_layout(height=300, yaxis_title="Equity (USDT)")
                        st.plotly_chart(eq_fig, use_container_width=True)
        else:
            st.info("No bots found. Start one above!")

        st.divider()

        # ---- Binance Futures account ----
        st.subheader("💰 Binance Futures Account")
        try:
            resp = requests.get(f"{self._api_base()}/api/bot/account", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    st.warning(f"Could not fetch account: {data['error']}")
                else:
                    # Balances
                    balances = data.get("balances", [])
                    if balances:
                        st.write("**Wallet Balances**")
                        st.dataframe(pd.DataFrame(balances), use_container_width=True)

                    # Open positions
                    positions = data.get("positions", [])
                    if positions:
                        st.write("**Open Futures Positions**")
                        st.dataframe(pd.DataFrame(positions), use_container_width=True)
                    else:
                        st.info("No open futures positions.")
            else:
                st.warning(f"API returned {resp.status_code}")
        except Exception as e:
            st.info(f"Could not fetch account info ({e}). Set BINANCE_API_KEY in .env.")

    def _generate_sample_performance_data(self):
        """Generate sample performance data for demonstration"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
        np.random.seed(42)
        
        # Generate equity curve
        returns = np.random.normal(0.0005, 0.02, len(dates))
        equity = 100000 * np.cumprod(1 + returns)
        equity_data = pd.DataFrame({'equity': equity}, index=dates)
        
        # Calculate metrics
        total_return = (equity[-1] / equity[0] - 1)
        annualized_return = total_return * 252 / len(dates)
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        equity_series = pd.Series(equity, index=dates)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        
        return {
            'equity_data': equity_data,
            'drawdown_data': drawdown,
            'returns': pd.Series(returns, index=dates),
            'total_return': total_return,
            'total_return_delta': 0.05,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_delta': 0.1,
            'sortino_ratio': sharpe_ratio * 1.1,
            'max_drawdown': drawdown.min(),
            'drawdown_delta': 0.02,
            'calmar_ratio': annualized_return / abs(drawdown.min()) if drawdown.min() != 0 else 0,
            'win_rate': 0.58,
            'win_rate_delta': 0.03,
            'profit_factor': 1.45,
            'total_trades': 156
        }
    
    def _generate_sample_backtest_results(self):
        """Generate sample backtest results"""
        return {
            'total_return': 0.25,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.12,
            'win_rate': 0.65,
            'total_trades': 89,
            'equity_curve': self._generate_sample_performance_data()['equity_data']
        }
    
    @st.cache_data(ttl=300)
    def _fetch_binance_data(_self, symbol, timeframe="1Y", interval="1d"):
        """Fetch real OHLCV data from Binance public API"""
        # Map timeframe to number of candles
        timeframe_map = {
            "1D": 24, "1W": 168, "1M": 30, "3M": 90, "6M": 180, "1Y": 365
        }
        limit = timeframe_map.get(timeframe, 365)

        # Adjust limit for hourly intervals
        if interval in ("1h", "4h"):
            hour_multiplier = {"1h": 1, "4h": 4}
            days = timeframe_map.get(timeframe, 365)
            limit = min(int(days * 24 / hour_multiplier.get(interval, 1)), 1000)

        # Normalize symbol for Binance
        pair = symbol.upper()
        if not pair.endswith(("USDT", "BTC", "ETH", "BUSD")):
            pair = pair + "USDT"

        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Binance API error for {pair}: {resp.status_code}")
                return None
            klines = resp.json()

            data = []
            for k in klines:
                ts_ms = k[0]
                dt = datetime.utcfromtimestamp(ts_ms / 1000)
                data.append({
                    "date": dt,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            df = pd.DataFrame(data)
            df.set_index("date", inplace=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch Binance data for {pair}: {e}")
            return None

    def _generate_sample_market_data(self, symbol):
        """Generate sample market data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
        np.random.seed(42)
        
        # Generate price data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        # Generate volume data
        volume = np.random.lognormal(10, 0.5, len(dates))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': volume
        }, index=dates)
    
    def _generate_sample_factor_data(self):
        """Generate sample factor data"""
        return pd.DataFrame({
            'momentum': np.random.normal(0, 1, 100),
            'value': np.random.normal(0, 1, 100),
            'quality': np.random.normal(0, 1, 100),
            'size': np.random.normal(0, 1, 100)
        })
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _display_backtest_results(self, results):
        """Display real backtest results"""
        lev = results.get("leverage", 1)
        lev_str = f" ({lev}x leverage)" if lev > 1 else ""
        st.success(f"Backtest completed — {results['strategy_name']} on {', '.join(results['symbols'])}{lev_str}")
        if lev > 1:
            liqs = results.get("liquidation_count", 0)
            if liqs > 0:
                st.warning(f"Liquidations: {liqs}")
            st.caption(f"Futures mode: {lev}x leverage | Effective position size: {lev}x capital")

        # Key metrics row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{results['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
        with col4:
            st.metric("Win Rate", f"{results['win_rate']:.1%}")

        # Key metrics row 2
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Value", f"${results['final_value']:,.2f}")
        with col2:
            st.metric("Sortino Ratio", f"{results['sortino_ratio']:.2f}")
        with col3:
            st.metric("Calmar Ratio", f"{results['calmar_ratio']:.2f}")
        with col4:
            st.metric("Profit Factor", f"{results['profit_factor']:.2f}")

        # Equity curve chart
        st.subheader("📈 Equity Curve")
        eq_data = results["equity_curve"]
        eq_df = pd.DataFrame(eq_data)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=eq_df["date"], y=eq_df["value"],
            mode="lines", name="Portfolio Value",
            line=dict(color="royalblue", width=2),
            fill="tozeroy", fillcolor="rgba(65,105,225,0.1)"
        ))
        eq_fig.add_hline(y=results["initial_capital"], line_dash="dash", line_color="gray",
                         annotation_text=f"Initial: ${results['initial_capital']:,.0f}")
        eq_fig.update_layout(height=400, yaxis_title="Portfolio Value ($)", xaxis_title="Date")
        st.plotly_chart(eq_fig, use_container_width=True)

        # Drawdown chart
        st.subheader("📉 Drawdown")
        dd_data = results.get("drawdown_series", [])
        if dd_data:
            dd_df = pd.DataFrame(dd_data)
            dd_df["date"] = pd.to_datetime(dd_df["date"])
            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(
                x=dd_df["date"], y=dd_df["drawdown"],
                mode="lines", name="Drawdown",
                line=dict(color="crimson", width=1),
                fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
            ))
            dd_fig.update_layout(height=300, yaxis_title="Drawdown", yaxis_tickformat=".1%")
            st.plotly_chart(dd_fig, use_container_width=True)

        # Trade log
        trades = results.get("trades", [])
        if trades:
            st.subheader(f"📋 Trade Log ({len(trades)} trades)")
            trades_df = pd.DataFrame(trades)
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            st.dataframe(trades_df, use_container_width=True)

        # Detailed metrics table
        st.subheader("Detailed Metrics")
        detail_rows = [
            ["Initial Capital", f"${results['initial_capital']:,.2f}"],
            ["Leverage", f"{results.get('leverage', 1)}x"],
            ["Final Value", f"${results['final_value']:,.2f}"],
            ["Total Return", f"{results['total_return']:.2%}"],
            ["Annualized Return", f"{results['annualized_return']:.2%}"],
            ["Volatility", f"{results['volatility']:.2%}"],
            ["Sharpe Ratio", f"{results['sharpe_ratio']:.2f}"],
            ["Sortino Ratio", f"{results['sortino_ratio']:.2f}"],
            ["Max Drawdown", f"{results['max_drawdown']:.2%}"],
            ["Calmar Ratio", f"{results['calmar_ratio']:.2f}"],
            ["Win Rate", f"{results['win_rate']:.1%}"],
            ["Profit Factor", f"{results['profit_factor']:.2f}"],
            ["Total Trades", str(results['total_trades'])],
            ["Winning Trades", str(results['winning_trades'])],
            ["Losing Trades", str(results['losing_trades'])],
            ["Avg Win", f"${results['avg_win']:,.2f}"],
            ["Avg Loss", f"${results['avg_loss']:,.2f}"],
        ]
        if results.get("liquidation_count", 0) > 0:
            detail_rows.append(["Liquidations", str(results['liquidation_count'])])
        metrics_df = pd.DataFrame(detail_rows, columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = TradingDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Error running dashboard: {e}")
        logger.exception("Dashboard error")

main() 