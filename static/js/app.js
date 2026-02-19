// QuantMuse Trading System Dashboard

class TradingDashboard {
    constructor() {
        this.apiBaseUrl = '/api';
        this.currentPage = 'dashboard';
        this.charts = {};
        this.candlestickChart = null;
        this.candlestickSeries = null;
        this.smaSeries = null;
        this.emaSeries = null;
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupForms();
        this.setupCharts();
        this.loadDashboardData();
    }

    // --- Navigation ---
    setupNavigation() {
        document.querySelectorAll('[data-page]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.currentTarget.getAttribute('data-page');
                this.navigateToPage(page);
            });
        });

        document.getElementById('load-chart-btn')?.addEventListener('click', () => {
            this.loadCandlestickChart();
        });

        document.querySelectorAll('#show-sma, #show-ema').forEach(cb => {
            cb.addEventListener('change', () => this.updateChartIndicators());
        });

        setInterval(() => {
            if (this.currentPage === 'dashboard') this.loadDashboardData();
        }, 30000);
    }

    setupForms() {
        document.getElementById('backtest-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.runBacktest();
        });

        document.getElementById('sentiment-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeSentiment();
        });

        document.getElementById('strategy-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.showNotification('Strategy saved (demo mode)', 'success');
        });
    }

    navigateToPage(page) {
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        const target = document.getElementById(`${page}-page`);
        if (target) target.classList.add('active');

        document.querySelectorAll('[data-page]').forEach(link => link.classList.remove('active'));
        document.querySelector(`[data-page="${page}"]`)?.classList.add('active');

        this.currentPage = page;

        switch (page) {
            case 'dashboard': this.loadDashboardData(); break;
            case 'strategies': this.loadStrategies(); break;
            case 'portfolio': this.loadPortfolioData(); break;
            case 'charts':
                this.initCandlestickChart();
                this.loadCandlestickChart();
                break;
        }
    }

    // --- Dashboard ---
    async loadDashboardData() {
        try {
            const [statusRes, tradesRes, portfolioRes] = await Promise.all([
                fetch(`${this.apiBaseUrl}/system/status`),
                fetch(`${this.apiBaseUrl}/trades/recent?limit=10`),
                fetch(`${this.apiBaseUrl}/portfolio/status`)
            ]);

            const status = await statusRes.json();
            const trades = await tradesRes.json();
            const portfolio = await portfolioRes.json();

            this.setText('system-status', status.status || 'Running');
            this.setText('active-strategies', status.active_strategies);
            this.setText('total-trades', (status.total_trades || 0).toLocaleString());
            this.setText('portfolio-value', `$${(portfolio.total_value || 0).toLocaleString()}`);

            this.renderTradesTable(trades.trades || []);
        } catch (error) {
            console.error('Dashboard load error:', error);
        }
    }

    renderTradesTable(trades) {
        const tbody = document.getElementById('trades-table');
        if (!tbody) return;

        tbody.innerHTML = trades.map(t => {
            const sideClass = t.side === 'buy' ? 'text-success' : 'text-danger';
            return `<tr>
                <td>${this.formatTime(t.timestamp)}</td>
                <td><strong>${t.symbol}</strong></td>
                <td class="${sideClass}">${t.side.toUpperCase()}</td>
                <td>${t.quantity}</td>
                <td>$${t.price.toFixed(2)}</td>
                <td><span class="badge bg-success">${t.status}</span></td>
            </tr>`;
        }).join('');
    }

    // --- Charts (Dashboard) ---
    setupCharts() {
        this.setupEquityChart();
        this.setupAllocationChart();
    }

    setupEquityChart() {
        const ctx = document.getElementById('equity-chart')?.getContext('2d');
        if (!ctx) return;

        this.charts.equity = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#1f77b4',
                    backgroundColor: 'rgba(31, 119, 180, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: true, ticks: { maxTicksLimit: 10 } },
                    y: { display: true, title: { display: true, text: 'Value ($)' } }
                }
            }
        });

        this.loadEquityData();
    }

    setupAllocationChart() {
        const ctx = document.getElementById('allocation-chart')?.getContext('2d');
        if (!ctx) return;

        fetch(`${this.apiBaseUrl}/portfolio/status`)
            .then(r => r.json())
            .then(data => {
                const labels = (data.positions || []).map(p => p.symbol);
                const values = (data.positions || []).map(p => p.value);
                labels.push('Cash');
                values.push(data.cash || 0);

                this.charts.allocation = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels,
                        datasets: [{
                            data: values,
                            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'],
                            borderWidth: 2,
                            borderColor: '#fff'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'bottom', labels: { padding: 15, usePointStyle: true } } }
                    }
                });
            })
            .catch(err => console.error('Allocation chart error:', err));
    }

    async loadEquityData() {
        try {
            const res = await fetch(`${this.apiBaseUrl}/market/data/BTCUSDT?period=6m`);
            const data = await res.json();
            if (this.charts.equity && data.data) {
                this.charts.equity.data.labels = data.data.map(d => d.date);
                this.charts.equity.data.datasets[0].data = data.data.map(d => d.close);
                this.charts.equity.data.datasets[0].label = 'BTC/USDT';
                this.charts.equity.update();
            }
        } catch (error) {
            console.error('Equity data error:', error);
        }
    }

    // --- Strategies ---
    async loadStrategies() {
        try {
            const res = await fetch(`${this.apiBaseUrl}/strategies`);
            const data = await res.json();
            const container = document.getElementById('strategies-list');
            if (!container) return;

            container.innerHTML = (data.strategies || []).map(s => `
                <div class="col-md-6 mb-3">
                    <div class="card strategy-card h-100">
                        <div class="card-body">
                            <h5 class="card-title">${s.name}</h5>
                            <p class="card-text text-muted">${s.description}</p>
                            <span class="badge bg-success">Available</span>
                        </div>
                    </div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Strategies error:', error);
        }
    }

    // --- Portfolio ---
    async loadPortfolioData() {
        try {
            const res = await fetch(`${this.apiBaseUrl}/portfolio/status`);
            const data = await res.json();

            this.setText('portfolio-total-value', `$${(data.total_value || 0).toLocaleString()}`);
            this.setText('portfolio-cash', `$${(data.cash || 0).toLocaleString()}`);

            const dailyClass = (data.daily_pnl || 0) >= 0 ? 'text-success' : 'text-danger';
            const totalClass = (data.total_pnl || 0) >= 0 ? 'text-success' : 'text-danger';
            this.setHTML('portfolio-daily-pnl', `<span class="${dailyClass}">$${(data.daily_pnl || 0).toLocaleString()}</span>`);
            this.setHTML('portfolio-total-pnl', `<span class="${totalClass}">$${(data.total_pnl || 0).toLocaleString()}</span>`);

            const tbody = document.getElementById('positions-table');
            if (tbody) {
                tbody.innerHTML = (data.positions || []).map(p => {
                    const pnlClass = p.pnl >= 0 ? 'text-success' : 'text-danger';
                    const sign = p.pnl >= 0 ? '+' : '';
                    return `<tr>
                        <td><strong>${p.symbol}</strong></td>
                        <td>${p.quantity}</td>
                        <td>$${p.value.toLocaleString()}</td>
                        <td class="${pnlClass}">${sign}$${p.pnl.toLocaleString()}</td>
                    </tr>`;
                }).join('');
            }
        } catch (error) {
            console.error('Portfolio error:', error);
        }
    }

    // --- Candlestick Charts ---
    initCandlestickChart() {
        try {
            const container = document.getElementById('candlestick-chart');
            if (!container) return;

            // Destroy previous chart if exists
            if (this.candlestickChart) {
                this.candlestickChart.remove();
                this.candlestickChart = null;
                this.candlestickSeries = null;
                this.smaSeries = null;
                this.emaSeries = null;
            }

            container.innerHTML = '';

            const width = container.clientWidth || 800;
            this.candlestickChart = LightweightCharts.createChart(container, {
                width: width,
                height: 450,
                layout: { backgroundColor: '#ffffff', textColor: '#333' },
                grid: {
                    vertLines: { color: '#f0f0f0' },
                    horzLines: { color: '#f0f0f0' }
                },
                rightPriceScale: { borderColor: '#ccc' },
                timeScale: { borderColor: '#ccc' }
            });

            this.candlestickSeries = this.candlestickChart.addCandlestickSeries({
                upColor: '#26a69a', downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a', wickDownColor: '#ef5350'
            });

            this.smaSeries = this.candlestickChart.addLineSeries({
                color: '#2196F3', lineWidth: 2
            });

            this.emaSeries = this.candlestickChart.addLineSeries({
                color: '#FF9800', lineWidth: 2
            });

            window.addEventListener('resize', () => {
                if (this.candlestickChart && container.clientWidth > 0) {
                    this.candlestickChart.applyOptions({ width: container.clientWidth });
                }
            });
        } catch (error) {
            console.error('Chart init error:', error);
        }
    }

    async loadCandlestickChart() {
        const symbol = document.getElementById('symbol-input')?.value || 'BTCUSDT';
        const timeframe = document.getElementById('timeframe-select')?.value || '1y';

        try {
            const res = await fetch(`${this.apiBaseUrl}/market/data/${symbol}?period=${timeframe}`);
            if (!res.ok) {
                const err = await res.json();
                this.showNotification(err.detail || 'Failed to load data', 'danger');
                return;
            }
            const data = await res.json();

            if (data.data && data.data.length > 0) {
                // Use real OHLCV from Binance
                const candles = data.data.map(item => ({
                    time: item.date,
                    open: item.open,
                    high: item.high,
                    low: item.low,
                    close: item.close
                }));

                if (this.candlestickSeries) {
                    this.candlestickSeries.setData(candles);
                }

                this.addMovingAverages(candles);

                if (this.candlestickChart) {
                    this.candlestickChart.timeScale().fitContent();
                }

                // Update chart info with real data
                const last = data.data[data.data.length - 1];
                const prev = data.data[data.data.length - 2];
                this.setText('current-symbol', data.symbol || symbol);
                this.setText('current-price', `$${last.close.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`);

                const change = ((last.close - prev.close) / prev.close * 100);
                const changeEl = document.getElementById('price-change');
                if (changeEl) {
                    changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                    changeEl.className = change >= 0 ? 'text-success' : 'text-danger';
                }
                this.setText('current-volume', last.volume.toLocaleString(undefined, {maximumFractionDigits: 0}));
            }
        } catch (error) {
            console.error('Chart data error:', error);
            this.showNotification('Failed to load chart data', 'danger');
        }
    }

    addMovingAverages(data) {
        if (!data || data.length < 20) return;
        const period = 20;

        const smaData = [];
        for (let i = period - 1; i < data.length; i++) {
            const sum = data.slice(i - period + 1, i + 1).reduce((a, d) => a + d.close, 0);
            smaData.push({ time: data[i].time, value: sum / period });
        }

        const emaData = [];
        const k = 2 / (period + 1);
        let ema = data[0].close;
        for (let i = 0; i < data.length; i++) {
            ema = data[i].close * k + ema * (1 - k);
            emaData.push({ time: data[i].time, value: ema });
        }

        if (this.smaSeries) this.smaSeries.setData(smaData);
        if (this.emaSeries) this.emaSeries.setData(emaData);
    }

    updateChartIndicators() {
        const showSMA = document.getElementById('show-sma')?.checked;
        const showEMA = document.getElementById('show-ema')?.checked;
        if (this.smaSeries) this.smaSeries.applyOptions({ visible: showSMA });
        if (this.emaSeries) this.emaSeries.applyOptions({ visible: showEMA });
    }

    // --- Backtest ---
    async runBacktest() {
        const strategy = document.getElementById('backtest-strategy')?.value || 'Momentum Strategy';
        const capital = parseFloat(document.getElementById('backtest-capital')?.value || 100000);
        const symbol = document.getElementById('backtest-symbol')?.value || 'BTCUSDT';
        const periodDays = parseInt(document.getElementById('backtest-period')?.value || '365');
        const interval = document.getElementById('backtest-interval')?.value || '1d';
        const lookback = parseInt(document.getElementById('backtest-lookback')?.value || '50');

        // Convert period in days to candle count based on interval
        const candlesPerDay = {'5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1};
        const period = String(periodDays * (candlesPerDay[interval] || 1));

        const div = document.getElementById('backtest-results');
        if (div) div.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2">Running backtest with real Binance data...</p></div>';

        try {
            const res = await fetch(`${this.apiBaseUrl}/backtest/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    strategy_config: {
                        strategy_name: strategy,
                        symbols: [symbol],
                        parameters: {interval: interval, lookback: lookback, mom_lookback: lookback},
                        start_date: '2025-01-01',
                        end_date: '2026-01-01',
                        initial_capital: capital
                    },
                    commission_rate: 0.001,
                    rebalance_frequency: period
                })
            });

            const result = await res.json();
            if (result.status !== 'success' || !result.results) {
                div.innerHTML = `<div class="alert alert-danger">Backtest failed: ${result.detail || 'Unknown error'}</div>`;
                return;
            }

            const r = result.results;
            const retClass = r.total_return >= 0 ? 'success' : 'danger';
            const ddClass = 'danger';

            let html = `
                <div class="alert alert-${retClass} mb-3">
                    <h5>📊 ${r.strategy_name} — ${r.symbols.join(', ')}</h5>
                    <small>Real backtest on Binance historical data | Initial: $${r.initial_capital.toLocaleString()} → Final: $${r.final_value.toLocaleString()}</small>
                </div>
                <div class="row g-3 mb-3">
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Total Return</small><h5 class="text-${retClass}">${(r.total_return * 100).toFixed(2)}%</h5></div></div>
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Sharpe Ratio</small><h5>${r.sharpe_ratio.toFixed(2)}</h5></div></div>
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Max Drawdown</small><h5 class="text-${ddClass}">${(r.max_drawdown * 100).toFixed(2)}%</h5></div></div>
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Win Rate</small><h5>${(r.win_rate * 100).toFixed(1)}%</h5></div></div>
                </div>
                <div class="row g-3 mb-3">
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Sortino Ratio</small><h5>${r.sortino_ratio.toFixed(2)}</h5></div></div>
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Calmar Ratio</small><h5>${r.calmar_ratio.toFixed(2)}</h5></div></div>
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Profit Factor</small><h5>${r.profit_factor.toFixed(2)}</h5></div></div>
                    <div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Total Trades</small><h5>${r.total_trades}</h5></div></div>
                </div>`;

            // Equity curve chart
            html += '<div id="backtest-equity-chart" style="height:350px;margin-bottom:20px;"></div>';

            // Trade log
            if (r.trades && r.trades.length > 0) {
                html += `<h6>Trade Log (${r.trades.length} trades)</h6>
                <div style="max-height:250px;overflow-y:auto;">
                <table class="table table-sm table-striped"><thead><tr>
                    <th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>P&L</th>
                </tr></thead><tbody>`;
                for (const t of r.trades) {
                    const sideClass = t.side === 'buy' ? 'text-success' : 'text-danger';
                    const pnl = t.pnl !== undefined ? `$${t.pnl.toFixed(2)}` : '-';
                    html += `<tr>
                        <td>${new Date(t.timestamp).toLocaleDateString()}</td>
                        <td>${t.symbol}</td>
                        <td class="${sideClass}">${t.side.toUpperCase()}</td>
                        <td>${t.quantity}</td>
                        <td>$${t.price.toLocaleString()}</td>
                        <td>${pnl}</td>
                    </tr>`;
                }
                html += '</tbody></table></div>';
            }

            div.innerHTML = html;

            // Render equity curve with Chart.js
            if (r.equity_curve && r.equity_curve.length > 0) {
                const ctx = document.getElementById('backtest-equity-chart');
                if (ctx) {
                    const canvas = document.createElement('canvas');
                    canvas.style.width = '100%';
                    canvas.style.height = '100%';
                    ctx.appendChild(canvas);
                    new Chart(canvas.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: r.equity_curve.map(e => e.date.split(' ')[0]),
                            datasets: [{
                                label: 'Portfolio Value',
                                data: r.equity_curve.map(e => e.value),
                                borderColor: r.total_return >= 0 ? '#28a745' : '#dc3545',
                                backgroundColor: r.total_return >= 0 ? 'rgba(40,167,69,0.1)' : 'rgba(220,53,69,0.1)',
                                borderWidth: 2, fill: true, tension: 0.1, pointRadius: 0
                            }]
                        },
                        options: {
                            responsive: true, maintainAspectRatio: false,
                            plugins: { legend: { display: false }, title: { display: true, text: 'Equity Curve' } },
                            scales: {
                                x: { ticks: { maxTicksLimit: 12 } },
                                y: { title: { display: true, text: 'Value ($)' } }
                            }
                        }
                    });
                }
            }

            this.showNotification('Backtest complete!', 'success');
        } catch (error) {
            console.error('Backtest error:', error);
            if (div) div.innerHTML = `<div class="alert alert-danger">Backtest failed: ${error.message}</div>`;
            this.showNotification('Backtest failed', 'danger');
        }
    }

    // --- Sentiment Analysis ---
    async analyzeSentiment() {
        const text = document.getElementById('sentiment-text')?.value;
        if (!text?.trim()) {
            this.showNotification('Please enter text to analyze', 'warning');
            return;
        }

        try {
            const res = await fetch(`${this.apiBaseUrl}/ai/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, analysis_type: 'sentiment' })
            });

            const result = await res.json();
            const r = result.results;
            const div = document.getElementById('sentiment-results');
            if (div && r) {
                const sentimentClass = r.sentiment === 'positive' ? 'success' :
                                       r.sentiment === 'negative' ? 'danger' : 'warning';
                div.innerHTML = `
                    <div class="alert alert-${sentimentClass}">
                        <h6>Analysis Results</h6>
                        <div class="metric"><label>Sentiment</label><span>${(r.sentiment || 'N/A').toUpperCase()}</span></div>
                        <div class="metric"><label>Confidence</label><span>${((r.confidence || 0) * 100).toFixed(1)}%</span></div>
                        <div class="metric"><label>Keywords</label><span>${(r.keywords || []).join(', ') || 'N/A'}</span></div>
                    </div>`;
            }
        } catch (error) {
            console.error('Sentiment error:', error);
            this.showNotification('Sentiment analysis failed', 'danger');
        }
    }

    // --- Utilities ---
    setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    setHTML(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    }

    formatTime(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    showNotification(message, type = 'info') {
        const div = document.createElement('div');
        div.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        div.style.cssText = 'top: 70px; right: 20px; z-index: 9999; min-width: 300px; max-width: 400px;';
        div.innerHTML = `${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
        document.body.appendChild(div);
        setTimeout(() => div.remove(), 4000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});
