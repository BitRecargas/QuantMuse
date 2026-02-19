#!/bin/bash
# QuantMuse Bellas Server Setup Script
# Run: bash setup_bellas.sh

set -e

echo "========================================="
echo "  QuantMuse Bellas Server Setup"
echo "========================================="

# 1. Clone or update repo
echo ""
echo "[1/6] Setting up repository..."
if [ -d "$HOME/QuantMuse" ]; then
    echo "  Repo exists, pulling latest..."
    cd "$HOME/QuantMuse"
    git pull
else
    echo "  Cloning repo..."
    cd "$HOME"
    git clone https://github.com/BitRecargas/QuantMuse.git
    cd "$HOME/QuantMuse"
fi

# 2. Create .env if missing
echo ""
echo "[2/6] Setting up config..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from template"
else
    echo "  .env already exists, skipping"
fi
mkdir -p data/sweeps

# 3. Python venv + sweep deps
echo ""
echo "[3/6] Setting up Python venv for sweeps..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created venv"
else
    echo "  venv exists"
fi
source venv/bin/activate
pip install --quiet numba pandas numpy requests matplotlib
echo "  Sweep dependencies installed"

# 4. Quick test
echo ""
echo "[4/6] Testing sweep engine (quick 500-candle test)..."
python -m data_service.backtest.strategy_sweep \
    --symbol ETHUSDT --interval 1h --limit 500 --workers 2 --top 3
echo "  Sweep engine works!"

# 5. Build full Docker stack
echo ""
echo "[5/6] Building Docker full stack (this takes a few minutes)..."
docker compose -f docker-compose.bellas.yml up --build -d
echo "  Docker stack running!"
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):9501"
echo "  API:       http://$(hostname -I | awk '{print $1}'):9000"

# 6. Print sweep commands
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "========================================="
echo "  To run mega sweeps (2 at a time):"
echo "========================================="
echo ""
echo "source ~/QuantMuse/venv/bin/activate && cd ~/QuantMuse"
echo ""
echo "# Batch 1 (run these first):"
echo "nohup python -m data_service.backtest.strategy_sweep --interval 1h  --limit 15000 --mega --walkforward --workers 8 > /tmp/sweep_1h.log 2>&1 &"
echo "nohup python -m data_service.backtest.strategy_sweep --interval 15m --limit 35000 --mega --walkforward --workers 8 > /tmp/sweep_15m.log 2>&1 &"
echo ""
echo "# Batch 2 (after batch 1 finishes):"
echo "nohup python -m data_service.backtest.strategy_sweep --interval 5m  --limit 50000 --mega --walkforward --workers 8 > /tmp/sweep_5m.log 2>&1 &"
echo "nohup python -m data_service.backtest.strategy_sweep --interval 1m  --limit 50000 --mega --walkforward --workers 8 > /tmp/sweep_1m.log 2>&1 &"
echo ""
echo "# Check progress:"
echo "tail -5 /tmp/sweep_1h.log"
echo "tail -5 /tmp/sweep_15m.log"
echo ""
echo "========================================="
