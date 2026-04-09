#!/usr/bin/env bash
# DVNC.AI — One-command startup script
# Usage: bash run_all.sh [--no-network] [--codex path] [--masterclass path] [--share]

set -e

DB_PATH="./data/dvnc.db"
EXTRA_ARGS=""
SHARE_FLAG=""
NO_NETWORK=""
CODEX=""
MASTERCLASS=""
TRIPLETS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-network) NO_NETWORK="--no-network" ;;
        --share) SHARE_FLAG="--share" ;;
        --codex) CODEX="--codex $2"; shift ;;
        --masterclass) MASTERCLASS="--masterclass $2"; shift ;;
        --triplets) TRIPLETS="--triplets $2"; shift ;;
    esac
    shift
done

echo "═══════════════════════════════════════════════════════"
echo "  DVNC.AI Connectome v3 — Starting Up"
echo "═══════════════════════════════════════════════════════"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "[startup] Creating virtual environment..."
    python3 -m venv .venv
fi

echo "[startup] Activating virtual environment..."
source .venv/bin/activate

echo "[startup] Installing dependencies..."
pip install -e . -q

echo "[startup] Building Connectome database..."
python scripts/build_db.py \
    --db "$DB_PATH" \
    $NO_NETWORK \
    $CODEX \
    $MASTERCLASS \
    $TRIPLETS

echo "[startup] Launching Gradio app..."
echo "[startup] Open http://127.0.0.1:7860 in your browser"
python scripts/run_gradio.py --db "$DB_PATH" $SHARE_FLAG
