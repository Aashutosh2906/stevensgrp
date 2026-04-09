@echo off
REM DVNC.AI — One-command startup script (Windows)

echo ═══════════════════════════════════════════════════════
echo   DVNC.AI Connectome v3 — Starting Up (Windows)
echo ═══════════════════════════════════════════════════════

if not exist .venv (
    echo [startup] Creating virtual environment...
    python -m venv .venv
)

echo [startup] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [startup] Installing dependencies...
pip install -e . -q

echo [startup] Building Connectome database...
python scripts\build_db.py --db .\data\dvnc.db

echo [startup] Launching Gradio app...
echo [startup] Opening http://127.0.0.1:7860 in your browser...
start http://127.0.0.1:7860
python scripts\run_gradio.py --db .\data\dvnc.db
