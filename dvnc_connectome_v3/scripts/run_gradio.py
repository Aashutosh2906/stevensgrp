#!/usr/bin/env python3
"""
Launch the DVNC.AI Gradio app.

Usage:
    python scripts/run_gradio.py --db ./data/dvnc.db [--host 0.0.0.0] [--port 7860] [--share]

Options:
    --db     Path to SQLite database (must exist — run build_db.py first)
    --host   Host to bind to (default: 127.0.0.1)
    --port   Port to bind to (default: 7860)
    --share  Create a public Gradio share link
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvnc_connectome.apps.gradio_app import make_app


def main():
    parser = argparse.ArgumentParser(description="Run DVNC.AI Gradio app")
    parser.add_argument("--db", default="./data/dvnc.db", help="Database path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[run_gradio] Database not found: {db_path}")
        print("[run_gradio] Run 'python scripts/build_db.py' first to build the database.")
        sys.exit(1)

    print(f"[run_gradio] Loading DVNC.AI from: {db_path}")
    app = make_app(str(db_path))

    print(f"[run_gradio] Launching at http://{args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
