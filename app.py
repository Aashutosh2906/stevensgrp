"""
DVNC.AI — HuggingFace Spaces Entry Point

This app.py lives at the repo root and handles:
  1. Detecting and fixing misplaced upgrade folders (e.g. 'stevens_upgrade (1)')
  2. Building the connectome database on first run
  3. Launching the Gradio app on port 7860

Drop this file at: DVNC.AI2/app.py  (replaces the existing one)
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

# ── Locate the project root ─────────────────────────────────────────────────

# On HuggingFace Spaces, app.py sits at /home/user/app/app.py
# The v3 codebase is at /home/user/app/dvnc_connectome_v3/
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR / "dvnc_connectome_v3"

if not PROJECT_DIR.exists():
    # Fallback: maybe app.py is inside dvnc_connectome_v3 itself
    if (APP_DIR / "src" / "dvnc_connectome").exists():
        PROJECT_DIR = APP_DIR
    else:
        print("[app.py] ERROR: Cannot find dvnc_connectome_v3/ directory.")
        print(f"[app.py] Searched in: {APP_DIR}")
        sys.exit(1)

print(f"[app.py] Project directory: {PROJECT_DIR}")


# ── Step 1: Fix misplaced Stevens upgrade files ─────────────────────────────

def fix_stevens_upgrade():
    """
    If the 'stevens_upgrade (1)' folder exists as a sibling inside PROJECT_DIR,
    move its contents to the correct locations and remove the leftover folder.
    """
    # Try common variations of the folder name
    candidates = [
        PROJECT_DIR / "stevens_upgrade (1)",
        PROJECT_DIR / "stevens_upgrade",
        PROJECT_DIR / "stevens_upgrade(1)",
    ]
    upgrade_dir = None
    for c in candidates:
        if c.is_dir():
            upgrade_dir = c
            break

    if upgrade_dir is None:
        return  # Nothing to fix

    print(f"[app.py] Found misplaced upgrade folder: {upgrade_dir.name}")
    print("[app.py] Moving files to correct locations...")

    # Walk through the upgrade folder and copy each file to the matching
    # location in the project tree
    count = 0
    for src_file in upgrade_dir.rglob("*"):
        if src_file.is_file():
            # Compute the relative path inside the upgrade folder
            rel = src_file.relative_to(upgrade_dir)
            dst = PROJECT_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)
            action = "NEW" if not dst.exists() else "REPLACED"
            print(f"  ✓ {rel}  ({action})")
            count += 1

    # Remove the upgrade folder
    shutil.rmtree(upgrade_dir)
    print(f"[app.py] Moved {count} files. Removed '{upgrade_dir.name}'.")


fix_stevens_upgrade()


# ── Step 2: Add src/ to Python path ─────────────────────────────────────────

SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Also ensure PROJECT_DIR is the working directory so relative paths in
# scripts (e.g. ./data/dvnc.db) resolve correctly
os.chdir(PROJECT_DIR)
print(f"[app.py] Working directory: {os.getcwd()}")
print(f"[app.py] Python path includes: {SRC_DIR}")


# ── Step 3: Build database if it doesn't exist ──────────────────────────────

DB_PATH = PROJECT_DIR / "data" / "dvnc.db"

# Force rebuild if DB is from a previous version (no text snippets in docs)
_REBUILD_MARKER = PROJECT_DIR / "data" / ".v3_with_snippets"
if DB_PATH.exists() and not _REBUILD_MARKER.exists():
    print("[app.py] Rebuilding database with text snippets for evidence retrieval...")
    DB_PATH.unlink()

if not DB_PATH.exists():
    print("[app.py] Database not found — building connectome...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    from dvnc_connectome.db.neurographdb import NeuroGraphDB
    from dvnc_connectome.curation.pipeline import run_full_pipeline

    t0 = time.time()
    db = NeuroGraphDB(str(DB_PATH))
    db.init()
    stats = run_full_pipeline(
        db=db,
        include_network=True,
        include_stevens=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[app.py] Database built in {elapsed:.1f}s")
    print(f"[app.py] Stats: {json.dumps(stats, indent=2)}")
    _REBUILD_MARKER.touch()
else:
    print(f"[app.py] Database found: {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)")


# ── Step 4: Launch Gradio ────────────────────────────────────────────────────

from dvnc_connectome.apps.gradio_app import make_app

print("[app.py] Starting DVNC.AI Gradio app...")
app = make_app(str(DB_PATH))
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
)
