# DVNC.AI Connectome v3

**Brain-inspired polymathic design discovery system.**

Built for The Da Vinci Network. Inspired by Jeff Lichtman's connectome research,
MiroFish's multi-agent propagation, and Leonardo da Vinci's polymathic methodology.

---

## What's inside

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                DVNC CONNECTOME (NeuroGraphDB)               │
│  Nodes: Concepts · Documents · Domains                      │
│  Synapses: CO_OCCURS · MENTIONS · EVOKES · PRIMES           │
│  Plasticity: Hebbian reinforcement on every upsert          │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  DA VINCI ROUTING SYSTEM   │
         │  - Spreading activation    │
         │  - Hub suppression         │
         │  - EVOKES cross-domain     │
         │  - Full visible trace      │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   6-AGENT DEBATE SYSTEM    │
         │                            │
         │  1. Problem Framer         │
         │  2. Evidence Judge         │
         │  3. Hypothesis Composer ←Claude Opus
         │  4. Adversarial Reviewer   │
         │  5. Provenance Checker     │
         │  6. Orchestrator       ←Claude Opus
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  GRADIO APP (3 tabs)       │
         │  - Discovery Engine        │
         │  - Connectome Explorer     │
         │  - Database Inspector      │
         └────────────────────────────┘
```

### Data Sources

The curation pipeline ingests from:
- **Wikipedia** (30 industrial design topics — biomechanics, ergonomics, materials, etc.)
- **OpenAlex** (open-access papers across 6 domain queries)
- **Seed corpus** (10 expert-written industrial design knowledge statements)
- **Your own files** (codex JSON, masterclass JSON, triplets CSV)

---

## Quickstart

### Option A: One command (Mac/Linux)

```bash
bash run_all.sh
```

Then open: http://127.0.0.1:7860

### Option B: One command (Windows)

```bat
run_all.bat
```

### Option C: Manual steps

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install
pip install -e .

# 3. Build the database (fetches Wikipedia + OpenAlex open datasets)
python scripts/build_db.py --db ./data/dvnc.db

# With your own files:
python scripts/build_db.py \
  --db ./data/dvnc.db \
  --codex ./codex_data.json \
  --masterclass ./Masterclass_data.json \
  --triplets ./dvnc_kg_triplets.csv

# Offline only (no network):
python scripts/build_db.py --db ./data/dvnc.db --no-network

# 4. Launch
python scripts/run_gradio.py --db ./data/dvnc.db

# Public share link (for demos):
python scripts/run_gradio.py --db ./data/dvnc.db --share
```

---

## Claude Opus Integration

To enable full AI synthesis (Agents 3 and 6):

1. Get an API key from https://console.anthropic.com
2. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY=sk-ant-...`
3. Or paste it directly into the Gradio app's API key field

**Without an API key:** The system still runs — routing, evidence gathering, and
provenance checking all work. Only the Claude synthesis steps fall back to templates.

---

## HuggingFace Spaces Deployment

1. Create a new Space: https://huggingface.co/new-space
2. Choose **Gradio** SDK
3. Upload all files from this folder
4. Add your `ANTHROPIC_API_KEY` as a Space Secret
5. Set the entry point in `app.py`:

```python
# app.py (create this in the Space root)
import subprocess, sys
subprocess.run([sys.executable, "scripts/build_db.py", "--db", "./data/dvnc.db", "--no-network"])

from dvnc_connectome.apps.gradio_app import make_app
app = make_app("./data/dvnc.db")
app.launch()
```

**Free tier is sufficient.** HuggingFace Pro (£9/month) gives you faster hardware
(T4 GPU, 16GB RAM) which improves responsiveness but is not required.

---

## The Da Vinci Routing System

The routing panel shows every decision explicitly:

- **PRIMARY ROUTE**: The best activation path through the connectome
- **ALTERNATIVE ROUTES**: 3 diverse non-overlapping paths
- **SUPPRESSED HUBS**: Overly-connected nodes deliberately bypassed
- **LMM ACTIVATIONS**: Which of Leonardo's 10 mental models fired
- **SPREADING ACTIVATION MAP**: Full activation landscape with scores
- **NOVELTY SCORE**: How non-obvious is this route? (0–1)
- **CROSS-DOMAIN LEAPS**: Count of domain bridges crossed

---

## Hebbian Plasticity (Self-Growing Graph)

Every time an Innovation Card scores above 0.75 overall:
- The concept pairs in that route are automatically reinforced
- Synapse weights increase proportionally to the score
- The graph gets smarter with every use

No other AI system does this at the architecture level.

---

## LMM (Leonardo Mental Models)

| Code | Name | Description |
|------|------|-------------|
| LMM_001 | Geometric Structure | Underlying structural patterns |
| LMM_002 | Dynamic Flow | Motion, change, and flux |
| LMM_003 | Analogical Bridging | Cross-domain structural analogies |
| LMM_004 | Systems Thinking | Holistic interconnection |
| LMM_005 | Sensory Detail | Precise observation |
| LMM_006 | Spatial Reasoning | 3D and geometric reasoning |
| LMM_007 | Living Machine | Biological as mechanism |
| LMM_008 | Sfumato | Productive ambiguity |
| LMM_009 | Art/Science Fusion | Beauty meets function |
| LMM_010 | Corporalita | Body-mind integration |
