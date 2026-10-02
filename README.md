# Overcooked-AI Research Workspace

This repository contains Adrian Cardona's senior project research on cooperative AI agents in Overcooked-AI, completed under the guidance of Professor Rodrigo Canaan.

---

## 🚀 Quick Start & Environment Setup

### Requirements
- Git
- Python 3.10 (`>=3.10,<3.11`)

### 1. Clone & Initialize Submodules
```bash
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git
cd overcooked-ai-research
```

For an existing clone:
```bash
git submodule update --init --recursive
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3.10 -m venv overcooked-agent-eval/.venv
source overcooked-agent-eval/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./external/overcooked_ai
pip install "protobuf==3.19.6" "grpcio==1.51.3" "streamlit==1.19.0" "altair==4.2.2" seaborn matplotlib pandas
```

---

## 🎮 Play & Render Gameplay

### 1. Interactive Play (Human vs. Greedy Agent)
Control **Chef 0** with WASD / Arrow keys while **Chef 1** is controlled by the BFS Greedy Search agent:

```bash
python render.py --agent-0 human --agent-1 greedy --layout cramped_room
```

**Controls:**
- `[WASD / Arrow Keys]` : Move North, South, West, East
- `[SPACE / ENTER / F]` : Interact (pick up items, drop, chop, cook)
- `[P]`                 : Pause / resume simulation
- `[R]`                 : Restart episode
- `[ESC / Q]`           : Quit window

---

## 🏰 Available Kitchen Layouts / Rooms to Experiment With

You can pass any of the following kitchen room layouts to the `--layout` parameter:

| Layout Name | Description & Experiment Focus |
| :--- | :--- |
| `cramped_room` | **Default:** Compact 5x5 layout where agents easily collide. |
| `asymmetric_advantages` | Asymmetric layout where one agent has easier access to onions/pots. |
| `coordination_ring` | Ring layout requiring agents to coordinate and yield space. |
| `forced_coordination` | Kitchen divided by a central counter requiring item transfers. |
| `counter_circuit` | Long perimeter kitchen requiring navigation around counters. |
| `bottleneck` | Tight single-tile choke points where collision handling is critical. |
| `corridor` | Long narrow hallway testing passing & movement coordination. |
| `large_room` | Wide open kitchen testing long-distance path planning. |

#### Example Commands across Different Rooms:
```bash
# Experiment on Asymmetric Advantages
python render.py --agent-0 human --agent-1 greedy --layout asymmetric_advantages

# Experiment on Coordination Ring
python render.py --agent-0 human --agent-1 greedy --layout coordination_ring

# Experiment on Forced Coordination
python render.py --agent-0 human --agent-1 greedy --layout forced_coordination

# Experiment on Counter Circuit
python render.py --agent-0 human --agent-1 greedy --layout counter_circuit
```

---

## 🎥 Exporting Video & Headless Replays

Export a 200-step gameplay MP4 video without opening a GUI window:
```bash
mkdir -p overcooked-agent-eval/results
python render.py --agent-0 human --agent-1 greedy --layout cramped_room --mode video --horizon 200 --output overcooked-agent-eval/results/last_render_run.csv
```

---

## 📊 Streamlit Telemetry Dashboard

Launch the interactive web dashboard to analyze performance metrics, reward accumulation curves, and agent action distributions:

```bash
streamlit run dashboard/run_telemetry.py
```

*Automatically loads telemetry from `overcooked-agent-eval/results/last_render_run.csv`.*

---

## 📁 Repository Structure

- `agents/` — Custom agent implementations (e.g. `greedy_symbol_search.py` BFS rule-based agent).
- `dashboard/` — Streamlit dashboard scripts (`run_telemetry.py` and `coordination_matrix.py`).
- `overcooked-agent-eval/` — Evaluation scripts, telemetry outputs, and baseline runner.
- `project_docs/` — Proposal and research notes.
- `work_done/` — Milestone reports and reproduction details.
- `external/overcooked_ai/` — Upstream Overcooked-AI submodule.
