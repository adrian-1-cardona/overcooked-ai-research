# Overcooked-AI Research Workspace 🧑‍🍳🤖

This repository contains Adrian Cardona's senior project research on cooperative AI agents in Overcooked-AI, completed under the guidance of Professor Rodrigo Canaan.

The research asks: *How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?*

---

## 🌟 Star Feature: Visual Gameplay Renderer & Interactive Player

Bring Overcooked-AI to life! This research workspace features a high-performance gameplay visualizer, interactive human player, video exporter, and telemetry replayer built directly on top of Overcooked-AI's sprite rendering engine.

Whether you want to **watch AI agents coordinate live in a desktop window**, **play as Chef 0 with your keyboard**, **export MP4 gameplay videos headlessly**, or **visually replay recorded experiment trajectories**, you can do it all with a single unified command:

```bash
# ⚡ Quickest One-Line Launch (auto-uses .venv):
./render.sh

# Or using the root launcher:
python render.py

# 1. Watch AI agents play live in an interactive Pygame desktop window
python render.py

# 2. Jump in and play as Chef 0 using WASD / Arrow keys alongside an AI partner!
python render.py --agent-0 human

# 3. Export episode gameplay to an MP4 video (works headlessly too)
python render.py --mode video --horizon 200 --output overcooked-agent-eval/results/gameplay.mp4

# 4. Visually replay a previously recorded experiment CSV
python render.py --replay-csv overcooked-agent-eval/results/random_baseline_cramped_room.csv

# 5. Explore different kitchen layouts at custom playback speeds
python render.py --layout asymmetric_advantages --fps 15
```

> **Note:** You can also run via the full path `python overcooked-agent-eval/experiments/render_gameplay.py [options]`.

### 🎮 Interactive Window Controls

When running in window mode (`--mode window` or default when a display is connected):

| Key | Action |
| --- | --- |
| `SPACE` / `P` | **Pause / Resume** simulation |
| `RIGHT ARROW` | **Step 1 frame forward** (when paused) |
| `UP` / `DOWN` | **Increase / Decrease FPS** playback speed |
| `R` | **Restart episode** from beginning |
| `ESC` / `Q` | **Exit window** cleanly |
| **Human Controls (`--agent-0 human`)** | `WASD` or `Arrow Keys` to move; `SPACE` / `ENTER` / `F` to interact (pick up, drop, chop, cook) |

### 🚀 Key Capabilities

- 🖥️ **Live Pygame Desktop Player**: Real-time rendering of chefs, onions, tomatoes, cooking pots, soup progress bars, recipes, timers, and scores.
- 🧑‍🍳 **Human-in-the-Loop Evaluation**: Jump into the kitchen as Chef 0 to test coordination fluidity with AI agents in real time.
- 🎥 **Headless-Safe Video Export**: Encodes high-fidelity `.mp4` video with OpenCV (`cv2.VideoWriter`), automatically padding dimensions to ensure 100% compatibility with QuickTime, web browsers, and media players.
- 🔁 **Telemetry CSV Replay**: Re-simulates and visually replays recorded single-episode or multi-episode experiment CSVs with automatic layout detection.
- 🗺️ **Full Layout Support**: Discovers and validates across all 49 built-in Overcooked-AI kitchen layouts (e.g. `cramped_room`, `asymmetric_advantages`, `coordination_ring`, `forced_coordination`, `counter_circuit`).

### ⚙️ Command-Line Options (`render_gameplay.py`)

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--layout` | `str` | `cramped_room` | Overcooked kitchen layout name |
| `--mode` | `str` | `auto` | `window` (live GUI), `video` (MP4), `frames` (PNGs), or `auto` |
| `--horizon` | `int` | `400` | Maximum timesteps per episode |
| `--fps` | `int` | `10` | Playback speed and video framerate |
| `--agent-0` | `str` | `random` | Policy for Agent 0: `random`, `stay`, or `human` |
| `--agent-1` | `str` | `random` | Policy for Agent 1: `random` or `stay` |
| `--output` | `Path` | `results/gameplay.mp4` | File path for exported video or directory for frames |
| `--replay-csv` | `Path` | `None` | Path to recorded CSV telemetry file to replay |
| `--episode` | `int` | `1` | Episode number to replay when CSV contains multiple runs |
| `--seed` | `int` | `42` | Random seed for reproducible agent sampling |
| `--tile-size` | `int` | `75` | Pixel scale per grid cell |

---

## Requirements

- Git
- Python 3.10 (`>=3.10,<3.11`)
- The `external/overcooked_ai` Git submodule

## Setup

For a new clone:

```bash
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git
cd overcooked-ai-research
```

For an existing clone, initialize the submodule from the project root:

```bash
git submodule update --init --recursive
```

Create the virtual environment and install dependencies:

```bash
python3.10 -m venv overcooked-agent-eval/.venv
source overcooked-agent-eval/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./external/overcooked_ai
```

When opening a new terminal, reactivate the environment from the project root:

```bash
source overcooked-agent-eval/.venv/bin/activate
```

---

## Baseline Experiments & Telemetry

### Run the Random Baseline

To run a quantitative baseline experiment without graphical rendering:

```bash
python overcooked-agent-eval/experiments/run_random_baseline.py
```

The default run:
- Uses the `cramped_room` layout.
- Runs one episode for 400 timesteps using random seed 42.
- Evaluates two random agents sampling from all actions (`north`, `south`, `east`, `west`, `stay`, `interact`).
- Saves per-timestep telemetry to `overcooked-agent-eval/results/random_baseline_cramped_room.csv`.

### Terminal Output

```text
Random baseline complete
Layout: cramped_room
Episode length: 400
Total reward / score: 0.0
Timesteps logged: 400
Output CSV: <path>/overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

### Telemetry CSV Structure

The generated CSV records 11 fields per timestep:
`episode`, `timestep`, `agent_0_action`, `agent_1_action`, `sparse_reward`, `agent_0_shaped_reward`, `agent_1_shaped_reward`, `cumulative_sparse_reward`, `agent_0_position`, `agent_1_position`, and `done`.

### Summarize Results

The result summarizer reads any generated CSV and prints a diagnostic report:

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py \
  overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

---

## Repository Structure

```text
overcooked-ai-research/
├── README.md                      # Main research overview & gameplay renderer guide
├── render.py                      # ⚡ One-line root gameplay renderer launcher
├── render.sh                      # ⚡ Executable one-line shell launcher
├── external/                      # Git submodules
│   └── overcooked_ai/             # Upstream Overcooked-AI benchmark (read-only)
├── overcooked-agent-eval/         # Evaluation framework
│   ├── experiments/
│   │   ├── render_gameplay.py     # 🌟 Core gameplay visualizer, player & exporter
│   │   ├── run_random_baseline.py # Reproducible baseline experiment runner
│   │   └── summarize_episode_metrics.py # Result summarizer
│   ├── tests/
│   │   └── test_render_gameplay.py # Automated test suite (8 tests)
│   ├── results/                   # Generated telemetry CSVs and MP4 videos
│   ├── agents/                    # Future custom cooperative agents
│   └── metrics/                   # Future partner compatibility metrics
├── project_docs/                  # Project proposal and research notes
└── work_done/                     # Milestone write-ups and documentation
```
