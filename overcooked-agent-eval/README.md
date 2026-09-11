# Overcooked Agent Evaluation Framework

This folder contains my custom framework for evaluating cooperative AI agents. The upstream Overcooked-AI environment lives in `../external/overcooked_ai` as a Git submodule and should be treated as read-only. Custom agents, experiments, telemetry, metrics, tests, and analysis should be added here.

## Setup

Overcooked-AI currently requires Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run `git submodule update --init --recursive` from the repository root first if `../external/overcooked_ai` is empty.

## Experiments

### Gameplay rendering

Render Overcooked-AI gameplay live in an interactive Pygame desktop window, play interactively as Chef 0, export to an MP4 video, or replay existing CSV telemetry.

```bash
# 1. Live Pygame desktop window (default when GUI is available)
python experiments/render_gameplay.py

# 2. Interactive human mode: Play as Chef 0 with keyboard against an AI partner
python experiments/render_gameplay.py --agent-0 human

# 3. Export episode to an MP4 video (headless-safe)
python experiments/render_gameplay.py --mode video --horizon 200 --output results/gameplay.mp4

# 4. Replay a previously recorded CSV telemetry run
python experiments/render_gameplay.py --replay-csv results/random_baseline_cramped_room.csv

# 5. Render on different kitchen layouts at custom FPS
python experiments/render_gameplay.py --layout asymmetric_advantages --fps 15

# 6. Automatically generate and display Matplotlib performance graphs when done
python experiments/render_gameplay.py --plot
python experiments/render_gameplay.py --agent-0 human --plot

# 7. Plot dashboard from any existing telemetry CSV
python experiments/plot_run.py results/random_baseline_cramped_room.csv
```

**Window Controls:**
- `SPACE` / `P`: Pause / Resume simulation
- `RIGHT ARROW`: Step 1 timestep forward (when paused)
- `UP` / `DOWN`: Increase / Decrease FPS playback speed
- `R`: Restart episode
- `ESC` / `Q`: Exit window cleanly
- **Human Player Controls (`--agent-0 human`):** `WASD` / `Arrow Keys` to move; `SPACE` / `ENTER` / `F` to interact (pick up, drop, chop, cook).

### Random baseline

Runs two built-in random agents on `cramped_room`. Both agents sample from every available action, including `interact`, using a fixed seed so the run can be repeated.

```bash
python experiments/run_random_baseline.py
```

The script prints a short summary and creates `results/random_baseline_cramped_room.csv`. The CSV contains the episode number, timestep, each agent's action, timestep sparse reward, per-agent shaped rewards, cumulative sparse reward, and both player positions.

Optional settings are available with `python experiments/run_random_baseline.py --help`.


### Summarising results

`summarize_episode_metrics.py` reads any CSV produced by this framework and prints a formatted summary to stdout. It auto-detects the file format.

```bash
# Summarise all CSVs in results/
python experiments/summarize_episode_metrics.py

# Summarise a specific file
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room_episode_metrics.csv
```

Three CSV formats are supported:

| Format | Typical filename pattern | Contents |
|---|---|---|
| Single-episode timestep telemetry | `random_baseline_*.csv` | One row per timestep; score, actions, positions |
| Multi-episode timestep telemetry | `multi_episode_*.csv` | One row per timestep across multiple episodes; richer agent state |
| Episode-level metrics | `*_episode_metrics.csv` | One row per episode; score, movement, idle, and collision stats |

For episode-level metrics files the summary includes mean, std, min, and max across all episodes.

Generated CSV files are ignored by Git, so running experiments does not add result data to a commit by default.

## Folder structure

- `agents/` - custom cooperative agent strategies
- `experiments/` - repeatable experiment runners
- `metrics/` - coordination and performance metrics
- `results/` - generated experiment output
- `notebooks/` - exploratory analysis
- `dashboard/` - future visualization tools
- `tests/` - automated checks

## Future milestones

Future work will add meaningful baseline strategies, coordination metrics, experiments across layouts and pairings, and partner compatibility analysis. The framework may later support reinforcement learning, evolutionary methods, or quality-diversity approaches, but the immediate goal is a clean and reliable evaluation foundation.
