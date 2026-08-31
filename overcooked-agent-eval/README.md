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
