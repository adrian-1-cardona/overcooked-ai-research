# Overcooked Agent Evaluation Framework

This folder contains the custom evaluation and experimentation framework for cooperative AI agents in Overcooked-AI. The upstream Overcooked-AI environment is in `../external/overcooked_ai` as a Git submodule and remains completely read-only.

---

## Setup

Overcooked-AI requires **Python 3.10**.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!IMPORTANT]
> Make sure `.venv` is activated before executing any experiment or test commands.

---

## Step-by-Step Guide: How to Run Each Milestone

### 1. Milestone 1 — Single-Episode Random Baseline

Runs two built-in random agents on `cramped_room` for 400 steps:

```bash
python experiments/run_random_baseline.py
```
- **Output:** `results/random_baseline_cramped_room.csv`

---

### 2. Milestone 2 — Multi-Episode & Multi-Layout Experiments

Runs repeatable seeded batches with structured telemetry logging across single or multiple kitchen layouts:

```bash
# Default run: 5 episodes on cramped_room
python experiments/run_multi_episode_baseline.py

# Multi-layout run: 10 episodes across cramped_room, asymmetric_advantages, and coordination_ring
python experiments/run_multi_episode_baseline.py --layouts cramped_room asymmetric_advantages coordination_ring --episodes 10
```

CLI options:
- `--layouts` / `--layout`: One or more layout names.
- `--episodes`: Number of episodes per layout (default: 5).
- `--horizon`: Timesteps per episode (default: 400).
- `--base-seed` / `--seed`: Base seed for episode 1 (default: 42). Episode $k$ uses seed $\text{base\_seed} + k - 1$.
- `--output`: Optional custom output path.

Outputs generated per layout:
- Telemetry CSV: `results/multi_episode_random_baseline_<layout>.csv`
- Reproducibility Manifest: `results/multi_episode_random_baseline_<layout>.manifest.json`

---

### 3. Milestone 2 — Performance & Coordination Summarizer

`summarize_episode_metrics.py` reads telemetry CSVs, computes 68-field per-episode performance and coordination metrics, and prints batch aggregate statistics:

```bash
# Summarise all CSVs in results/
python experiments/summarize_episode_metrics.py

# Process a specific multi-episode telemetry CSV
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room.csv

# Inspect an existing episode metrics CSV
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room_episode_metrics.csv
```

Outputs:
- Per-episode metrics CSV: `results/*_episode_metrics.csv`
- Detailed formatted terminal tables reporting Sample Size ($N$), Mean, Std, Min, and Max across:
  - **Performance:** Team score, soups delivered, episode length.
  - **Movement & Space:** Distance traveled, explicit idle time and rates.
  - **Conflict & Collisions:** Wall collisions, teammate blocking events, same-target collisions, swap collisions, interference timesteps/rates, repeated interference.
  - **Role Proxies:** Held-object time-shares (`none`, `onion`, `tomato`, `dish`, `soup`), potting events, dish/soup pickups, delivery counts/rates.
  - **Task Duplication & Pipeline:** Redundant gather/dish/deliver timesteps, team task duplication rate, unused pipeline timesteps/rates.

---

### 4. Running Automated Tests

Execute the full test suite with:
```bash
python -m unittest discover -s tests
```

---

## File Formats & Documentation

- **[`telemetry/README.md`](telemetry/README.md):** 27-field per-timestep telemetry schema and JSON reproducibility manifest specification.
- **[`metrics/README.md`](metrics/README.md):** Formal methods-grade mathematical specifications for every coordination, performance, role proxy, and duplication metric.

---

## Folder Structure

- `agents/` — Custom cooperative agent implementations.
- `experiments/` — Repeatable experiment execution scripts and summary tools.
- `metrics/` — Metric computation, aggregation, and formal specifications.
- `telemetry/` — Validated telemetry logging and reproducibility manifests.
- `results/` — Generated experiment outputs (ignored by git).
- `tests/` — Automated test suite.
