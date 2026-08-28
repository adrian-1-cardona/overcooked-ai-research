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

## Random baseline experiment

The first milestone runs two built-in random agents on `cramped_room`. Both agents sample from every available action, including `interact`, using a fixed seed so the run can be repeated.

```bash
python experiments/run_random_baseline.py
```

The script prints a short summary and creates `results/random_baseline_cramped_room.csv`.

The CSV contains the episode number, timestep, each agent’s action, timestep sparse reward, per-agent shaped rewards, cumulative sparse reward, and both player positions. Generated CSV files are ignored by Git, so running the experiment does not add result data to a commit by default.

Optional settings are available with `python experiments/run_random_baseline.py --help`.

## Multi-episode baseline and telemetry

Milestone 2A makes the baseline more useful. Instead of just proving the environment runs once, this adds a reusable way to log structured telemetry across multiple episodes.

```bash
python experiments/run_multi_episode_baseline.py
```

By default, two random agents play five 400-timestep episodes on `cramped_room`. The terminal summary reports the episode count, timesteps logged, average episode length, average reward, and output path. The run creates `results/multi_episode_random_baseline_cramped_room.csv`.

Each CSV row records the deterministic run ID, episode ID and seed, timestep, layout, player IDs, agent names, actions, sparse and shaped rewards, completion status, positions, orientations, held objects, and Overcooked events. The runner validates the complete CSV after saving it. The full field and missing-value definitions are in [`telemetry/README.md`](telemetry/README.md).

The base seed defaults to `42`, and episode seeds increase from that value (`42`, `43`, and so on). Repeating the same configuration produces the same run ID, actions, states, rewards, and CSV contents. A different layout automatically gets a different output filename. Use `--base-seed` to choose the first episode seed; `--seed` remains available as a shorter alias. Optional settings are available with `python experiments/run_multi_episode_baseline.py --help`.

For example, this acceptance run writes 10 deterministic episodes:

```bash
python experiments/run_multi_episode_baseline.py --episodes 10
```

## Folder structure

- `agents/` — custom cooperative agent strategies
- `experiments/` — repeatable experiment runners
- `metrics/` — coordination and performance metrics
- `results/` — generated experiment output
- `notebooks/` — exploratory analysis
- `dashboard/` — future visualization tools
- `tests/` — automated checks

## Future milestones

Next steps are core performance and coordination metrics, blocking and interference measures, experiments across multiple layouts, and deterministic baseline agents. The framework may later support reinforcement learning, evolutionary methods, or quality-diversity approaches, but the immediate goal is a clean and reliable evaluation foundation.
