# Overcooked-AI Research Workspace

This repository is a research framework for evaluating cooperative AI agents in Overcooked-AI. It runs repeatable games, records timestep telemetry, computes episode metrics, and helps explain how well two agents coordinate.

The current runnable baseline uses two upstream `RandomAgent` instances. The `overcooked-agent-eval/agents/` directory is a placeholder for future agent strategies.

## Run Everything From the Project Root

Every command in this README is meant to run from the project root, which is the directory containing:

```text
README.md
external/
overcooked-agent-eval/
```

Do not change into `overcooked-agent-eval/` before running the commands below.

## Setup

Overcooked-AI requires Python 3.10. Its package metadata supports Python `>=3.10,<3.11`.

For a new clone:

```bash
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git
cd overcooked-ai-research
```

For an existing clone, make sure the submodule is available:

```bash
git submodule update --init --recursive
```

Create the virtual environment and install Overcooked-AI:

```bash
python3.10 -m venv overcooked-agent-eval/.venv
source overcooked-agent-eval/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./external/overcooked_ai
```

When opening a new terminal, return to the project root and reactivate the environment:

```bash
source overcooked-agent-eval/.venv/bin/activate
```

## Quick Start

### 1. Run the default experiment

```bash
python overcooked-agent-eval/experiments/run_multi_episode_baseline.py
```

The default experiment:

- Uses the `cramped_room` layout.
- Runs 5 episodes with two random agents.
- Runs each episode for 400 timesteps.
- Uses seeds 42 through 46.
- Writes one telemetry row per timestep, for 2,000 rows total.

It creates:

```text
overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room.csv
overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room.manifest.json
```

### 2. Compute and view episode metrics

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py \
  overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room.csv
```

This validates the telemetry, prints a selected summary to the terminal, and creates:

```text
overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room_episode_metrics.csv
```

### 3. Save full batch statistics

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py \
  --input overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room.csv \
  --aggregate-output overcooked-agent-eval/results/cramped_room_aggregate_metrics.csv
```

This also writes a batch-level CSV containing the mean, sample standard deviation, minimum, and maximum for every numeric episode metric.

### 4. Run the test suite

```bash
PYTHONPATH=overcooked-agent-eval \
  python -m unittest discover -s overcooked-agent-eval/tests
```

## Configure an Experiment

Common runner options:

| Option | Meaning | Default |
| --- | --- | --- |
| `--layout` or `--layouts` | One or more Overcooked-AI layout names | `cramped_room` |
| `--episodes` | Number of episodes to run per layout | `5` |
| `--horizon` | Maximum timesteps in each episode | `400` |
| `--base-seed` or `--seed` | Seed for episode 1; later episodes add 1 | `42` |
| `--output` | Custom telemetry CSV path | Layout-based path in `results/` |

Example with three layouts and ten episodes per layout:

```bash
python overcooked-agent-eval/experiments/run_multi_episode_baseline.py \
  --layouts cramped_room asymmetric_advantages coordination_ring \
  --episodes 10 \
  --horizon 400 \
  --base-seed 42
```

Each layout uses seeds 42 through 51. Layout names are checked before any experiment starts. If a name is invalid, the command stops before writing results.

When `--output` is used with multiple layouts, the layout name is added to each filename. For example, `--output results/baseline.csv` produces files such as `results/baseline_cramped_room.csv`. A relative custom path is resolved from the project root.

Use `--help` to see every supported option:

```bash
python overcooked-agent-eval/experiments/run_multi_episode_baseline.py --help
python overcooked-agent-eval/experiments/summarize_episode_metrics.py --help
```

## What the Experiment Output Means

After each layout finishes, the runner prints:

| Terminal value | Meaning |
| --- | --- |
| `Layout` | Kitchen map used for the batch |
| `Episodes` | Number of completed games |
| `Timesteps logged` | Total telemetry rows across all episodes |
| `Average episode length` | Mean number of timesteps per game |
| `Average reward / score` | Mean team sparse reward, which is the game score |
| `Output CSV` | Absolute path to the raw telemetry |
| `Manifest JSON` | Absolute path to the run metadata |

Shaped rewards are recorded for analysis, but they are not included in `Average reward / score`.

### Generated files

| File | Contents |
| --- | --- |
| `multi_episode_random_baseline_LAYOUT.csv` | Raw telemetry with 27 columns and one row per timestep |
| `multi_episode_random_baseline_LAYOUT.manifest.json` | Run configuration and provenance |
| `multi_episode_random_baseline_LAYOUT_episode_metrics.csv` | 91 columns with one row per episode |
| Custom aggregate CSV | 336 columns with one row per layout and agent pairing |

The episode metrics CSV contains 8 identity columns and 83 numeric metrics. The aggregate CSV contains 4 identifying columns plus mean, sample standard deviation, minimum, and maximum columns for all 83 metrics.

Generated files are overwritten when the same output path is used again.

### Raw telemetry CSV

The 27 telemetry columns are grouped as follows:

| Group | What it records |
| --- | --- |
| Run identity | Run ID, episode, seed, timestep, layout, and agent names and IDs |
| Actions | The action selected by each agent, such as `north`, `stay`, or `interact` |
| Rewards | Team sparse reward plus per-agent sparse and shaped rewards |
| Completion | `done` is true only on the final row of an episode |
| State | Previous and current positions, orientations, and held objects |
| Events | Semicolon-separated game events for each agent, such as potting or delivery |

The manifest records the layout, agents, episode count, horizon, full seed list, output path, Python version, Overcooked-AI version, evaluation Git commit, and creation time. It is the record of how a batch was produced.

## What the Metrics Mean

The terminal report shows a useful subset of the complete episode metrics. The episode metrics CSV contains the full set.

| Metric group | Meaning |
| --- | --- |
| Performance | Team score, soups delivered, and episode length |
| Movement | Grid distance traveled by each agent and by the team |
| Idle | Explicit `stay` actions and the fraction of available agent timesteps spent staying |
| Wall collisions | Directional moves that failed because of a wall or counter |
| Teammate blocking | An agent tried to enter a cell that its teammate did not leave |
| Joint collisions | Both agents failed while targeting the same cell or trying to swap cells |
| Interference | Timesteps containing teammate blocking or a joint collision |
| Repeated interference | The second and later timesteps in a consecutive interference streak |
| Held-object role proxies | Time each agent spent holding nothing, onion, tomato, dish, or soup |
| Task-event role proxies | Logged potting, dish pickup, soup pickup, and delivery event counts |
| Task duplication | Both agents appeared to perform the same gather, dish, or deliver task |
| Unused pipeline | Soup remained outstanding after pickup while neither agent appeared to service the dish or delivery pipeline |

Important interpretation notes:

- `team_score` is the sum of team sparse reward. It is separate from shaped reward.
- `soups_delivered` counts delivery event labels. Do not assume score is always a fixed multiple of this count.
- Distance is Manhattan grid distance. A failed movement action adds no distance.
- Idle counts only an explicit `stay` action. A failed movement action is not idle.
- Collision, interference, role, duplication, and pipeline values are behavioral proxies. They describe observable patterns, not intent or blame.
- Each agent's five held-object shares add up to 1 for a complete episode.
- Task duplication means both agents were classified in the same task category. It does not prove that the work was unnecessary.
- The unused pipeline metric tracks soup after pickup until delivery. It does not directly inspect whether a cooked soup is waiting in a pot.
- Event metrics count logged event labels. They are not always the same as unique physical actions.

For reports with more than one episode, the terminal displays:

- `N`: Number of episodes.
- `mean`: Average value.
- `std`: Sample standard deviation, showing variation between episodes.
- `min`: Lowest episode value.
- `max`: Highest episode value.

For one episode, the terminal prints only the metric value. In an aggregate CSV, standard deviation is stored as `0` when there is only one episode.

## Summarizer Input and Output Behavior

Processing one raw telemetry file is the clearest workflow:

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py \
  overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room.csv
```

Running the summarizer without an input scans every non-empty CSV in `overcooked-agent-eval/results/`:

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py
```

That scan includes raw telemetry, legacy telemetry, and episode metric CSVs. If both raw and previously generated metric files exist, similar reports can appear more than once. Use a specific raw input when you only want one report.

`--output` changes the episode metrics destination. `--aggregate-output` is required to save aggregate statistics. Use custom output options with one raw input at a time so multiple inputs do not overwrite the same destination.

## Legacy Single-Episode Baseline

The earlier one-episode baseline is still available:

```bash
python overcooked-agent-eval/experiments/run_random_baseline.py
```

It runs one 400-step random-agent episode and writes:

```text
overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

## Repository Structure

- `external/overcooked_ai/`: Upstream Overcooked-AI Git submodule.
- `overcooked-agent-eval/experiments/`: Experiment runners and result summarizer.
- `overcooked-agent-eval/metrics/`: Episode metrics and aggregation logic.
- `overcooked-agent-eval/telemetry/`: Telemetry schema, validation, logging, and manifests.
- `overcooked-agent-eval/agents/`: Placeholder for future custom agent strategies.
- `overcooked-agent-eval/results/`: Generated CSV and JSON output.
- `overcooked-agent-eval/tests/`: Automated unit tests.
- `project_docs/`: Research proposal and specifications.
- `work_done/`: Milestone and pull request documentation.

## Planned Work

Future milestones add deterministic baseline agents, cross-play partner evaluation, compatibility fingerprints, failure-mode analysis, robustness experiments, and the final research report.
