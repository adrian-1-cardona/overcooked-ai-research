# Overcooked-AI Research Workspace

This repository contains Adrian Cardona's senior project research on cooperative AI agents in Overcooked-AI, completed under the guidance of Professor Rodrigo Canaan.

The research asks: How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?

## Milestone 1

Milestone 1 provides the first repeatable baseline experiment. Two upstream `RandomAgent` instances play one episode while the project records their actions, rewards, positions, and completion state at every timestep.

This milestone establishes a simple data collection workflow before more advanced agents and coordination metrics are added.

## Run From the Project Root

Run every command in this README from the project root, which is the directory containing:

```text
README.md
external/
overcooked-agent-eval/
```

Do not change into `overcooked-agent-eval/` before running the commands below.

## Requirements

- Git
- Python 3.10
- The `external/overcooked_ai` Git submodule

The checked-out Overcooked-AI package requires Python `>=3.10,<3.11`.

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

Create the virtual environment and install the local Overcooked-AI package:

```bash
python3.10 -m venv overcooked-agent-eval/.venv
source overcooked-agent-eval/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./external/overcooked_ai
```

When opening a new terminal, return to the project root and reactivate the environment:

```bash
source overcooked-agent-eval/.venv/bin/activate
```

## Run the Milestone 1 Baseline

```bash
python overcooked-agent-eval/experiments/run_random_baseline.py
```

The default run:

- Uses the `cramped_room` layout.
- Runs one episode.
- Stops after 400 timesteps.
- Uses random seed 42.
- Uses two random agents that can move, stay, or interact.
- Writes one CSV row after every environment step.

The result is saved to:

```text
overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

Running the command again replaces the existing file at that path. It does not append new rows.

## Terminal Output

A completed run prints a summary in this format:

```text
Random baseline complete
Layout: cramped_room
Episode length: 400
Total reward / score: <final score>
Timesteps logged: 400
Output CSV: <absolute path>/overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

| Value | Meaning |
| --- | --- |
| `Layout` | The Overcooked-AI kitchen map used for the episode |
| `Episode length` | The number of environment steps completed |
| `Total reward / score` | The final cumulative team sparse reward |
| `Timesteps logged` | The number of data rows written to the CSV |
| `Output CSV` | The absolute path to the generated result file |

The score includes sparse team reward only. Per-agent shaped rewards are logged in the CSV for analysis, but they are not added to `Total reward / score`.

## CSV Output

The CSV has 11 columns and one row per completed environment step. The default run produces 400 data rows plus the header row.

| Column | Meaning |
| --- | --- |
| `episode` | Episode number. It is always `1` in this single-episode runner. |
| `timestep` | Completed step number. It starts at `1` and ends at `400` by default. |
| `agent_0_action` | Action attempted by agent 0: `north`, `south`, `east`, `west`, `stay`, or `interact`. |
| `agent_1_action` | Action attempted by agent 1. |
| `sparse_reward` | Team reward received during this step, usually from completing a delivery. |
| `agent_0_shaped_reward` | Additional diagnostic reward assigned to agent 0 during this step. |
| `agent_1_shaped_reward` | Additional diagnostic reward assigned to agent 1 during this step. |
| `cumulative_sparse_reward` | Running total of `sparse_reward`. Its final value is the terminal score. |
| `agent_0_position` | Agent 0's grid position after the action, stored as an `(x, y)` coordinate. |
| `agent_1_position` | Agent 1's grid position after the action. |
| `done` | `True` when the episode has ended and `False` otherwise. |

Each row is written after an environment step. The action columns contain the actions attempted during that step, and the position columns contain the resulting post-step positions. A movement action does not guarantee that an agent moved because a wall, counter, or teammate may block it.

## Change the Run Settings

The runner supports four options:

| Option | Meaning | Default |
| --- | --- | --- |
| `--layout` | Overcooked-AI layout name | `cramped_room` |
| `--horizon` | Maximum number of timesteps | `400` |
| `--seed` | NumPy random seed used by both random agents | `42` |
| `--output` | CSV output path | `overcooked-agent-eval/results/random_baseline_cramped_room.csv` |

Example:

```bash
python overcooked-agent-eval/experiments/run_random_baseline.py \
  --layout asymmetric_advantages \
  --horizon 200 \
  --seed 7 \
  --output overcooked-agent-eval/results/random_baseline_asymmetric_advantages.csv
```

When changing `--layout`, also set `--output`. Otherwise, the runner still uses the default filename containing `cramped_room` and replaces that file with data from the selected layout.

View the built-in option reference with:

```bash
python overcooked-agent-eval/experiments/run_random_baseline.py --help
```

## Summarize the Result

The result summarizer can read the Milestone 1 CSV and print a short report:

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py \
  overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

For this single-episode format, the report shows:

- Team sparse reward.
- Cumulative shaped reward for each agent.
- Episode length.
- Total timesteps.

This command prints the report to the terminal. It does not create another CSV for the Milestone 1 format.

## Reproducibility and Limits

The runner seeds NumPy before creating the two random agents. Repeating the same command with the same code and dependency versions should produce the same trajectory.

Milestone 1 has several intentional limits:

- It runs only one episode, so it does not provide averages or variation across games.
- It uses random agents rather than coordinated strategies.
- It records basic actions, rewards, and positions but not held objects, orientations, or detailed game events.
- It does not create a separate run manifest.
- Dependencies are not fully version locked, so identical behavior across different environments is not guaranteed.

## Repository Structure

- `external/overcooked_ai/`: Upstream Overcooked-AI Git submodule.
- `overcooked-agent-eval/experiments/`: Experiment runner and result summarizer.
- `overcooked-agent-eval/results/`: Generated CSV output.
- `overcooked-agent-eval/agents/`: Location for future custom agents.
- `overcooked-agent-eval/metrics/`: Location for evaluation metrics.
- `project_docs/`: Research proposal and planning documents.
- `work_done/`: Completed milestone and pull request write-ups.
