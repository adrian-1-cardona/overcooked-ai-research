# Overcooked-AI Research Workspace

This repository contains Adrian Cardona's senior project research on cooperative AI agents in Overcooked-AI, completed under the guidance of Professor Rodrigo Canaan.

Research question: *How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?*

## Requirements

- Git
- Python 3.10 (`>=3.10,<3.11`)

## Setup

Clone the repository with its Overcooked-AI submodule:

```bash
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git
cd overcooked-ai-research
```

For an existing clone, initialize the submodule from the repository root:

```bash
git submodule update --init --recursive
```

Create the project environment and install Overcooked-AI with its dependencies, including Pygame:

```bash
python3.10 -m venv overcooked-agent-eval/.venv
source overcooked-agent-eval/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./external/overcooked_ai
```

Activate the environment again before running experiment scripts in a new terminal. The root renderer launcher selects this environment automatically when it exists.

## Render Gameplay

Run the default renderer:

```bash
python render.py
```

Common alternatives:

```bash
python render.py --agent-0 human
python render.py --mode video --horizon 200 --output overcooked-agent-eval/results/gameplay.mp4
python render.py --replay-csv overcooked-agent-eval/results/random_baseline_cramped_room.csv
python render.py --help
```

In human mode, use WASD or the arrow keys to move and Space, Enter, or F to interact. See [`overcooked-agent-eval/README.md`](overcooked-agent-eval/README.md) for window controls and additional examples.

## Run the Baseline

Run the reproducible random-agent baseline:

```bash
python overcooked-agent-eval/experiments/run_random_baseline.py
```

Summarize the generated telemetry:

```bash
python overcooked-agent-eval/experiments/summarize_episode_metrics.py \
  overcooked-agent-eval/results/random_baseline_cramped_room.csv
```

Generated results are written under `overcooked-agent-eval/results/` and are ignored by Git.

## Project Documentation

- [`overcooked-agent-eval/`](overcooked-agent-eval/) contains custom agents, experiments, metrics, and tests.
- [`project_docs/`](project_docs/) contains the proposal and research notes.
- [`work_done/`](work_done/) contains milestone reports and reproduction details.
- [`external/overcooked_ai/`](external/overcooked_ai/) is the read-only upstream submodule.
