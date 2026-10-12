# Overcooked-AI Research Workspace

This repository contains Adrian Cardona's senior project research on cooperative AI agents in Overcooked-AI, completed under the guidance of Professor Rodrigo Canaan.

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
python runover.py
```

Generated results are written under `overcooked-agent-eval/results/` and are ignored by Git.
## Project Documentation

- [`overcooked-agent-eval/`](overcooked-agent-eval/) contains custom agents, experiments, metrics, and tests.
- [`project_docs/`](project_docs/) contains the proposal and research notes.
- [`work_done/`](work_done/) contains milestone reports and reproduction details.
- [`external/overcooked_ai/`](external/overcooked_ai/) is the read-only upstream submodule.
