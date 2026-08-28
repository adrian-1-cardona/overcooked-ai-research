# Milestone 1: Random Baseline Experiment

## What I did

This milestone gets the first baseline running in Overcooked-AI from my own project folder.

- Added a random baseline experiment using two built-in random agents.
- Used Overcooked-AI as the environment and `cramped_room` as the default layout.
- Logged actions, rewards, positions, and timesteps to CSV.
- Added a fixed random seed so the default run is repeatable.
- Updated the setup and run instructions.

## Why this matters

This milestone is basically the first real sanity check for the project. Before I try to build smarter agents, I need to make sure I can run Overcooked-AI from my own project folder, collect basic data, and save results in a repeatable way.

## Files changed

- `overcooked-agent-eval/experiments/run_random_baseline.py`
- `overcooked-agent-eval/requirements.txt`
- `overcooked-agent-eval/README.md`
- `README.md`
- `project_docs/project_proposal.md`
- `work_done/README.md`
- `.github/pull_request_template.md`

## How to run it

Overcooked-AI currently requires Python 3.10. From the repository root:

```bash
git submodule update --init --recursive
cd overcooked-agent-eval
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python experiments/run_random_baseline.py
```

## Expected output

The terminal prints the layout, episode length, total reward or score, number of logged timesteps, and CSV path. The script creates `results/random_baseline_cramped_room.csv`, with one row per timestep containing both agents’ actions, rewards, cumulative sparse reward, positions, and episode completion status.

## Notes / assumptions

- The Overcooked-AI Git submodule must be initialized at `external/overcooked_ai`.
- The checked-out Overcooked-AI version declares support for Python 3.10.
- The default episode horizon is 400 timesteps and the default random seed is 42.
- Generated result CSV files are ignored by Git.

## Next steps

- Add more meaningful baseline agents.
- Add coordination metrics.
- Compare agent pairings across layouts.
- Start moving toward partner compatibility evaluation.
