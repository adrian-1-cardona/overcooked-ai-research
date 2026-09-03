# Overcooked-AI Research Workspace

This repo is where I’m building out my senior project research work around cooperative AI agents in Overcooked-AI. The goal is not just to train an agent to play Overcooked. The goal is to build a reusable evaluation framework for running agents, collecting telemetry, comparing coordination behavior, and eventually studying partner compatibility across different cooperative strategies. Done by Adrian Cardona under advisor Professor Rodrigo Canaan.

## Research question

**How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?**

Overcooked-AI is useful for this work because two agents have to share space, divide tasks, and coordinate timing to prepare and deliver soups. A final score shows whether a team succeeded, but it does not fully explain blocking, duplicated work, idle time, specialization, or compatibility with a particular partner.

## Current status

Milestone 1 adds the first repeatable baseline experiment. Two random agents play one episode of `cramped_room`, and the script records their actions and rewards at every timestep. Before I try to build smarter agents, I need to make sure I can run the environment and collect useful data reliably.

## Repository structure

- `external/overcooked_ai/` — the upstream Overcooked-AI Git submodule (treated as read-only)
- `overcooked-agent-eval/` — my custom experiments, agents, metrics, and results
- `project_docs/` — project proposal and other planning documents
- `work_done/` — short write-ups for each completed milestone or PR

## Setup

Overcooked-AI currently requires Python 3.10. Clone the repo with its submodule, create a virtual environment, and install the project dependencies:

```bash
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git
cd overcooked-ai-research/overcooked-agent-eval
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Important:** All `python` commands must be run with the virtual environment active. If you open a new terminal, re-activate it first from inside `overcooked-agent-eval/`:
>
> ```bash
> source .venv/bin/activate
> ```
>
> Then run your script as normal:
>
> ```bash
> python experiments/run_random_baseline.py
> ```
>
> Or skip activation and call the venv's Python directly:
>
> ```bash
> .venv/bin/python experiments/run_random_baseline.py
> ```
>
> Running the system `python` without activating the venv will cause `ModuleNotFoundError: No module named 'overcooked_ai_py'`.

If the repo was cloned without submodules, initialize them from the repository root first:

```bash
git submodule update --init --recursive
```

## Run Milestone 1

From `overcooked-agent-eval/`, run the random baseline experiment:

```bash
python experiments/run_random_baseline.py
```

The terminal prints the layout, episode length, total sparse reward, number of logged timesteps, and output path. The detailed timestep log is written to `overcooked-agent-eval/results/random_baseline_cramped_room.csv`.

The default run uses one 400-timestep episode and a fixed random seed. Use `python experiments/run_random_baseline.py --help` to see options for the layout, horizon, seed, and output path.

Then summarise the results:

```bash
python experiments/summarize_episode_metrics.py
```

This reads every CSV in `results/` and prints a formatted breakdown to the terminal. For a single-episode file it shows total score, shaped rewards per agent, and episode length. For multi-episode files it adds mean, std, min, and max across all episodes, covering score, movement distance, idle rate, wall collisions, teammate-blocking events, and interference rate.

You can also point it at a specific file:

```bash
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room_episode_metrics.csv
```

## Next steps

- Add more meaningful rule-based baseline agents.
- Add coordination metrics such as idle time, blocking, and role specialization.
- Compare agent pairings across different layouts.
- Move toward evaluating partner compatibility instead of score alone.
