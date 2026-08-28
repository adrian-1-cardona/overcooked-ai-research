# Overcooked-AI Research Workspace

This repo is where I’m building out my senior project research work around cooperative AI agents in Overcooked-AI. The goal is not just to train an agent to play Overcooked. The goal is to build a reusable evaluation framework for running agents, collecting telemetry, comparing coordination behavior, and eventually studying partner compatibility across different cooperative strategies.

## Research question

**How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?**

Overcooked-AI is useful for this work because two agents have to share space, divide tasks, and coordinate timing to prepare and deliver soups. A final score shows whether a team succeeded, but it does not fully explain blocking, duplicated work, idle time, specialization, or compatibility with a particular partner.

## Current status

Milestone 1 added the first repeatable baseline experiment. Two random agents play one episode of `cramped_room`, and the script records their actions and rewards at every timestep.

Milestone 2A makes the baseline more useful. Instead of just proving the environment runs once, this adds a reusable way to log structured telemetry across multiple episodes. The default run records five random-agent episodes in one consistent CSV schema, which is the first step toward comparing coordination and partner compatibility.

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

If the repo was cloned without submodules, initialize them from the repository root first:

```bash
git submodule update --init --recursive
```

## Run Milestone 1

From `overcooked-agent-eval/`, run:

```bash
python experiments/run_random_baseline.py
```

The terminal prints the layout, episode length, total sparse reward, number of logged timesteps, and output path. The detailed timestep log is written to `overcooked-agent-eval/results/random_baseline_cramped_room.csv`.

The default run uses one 400-timestep episode and a fixed random seed. Use `python experiments/run_random_baseline.py --help` to see options for the layout, horizon, seed, and output path.

## Run Milestone 2A

From `overcooked-agent-eval/`, run:

```bash
python experiments/run_multi_episode_baseline.py
```

The default experiment runs five random-agent episodes on `cramped_room`. It prints the episode count, total timesteps, average episode length, average score, and output path. Structured per-timestep telemetry is saved to `overcooked-agent-eval/results/multi_episode_random_baseline_cramped_room.csv`.

This reusable telemetry is important because future metrics and agent comparisons need the same fields across every episode. Each episode records its deterministic seed, and every saved CSV is schema-validated. Repeating the same configuration produces the same telemetry. See `overcooked-agent-eval/telemetry/README.md` for the complete field definitions and use `python experiments/run_multi_episode_baseline.py --help` to change the episode count, layout, horizon, base seed, or output path.

## Next steps

- Add core performance and coordination metrics.
- Add blocking, collision, and interference metrics.
- Run experiments across multiple layouts.
- Implement deterministic baseline agents.
