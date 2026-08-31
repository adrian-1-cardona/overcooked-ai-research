# Overcooked-AI Research Workspace

This repository contains my senior project research on cooperative AI agents in Overcooked-AI. The goal is to build a reusable, scientific evaluation framework for running agents, collecting structured telemetry, measuring coordination quality, identifying failure modes, and studying partner compatibility across different cooperative strategies.

---

## Research Question

> **How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?**

Overcooked-AI requires two agents to share physical space, coordinate subtasks (gathering ingredients, cooking pots, plating soup, and delivering to service counters), and adapt to partner timing. A final score alone indicates whether a team succeeded, but it hides critical dynamics such as hallway blocking, duplicated work, idle time, role specialization, and partner friction.

---

## Current Status

- **Milestone 1 (Baseline Experiment):** Repeatable single-episode baseline experiment logging per-timestep actions and rewards.
- **Milestone 2 (Evaluation Foundation):** Comprehensive evaluation framework featuring:
  - Validated 27-field per-timestep telemetry schema.
  - Multi-episode and multi-layout runner (`cramped_room`, `asymmetric_advantages`, `coordination_ring`, etc.).
  - Reproducibility manifests (`*.manifest.json`) recording seeds, versions, and git commits.
  - 68-field per-episode performance, coordination (blocking, collisions, interference), role proxy (held-object shares, event rates), and task duplication metrics.
  - Batch aggregation tool reporting sample size ($N$), mean, sample standard deviation ($s$), min, and max.
  - Formal methods-grade metric specification.

---

## Repository Structure

- `external/overcooked_ai/` — Upstream Overcooked-AI Git submodule (read-only)
- `overcooked-agent-eval/` — Custom research framework:
  - `agents/` — Cooperative agent strategies and baselines
  - `experiments/` — Repeatable experiment runners and summarizers
  - `metrics/` — Coordination, role proxy, duplication, and aggregation metrics
  - `telemetry/` — Schema validation, logging, and reproducibility manifests
  - `results/` — Generated experiment CSVs and JSON manifests
  - `tests/` — Automated unittest suite
- `project_docs/` — Project proposal and research specifications
- `work_done/` — Milestones and PR write-ups

---

## Setup & Installation

Overcooked-AI requires **Python 3.10**.

```bash
# 1. Clone the repo and submodules
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git

# 2. Enter the evaluation directory and create the Python 3.10 virtual environment
cd overcooked-ai-research/overcooked-agent-eval
python3.10 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies and the Overcooked-AI submodule
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!IMPORTANT]
> All `python` commands must be executed with the virtual environment active. Whenever you open a new terminal:
> ```bash
> cd overcooked-agent-eval
> source .venv/bin/activate
> ```
> Alternatively, you can call the environment's Python directly:
> ```bash
> .venv/bin/python experiments/run_random_baseline.py
> ```

---

## Step-by-Step Guide: How to Run Each Milestone

Below are the exact commands to run through each milestone, whether working on this PR branch or after merging into `main`.

### Step 1: Milestone 1 — Run the Single-Episode Random Baseline

Runs one 400-step episode of two random agents on `cramped_room`:

```bash
python experiments/run_random_baseline.py
```
- **What it does:** Runs 1 episode and writes per-step actions and rewards to `results/random_baseline_cramped_room.csv`.

---

### Step 2: Milestone 2 — Run the Multi-Episode Experiment Runner

Runs a seeded batch of 5 reproducible episodes on `cramped_room`:

```bash
python experiments/run_multi_episode_baseline.py
```
- **What it does:** 
  - Runs 5 games (seeds 42 to 46).
  - Logs 27 validated fields per step to `results/multi_episode_random_baseline_cramped_room.csv`.
  - Saves a companion recipe manifest to `results/multi_episode_random_baseline_cramped_room.manifest.json`.

---

### Step 3: Milestone 2 — Run Experiments Across Multiple Layouts

Runs experiments across several different Overcooked-AI kitchen maps with one command:

```bash
python experiments/run_multi_episode_baseline.py --layouts cramped_room asymmetric_advantages coordination_ring --episodes 10
```
- **What it does:** 
  - Pre-validates map names before starting.
  - Runs 10 episodes for each layout.
  - Generates distinct CSV and JSON manifest files for each layout in `results/`.

---

### Step 4: Milestone 2 — Compute & Summarize Coordination, Role Proxies, & Duplication

Analyzes saved telemetry to compute 68 episode metrics and statistical batch summaries:

```bash
# Summarize all results in results/
python experiments/summarize_episode_metrics.py

# Or process a specific multi-episode telemetry file:
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room.csv
```
- **What it does:**
  - Computes `results/multi_episode_random_baseline_cramped_room_episode_metrics.csv` containing 68 metrics per game.
  - Prints clean terminal tables showing Average (mean), Variation (std), Min, Max, and Sample Size ($N$) for:
    - **Team Performance:** Score, soups delivered, episode length.
    - **Movement & Idle:** Distance walked, explicit idle time and rates.
    - **Collisions & Blocking:** Wall bumps, teammate blocking, same-target collisions, swaps, repeated interference.
    - **Role Proxies:** Held-object time-shares (summing to 100%), ingredient potting counts, dish/soup pickups, soup deliveries.
    - **Task Duplication & Pipeline:** Redundant gathering/dishing/delivering, unused pipeline timesteps.

---

### Step 5: Run the Automated Test Suite

Runs all 30 unit tests covering schema validation, deterministic seeding, multi-layout execution, manifests, and metrics:

```bash
python -m unittest discover -s tests
```

---

## Research Roadmap

- **Milestone 3 (Baseline Agents):** Implement deterministic baselines (`GreedyAgent`, `PrepAgent`, `RunnerAgent`, `SupportAgent`).
- **Milestone 4 (Partner Compatibility Matrix):** Run cross-play matrix across agent pairings and layouts.
- **Milestone 5 (Compatibility Fingerprints):** Fingerprint agent coordination profiles and identify self-play traps.
- **Milestone 6 (Failure Mode Taxonomy):** Formalize failure taxonomy (role mismatch, spatial gridlock, pipeline bottlenecks).
- **Milestone 7 (Robustness & Generalization):** Stress-test agents with noisy and sub-optimal teammates.
- **Milestone 8 (Research Write-Up):** Assemble figures, tables, and reproducible project thesis report.
