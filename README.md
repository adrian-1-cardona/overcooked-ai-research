# Overcooked-AI Research Workspace

This repo contains my senior project research on cooperative AI agents in Overcooked-AI. The goal is to build a reusable, scientific evaluation framework for running agents, collecting structured telemetry, measuring coordination quality, identifying failure modes, and studying partner compatibility across different cooperative strategies.

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

## Setup

Overcooked-AI requires **Python 3.10**.

```bash
git clone --recurse-submodules https://github.com/adrian-1-cardona/overcooked-ai-research.git
cd overcooked-ai-research/overcooked-agent-eval
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!IMPORTANT]
> All `python` commands must be executed with the virtual environment active. In a new terminal, activate it first:
> ```bash
> source .venv/bin/activate
> ```
> Alternatively, invoke the environment's Python binary directly:
> ```bash
> .venv/bin/python experiments/run_multi_episode_baseline.py
> ```

If cloned without submodules, initialize them from the repository root:
```bash
git submodule update --init --recursive
```

---

## Running Experiments

### 1. Milestone 1: Single-Episode Baseline

Run a single-episode baseline on `cramped_room`:
```bash
python experiments/run_random_baseline.py
```
Output: `results/random_baseline_cramped_room.csv`.

---

### 2. Milestone 2: Multi-Episode & Multi-Layout Experiments

Run repeatable seeded batches across single or multiple layouts:

```bash
# Run 5 episodes on default cramped_room
python experiments/run_multi_episode_baseline.py

# Run 10 episodes across multiple layouts
python experiments/run_multi_episode_baseline.py --layouts cramped_room asymmetric_advantages coordination_ring --episodes 10
```

Each run generates:
- Structured per-timestep telemetry: `results/multi_episode_random_baseline_<layout>.csv`
- Companion reproducibility manifest: `results/multi_episode_random_baseline_<layout>.manifest.json`

---

### 3. Milestone 2: Summarize Performance & Coordination

Compute episode-level performance, coordination, role proxies, duplication metrics, and batch aggregates:

```bash
# Summarise all CSVs in results/
python experiments/summarize_episode_metrics.py

# Process a specific telemetry CSV and generate episode metrics CSV
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room.csv
```

Outputs:
- Per-episode metrics: `results/multi_episode_random_baseline_<layout>_episode_metrics.csv`
- Formatted summary tables to stdout (Sample size $N$, Mean, Std, Min, Max)

---

## Testing

Run the automated test suite from `overcooked-agent-eval/`:
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
