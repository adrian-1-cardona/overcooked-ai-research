# Milestone 1: Baseline Sanity Check

**Milestone Status:** Completed & Merged  
**Pull Request:** [PR #1: Add random baseline Overcooked experiment](https://github.com/adrian-1-cardona/overcooked-ai-research/pull/1)  
**Branch:** \`milestone-1-baseline-experiment\` (merged into \`main\`)

---

## 1. Executive Summary & Research Context

Milestone 1 is the initial sanity check and integration milestone for the Overcooked-AI research workspace.

Before designing advanced cooperative AI agents, custom telemetry schemas, or partner compatibility metrics, it is vital to establish a working, reproducible bridge between the local custom project framework and the upstream \`overcooked_ai\` benchmark environment.

This milestone verified:
1. Local execution of Overcooked-AI under Python 3.10.
2. Direct execution of two random baseline agents in the \`cramped_room\` kitchen layout.
3. Collection and persistence of per-timestep actions, rewards, positions, and cumulative scores to CSV.
4. Deterministic repeatability using explicit random seeds.

---

## 2. Scientific & Engineering Rationale: Why Milestone 1 Matters

### ① Environment Isolation & Submodule Integrity
Rather than forking or modifying the upstream \`overcooked_ai\` codebase directly, Milestone 1 embeds \`overcooked_ai\` as a read-only Git submodule at \`external/overcooked_ai/\`. This keeps the upstream benchmark pristine while building the custom evaluation architecture in \`overcooked-agent-eval/\`.

### ② Establishing the Baseline Lower Bound
The random agent pairing provides the true **lower-bound baseline** for all future coordination comparisons. In cooperative games, random agents rarely deliver soups due to the multi-step prerequisite sequence (gather 3 onions $\rightarrow$ pot $\rightarrow$ wait 20 ticks $\rightarrow$ grab dish $\rightarrow$ plate soup $\rightarrow$ deliver). Establishing this zero-point is essential for calculating relative improvement in subsequent milestones.

### ③ Deterministic Seeding Protocol
Milestone 1 established the project protocol of seeding both Python's \`random\` module and environment state generators, ensuring that baseline runs produce identical trajectory logs across machines.

---

## 3. Files Added & Modified in Milestone 1

- \`external/overcooked_ai/\` — Upstream Overcooked-AI Git submodule.
- \`overcooked-agent-eval/experiments/run_random_baseline.py\` — Single-episode baseline runner.
- \`overcooked-agent-eval/requirements.txt\` — Environment dependency specification.
- \`overcooked-agent-eval/README.md\` — Evaluation workspace documentation.
- \`README.md\` — Root repository guide.
- \`project_docs/project_proposal.md\` — Initial project research proposal.
- \`work_done/README.md\` — Milestone documentation index.
- \`.github/pull_request_template.md\` — PR quality checklist.

---

## 4. Step-by-Step Reproduction Guide

To run the Milestone 1 baseline experiment:

```bash
# 1. Initialize git submodule (if cloning fresh)
git submodule update --init --recursive

# 2. Enter evaluation framework and set up virtual environment
cd overcooked-agent-eval
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run the random baseline experiment
python experiments/run_random_baseline.py
```

---

## 5. Expected Output & Artifacts

The execution completes a 400-step episode and writes the output to \`results/random_baseline_cramped_room.csv\`.

**Terminal Output:**
```
Running Overcooked-AI baseline experiment...
Layout: cramped_room
Agents: RandomAgent vs RandomAgent
Horizon: 400
Episode complete!
Total reward / score: 0.0
Timesteps logged: 400
Saved results to: results/random_baseline_cramped_room.csv
```

**CSV Schema Structure:**
Each row logs: \`timestep\`, \`agent_0_action\`, \`agent_1_action\`, \`agent_0_pos\`, \`agent_1_pos\`, \`sparse_reward\`, \`shaped_reward\`, \`cumulative_sparse_reward\`, and \`done\`.

---

## 6. Assumptions & Limitations

- **Single Episode Only:** Milestone 1 evaluated a single episode ($N=1$), which is insufficient for statistical variance analysis. This was resolved in Milestone 2 by building the multi-episode batch runner.
- **Limited Telemetry:** Milestone 1 logged 9 flat fields. Milestone 2 expanded this into a formal 27-field schema with orientation, joint actions, and environment events.

---

## 7. Next Steps & Progression

Milestone 1 successfully established the foundation, paving the way for:
- **Milestone 2 (Completed):** 27-field schema, multi-episode/multi-layout runner, reproducibility manifests, and 68-field coordination metrics.
- **Milestone 3 (Next):** Deterministic heuristic baselines (\`Greedy\`, \`Prep\`, \`Runner\`, \`Support\`) and pretrained MARL adapter.
