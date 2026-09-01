# Milestone 2: Complete Evaluation Foundation

**Milestone Status:** Completed & Verified  
**Pull Request:** [PR #12: Build evaluation foundation: telemetry, runner, and metrics](https://github.com/adrian-1-cardona/overcooked-ai-research/pull/12)  
**Branch:** \`milestone-2a-telemetry-runner\`  
**Target Submission Venues Supported:** IEEE CoG, AIIDE, FDG, NeurIPS/ICLR Multi-Agent Workshops

---

## 1. Executive Summary & Research Context

Milestone 2 establishes the core measurement, telemetry, and reproducibility foundation for the entire Overcooked-AI research workspace. 

In multi-agent cooperative AI literature, a standard failure mode of research projects is treating the final game score as the sole metric of success. However, aggregate game scores mask critical spatial, temporal, and role dynamics—such as corridor gridlocks, task preemption, pipeline starvation, and partner friction. Milestone 2 replaces ad-hoc score tracking with a **methods-grade, 27-field telemetry logging schema** and a **68-field per-episode coordination, role-proxy, and duplication metric engine**.

This milestone ensures that every subsequent agent architecture (Milestones 3–7) automatically inherits complete diagnostic telemetry, statistical batch summarization, and cryptographic reproducibility without altering core code.

---

## 2. Scientific & Engineering Rationale: Why Milestone 2 is a Strong Research Base

### ① Diagnostic Granularity Beyond Scalar Rewards
Standard reinforcement learning and heuristic evaluation in Overcooked-AI only report episodic reward or soup count. Milestone 2 decomposes gameplay into fine-grained behavioral signals:
- **Spatial Friction:** Differentiating directional movement failure caused by environmental terrain/walls from true teammate corridor blocking, same-target collisions, agent swaps, and repeated interference.
- **Role Specialization:** Tracking held-object time-shares ($P(\text{none}), P(\text{onion}), P(\text{tomato}), P(\text{dish}), P(\text{soup})$ summing strictly to $1.0$) and event rates (potting, plating, delivering).
- **Pipeline Efficiency:** Quantifying task duplication (redundant gathering or plating) and unused pipeline work (cooking soups left unserved).

### ② Elimination of Metric Bias
In many multi-agent studies, metrics are constructed *after* agents are designed, creating unconscious confirmation bias toward specific behaviors. In this project, the entire 68-metric suite was formally specified and mathematically bounded in `metrics/README.md` **before** writing heuristic or learning agents in Milestone 3.

### ③ Zero-Friction Decoupling for Future Milestones
The architecture strictly decouples:
$$\text{Agent Policy} \longrightarrow \text{Telemetry Logger} \longrightarrow \text{Episode Metric Engine} \longrightarrow \text{Batch Aggregator}$$
When adding new agents in Milestone 3 (`Greedy`, `Prep`, `Runner`, `Support`, or pretrained `RLPolicyAgent`), they plug directly into `run_multi_episode_baseline.py` and immediately generate full 68-field coordination profiles across layouts without any metric plumbing.

### ④ Cryptographic & Environmental Reproducibility
Reviewers at top-tier venues (IEEE CoG, AIIDE, AAMAS) scrutinize multi-agent reproducibility. Milestone 2 automatically generates companion `*.manifest.json` files alongside every CSV batch, recording:
- Experiment random seeds ($N$ episodes)
- Layout configurations and episode horizons
- Python environment version & platform architecture
- Exact Git commit hash of the custom research repo
- Exact Git submodule commit hash of upstream `overcooked_ai`

### ⑤ Rigorous Test Coverage
Every schema validator, metric boundary, deterministic seeding mechanism, and multi-layout parser is covered by an automated test suite (**30/30 unit tests passing in <0.10s**), preventing silent regressions during future agent development.

---

## 3. Technical Deep-Dive & Architecture

```
overcooked-agent-eval/
├── telemetry/
│   ├── schema.py          # 27-field validated telemetry row definition
│   ├── logger.py          # CSV/dict telemetry logger with pre-validation
│   ├── manifest.py        # Environment & git commit manifest recorder
│   └── README.md          # Telemetry schema specification
├── metrics/
│   ├── episode_metrics.py # 68-field performance, coordination, & role engine
│   └── README.md          # Formal mathematical metric specifications
├── experiments/
│   ├── run_multi_episode_baseline.py  # Multi-layout, multi-episode CLI runner
│   ├── summarize_episode_metrics.py  # Statistical batch aggregation tool
│   └── run_random_baseline.py        # Single-episode baseline runner
└── tests/
    ├── test_telemetry.py              # Schema & logging unit tests
    ├── test_episode_metrics.py        # Metric calculation unit tests
    └── test_multi_episode_runner.py   # Multi-episode & multi-layout tests
```

### 3.1. Telemetry Schema (27 Validated Fields)
Every step $t \in [0, H]$ logs structured fields:
- **Identification:** `run_id`, `layout_name`, `episode_idx`, `timestep`, `random_seed`
- **Agent State (Per Agent):** `agent_idx`, `agent_type`, `agent_pos_x`, `agent_pos_y`, `agent_orientation`, `held_object`
- **Actions & Dynamics:** `joint_action`, `agent_action`, `step_sparse_reward`, `step_shaped_reward`, `cumulative_reward`
- **Environment Events:** `potting_events`, `soup_pickup_events`, `delivery_events`, `is_terminal`

### 3.2. Per-Episode Metric Categories (68 Fields)
1. **Team Performance:** Team score, soups delivered, episode completion time, total steps.
2. **Movement & Activity:** Total Manhattan distance walked, explicit idle timesteps ($a_t = \text{STAY}$ or no movement), and idle rates.
3. **Blocking & Interference:**
   - `wall_collision_count`: Agent attempted to move into a counter or impassable wall.
   - `teammate_blocking_count`: Agent attempted to move into the cell occupied by their partner.
   - `same_target_collision_count`: Both agents attempted to step into the exact same cell concurrently.
   - `agent_swap_count`: Both agents attempted to traverse through each other in opposite directions.
   - `repeated_interference_count`: Consecutive blocking events over multiple timesteps.
4. **Held-Object Role Proxies:** Continuous proportion of episode holding `none`, `onion`, `tomato`, `dish`, `soup` (guaranteed to sum to 1.0 per agent).
5. **Task Duplication & Pipeline Starvation:**
   - Observable task states: $\text{gather}, \text{dish}, \text{deliver}, \text{idle}, \text{other}$.
   - `task_duplication_rate`: Fraction of steps where both agents duplicate the same subtask.
   - `unused_pipeline_steps`: Soup left cooked in pots without being plated/delivered before the horizon ends.

---

## 4. Critical Pitfalls Avoided in Milestone 2

1. **The "Score-Only Illusion":**
   * *Pitfall:* Evaluating agent quality solely through total score, missing cases where an agent achieves score despite heavy corridor fighting.
   * *Resolution:* Dedicated spatial blocking metrics separate physical friction from scoring throughput.
2. **Wall vs. Teammate Collision Conflation:**
   * *Pitfall:* Grouping all failed movement into generic "collisions."
   * *Resolution:* Directional attribution explicitly distinguishes environment bounds from teammate interference.
3. **Partial Multi-Layout Failures:**
   * *Pitfall:* Running long batch experiments across multiple layouts only to crash on the last layout due to invalid map names.
   * *Resolution:* Strict pre-execution layout validation verifies map validity before initializing environment instances.
4. **Schema Drift across Scripts:**
   * *Pitfall:* Inconsistent CSV column ordering across different experiments.
   * *Resolution:* Centralized `TelemetryRow` dataclass with `validate()` enforcing exact field types and column orders.

---

## 5. Step-by-Step Reproduction Guide

To run and verify the entire Milestone 2 evaluation foundation:

```bash
# 1. Navigate to the evaluation workspace and activate the Python 3.10 virtualenv
cd overcooked-agent-eval
source .venv/bin/activate

# 2. Run multi-episode experiment across standard layouts (10 episodes each)
python experiments/run_multi_episode_baseline.py   --layouts cramped_room asymmetric_advantages coordination_ring   --episodes 10   --horizon 400

# 3. Summarize all telemetry CSVs into 68 episode metrics and statistical tables
python experiments/summarize_episode_metrics.py

# 4. Or process a specific telemetry file and export metrics to CSV:
python experiments/summarize_episode_metrics.py   results/multi_episode_random_baseline_cramped_room.csv   --output results/cramped_room_summary.csv

# 5. Run the complete automated test suite
python -m unittest discover -s tests
```

---

## 6. Verification & Test Results

- **Unit Test Suite:** 30/30 tests passed (`Ran 30 tests in 0.085s - OK`).
- **Telemetry Schema Integrity:** 27 fields validated across 10,000+ timesteps without missing values.
- **Reproducibility Manifests:** Successfully written and verified for every batch run (`*.manifest.json`).
- **Multi-Layout Support:** Fully verified on `cramped_room`, `asymmetric_advantages`, and `coordination_ring`.

---

## 7. Traceability: GitHub Issues Closed in PR #12

This milestone fully addresses and closes the following issues:
- **#2:** Implement core episode performance and coordination metrics
- **#3:** Measure teammate blocking, collisions, and interference
- **#4:** Define and validate the gameplay telemetry schema
- **#5:** Build a reusable multi-episode experiment runner
- **#8:** Add repeatable experiments across multiple layouts
- **#13:** Measure task duplication and unused pipeline work
- **#14:** Record a reproducibility manifest for every experiment batch
- **#15:** Aggregate episode metrics with mean, spread, and sample size
- **#16:** Write a formal specification for every coordination metric
- **#17:** Add held-object and task-event role proxies
- **#35:** Merge conflicts resolution

---

## 8. Immediate Next Steps: Transition to Milestone 3

With the evaluation foundation complete and verified:
1. **Merge PR #12 into `main`**.
2. **Implement Milestone 3 Agents (`overcooked-agent-eval/agents/`):**
   - Heuristic Baselines: `GreedyAgent` (#7), `SupportAgent` (#9), `PrepAgent` / `RunnerAgent` (#11), `RoleSpecialistAgent` (#22).
   - Pretrained MARL Adapter: `RLPolicyAgent` (#36) for Self-Play PPO and Fictitious Co-Play (FCP).
3. **Execute the Partner Compatibility Matrix (Milestone 4, #6)** to generate the empirical data for conference submission.
