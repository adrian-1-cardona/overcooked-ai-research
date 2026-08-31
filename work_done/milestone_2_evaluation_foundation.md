# Milestone 2: Complete Evaluation Foundation

## What was built

This milestone completes the evaluation foundation for the Overcooked-AI research workspace, transforming baseline executions into a repeatable, scientific evaluation framework.

### 1. Gameplay Telemetry & Reproducibility (Issues #4, #5, #14)
- **Validated 27-Field Schema:** Structured per-timestep telemetry capturing deterministic run, episode, agent, and layout IDs, exact coordinates, joint actions, orientations, sparse and shaped rewards, held objects, and Overcooked events.
- **Reproducibility Manifests:** Every experiment batch automatically records a JSON reproducibility manifest alongside its CSV (`*.manifest.json`) detailing seeds, horizon, layout, Python environment, `overcooked_ai` submodule commit, and framework git commit.
- **Multi-Episode Runner:** Reusable CLI supporting customizable seeds, horizon, episode counts, and outputs.

### 2. Multi-Layout Experiment Execution (Issue #8)
- **Multi-Layout Support:** The runner accepts multiple layouts (`--layouts cramped_room asymmetric_advantages coordination_ring`) or comma-separated lists.
- **Pre-Execution Validation:** Layouts are validated before execution to prevent partial run failures.
- **Isolated Identifiable Outputs:** Each layout generates distinct telemetry and manifest outputs.

### 3. Comprehensive Coordination, Role, & Duplication Metrics (Issues #2, #3, #13, #16, #17)
- **Core Performance:** Team score, soups delivered, episode length, Manhattan distance traveled, explicit idle time, and idle rates.
- **Blocking & Collisions:** Directional movement failure attribution separating wall/terrain collision attempts from teammate blocking, same-target collisions, swaps, and repeated consecutive interference.
- **Held-Object & Event Role Proxies:** Held-object time-shares for `none`, `onion`, `tomato`, `dish`, and `soup` (guaranteed to sum to 1.0 per agent), and event rates for potting, dish pickup, soup pickup, and soup delivery.
- **Task Duplication & Pipeline:** Observable task classification (`gather`, `dish`, `deliver`, `idle`, `other`), team task duplication timesteps/rates, and unused pipeline timesteps/rates.
- **Formal Metric Specification:** Methods-grade specification written in `metrics/README.md` defining every formula, input, and boundary.

### 4. Metrics Aggregation & Inspection (Issue #15)
- **Batch Aggregations:** Groups by layout and agent pairing, calculating sample size ($N$), mean, sample standard deviation ($s$), min, and max for all numeric metrics.
- **Multi-Format Summarizer:** `summarize_episode_metrics.py` unifies processing for single-episode, multi-episode telemetry, and episode-level metrics with rich stdout tables and optional summary CSV export.

### 5. Documentation Architecture & Conflict Resolution (Issue #35)
- Fully resolved merge conflicts with `main`.
- Clean, layered documentation architecture connecting root `README.md`, `overcooked-agent-eval/README.md`, and subpackage guides.

---

## Files Changed / Added

- `overcooked-agent-eval/telemetry/schema.py`
- `overcooked-agent-eval/telemetry/logger.py`
- `overcooked-agent-eval/telemetry/manifest.py`
- `overcooked-agent-eval/telemetry/__init__.py`
- `overcooked-agent-eval/telemetry/README.md`
- `overcooked-agent-eval/metrics/episode_metrics.py`
- `overcooked-agent-eval/metrics/__init__.py`
- `overcooked-agent-eval/metrics/README.md`
- `overcooked-agent-eval/experiments/run_multi_episode_baseline.py`
- `overcooked-agent-eval/experiments/summarize_episode_metrics.py`
- `overcooked-agent-eval/experiments/run_random_baseline.py`
- `overcooked-agent-eval/tests/test_telemetry.py`
- `overcooked-agent-eval/tests/test_multi_episode_runner.py`
- `overcooked-agent-eval/tests/test_episode_metrics.py`
- `README.md`
- `overcooked-agent-eval/README.md`
- `work_done/milestone_2_evaluation_foundation.md`

---

## How to Run

```bash
cd overcooked-agent-eval
source .venv/bin/activate

# 1. Run multi-episode experiment across multiple layouts
python experiments/run_multi_episode_baseline.py --layouts cramped_room asymmetric_advantages coordination_ring --episodes 10

# 2. Summarize performance, coordination, role proxies, and duplication
python experiments/summarize_episode_metrics.py

# 3. Run full automated test suite
python -m unittest discover -s tests
```

---

## Verification Results

- **Automated Tests:** 30/30 unit tests passed.
- **Telemetry Schema:** 27 fields validated across thousands of timesteps.
- **Episode Summary:** 68 metric fields computed and validated per episode.
- **Multi-Layout Execution:** Verified across `cramped_room`, `asymmetric_advantages`, and `coordination_ring`.
- **Reproducibility Manifests:** Successfully written and validated for every batch.

---

## Issues Closed

- Closes #2
- Closes #3
- Closes #4
- Closes #5
- Closes #8
- Closes #13
- Closes #14
- Closes #15
- Closes #16
- Closes #17
- Closes #35
