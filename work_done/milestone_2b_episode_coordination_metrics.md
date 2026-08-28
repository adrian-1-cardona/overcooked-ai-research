# Milestone 2B: Episode Performance and Coordination Metrics

## What I did

- Added pre-action positions to the validated telemetry schema.
- Added saved-telemetry metrics for team score, soups delivered, episode length, distance traveled, and idle time.
- Added teammate blocking, wall/terrain collision, same-target collision, swap, interference, and repeated-interference metrics.
- Added one joinable summary CSV row per episode.
- Added hand-checked fixtures for the core metrics and coordination edge cases.
- Kept every change in `overcooked-agent-eval` and left the Overcooked-AI submodule unchanged.
- Installed the read-only submodule as a normal local package so Python 3.10.21 does not depend on a skipped hidden editable-install path file.

## Why this matters

Score alone does not explain how a team played. These summaries make it possible to compare task performance with observable coordination behavior and to identify where a pairing loses time through blocking or conflicting movement.

## Files changed

- `overcooked-agent-eval/telemetry/schema.py`
- `overcooked-agent-eval/telemetry/README.md`
- `overcooked-agent-eval/experiments/run_multi_episode_baseline.py`
- `overcooked-agent-eval/experiments/summarize_episode_metrics.py`
- `overcooked-agent-eval/metrics/__init__.py`
- `overcooked-agent-eval/metrics/episode_metrics.py`
- `overcooked-agent-eval/metrics/README.md`
- `overcooked-agent-eval/tests/test_episode_metrics.py`
- `overcooked-agent-eval/requirements.txt`
- `README.md`
- `overcooked-agent-eval/README.md`

## How to run it

```bash
cd overcooked-agent-eval
python experiments/run_multi_episode_baseline.py
python experiments/summarize_episode_metrics.py
```

## Expected output

The first command saves validated per-timestep telemetry. The second reads that saved data, prints a short performance and coordination summary, and writes `results/multi_episode_random_baseline_cramped_room_episode_metrics.csv` with one row per episode.

## Metric assumptions

- Idle means an explicit `stay` action, not a failed movement.
- A teammate block is separated from a wall/terrain failure using the requested direction and both agents’ pre-action and post-action positions.
- Same-target movement and swap attempts are joint collision events.
- Repeated interference begins on the second consecutive interference timestep.
- These metrics describe observable behavior, not intent or blame.

## GitHub issues completed

- #2: core episode performance and coordination metrics.
- #3: teammate blocking, collision, and interference metrics.

## Verification completed

- All 21 automated tests passed, including hand-checked metric fixtures.
- A 10-episode random baseline produced 4,000 validated telemetry rows.
- The saved-data command produced 10 joinable episode rows with 42 metric fields.
- Team distance and idle totals matched the sums of both agents in every episode.
- The real batch reported 2 deliveries, 354 blocking events, 127 joint collision events, and 481 interference timesteps.

## Next steps

- Run experiments across multiple layouts.
- Implement deterministic baseline agents.
