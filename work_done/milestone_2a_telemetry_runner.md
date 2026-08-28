# Milestone 2A: Telemetry Schema and Multi-Episode Runner

## What I did

- Added a documented and validated telemetry format for each experiment timestep.
- Added a reusable logger that collects telemetry, saves it as CSV, and validates saved files.
- Added a deterministic runner for multiple seeded episodes with two random baseline agents.
- Kept Overcooked-AI as the external, read-only environment.
- Kept all custom experiment and telemetry code inside `overcooked-agent-eval`.

## Why this matters

This milestone turns the project from a one-off baseline script into the start of an actual evaluation pipeline. Before comparing smarter agents, I need clean data that can be saved, reused, and analyzed across multiple episodes.

## Files changed

- `overcooked-agent-eval/telemetry/schema.py`
- `overcooked-agent-eval/telemetry/logger.py`
- `overcooked-agent-eval/telemetry/README.md`
- `overcooked-agent-eval/experiments/run_multi_episode_baseline.py`
- `overcooked-agent-eval/experiments/run_random_baseline.py`
- `overcooked-agent-eval/tests/test_telemetry.py`
- `overcooked-agent-eval/tests/test_multi_episode_runner.py`
- `README.md`
- `overcooked-agent-eval/README.md`
- `work_done/milestone_2a_telemetry_runner.md`

## How to run it

```bash
cd overcooked-agent-eval
python experiments/run_multi_episode_baseline.py
```

Run the focused telemetry tests with:

```bash
python -m unittest discover -s tests
```

## Expected output

The terminal prints a short summary with the layout, episode count, timesteps logged, average episode length, average reward, and output path. The runner saves and validates structured per-timestep telemetry in `results/multi_episode_random_baseline_cramped_room.csv`.

The CSV includes deterministic run, episode, and player identifiers, the episode seed, agent actions and state, sparse and shaped rewards, and Overcooked events. A 10-episode acceptance run is available with `python experiments/run_multi_episode_baseline.py --episodes 10`.

## Verification completed

- All 13 telemetry and runner tests passed.
- The required 10-episode command completed 4,000 timesteps.
- The saved CSV validated with episodes 1–10, seeds 42–51, and all 27 fields.
- Repeating the full 10-episode configuration produced a byte-for-byte identical CSV.
- The original Milestone 1 command still completed 400 timesteps and wrote its legacy CSV.

This completes GitHub issues #4 and #5. The telemetry schema is documented and validated, and the runner now supports repeatable seeded batches without duplicating the environment setup used by the single-episode command.

## Notes / assumptions

- The project uses Python 3.10 and expects the dependencies in `requirements.txt` to be installed.
- The Overcooked-AI submodule must be initialized and installed through the project requirements.
- Episode seeds start at the base seed and increase by one. The `run_id` is derived from the configuration, so repeated configurations produce identical data and IDs.
- Generated result CSV files remain ignored by Git.

## Next steps

- Run experiments across multiple layouts.
- Start implementing deterministic baseline agents.
