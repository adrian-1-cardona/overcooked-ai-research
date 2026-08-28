# Milestone 2A: Telemetry Schema and Multi-Episode Runner

## What I did

- Added a structured telemetry format for each experiment timestep.
- Added a reusable logger that collects telemetry and saves it as CSV.
- Added a runner for multiple episodes with two random baseline agents.
- Kept Overcooked-AI as the external, read-only environment.
- Kept all custom experiment and telemetry code inside `overcooked-agent-eval`.

## Why this matters

This milestone turns the project from a one-off baseline script into the start of an actual evaluation pipeline. Before comparing smarter agents, I need clean data that can be saved, reused, and analyzed across multiple episodes.

## Files changed

- `overcooked-agent-eval/telemetry/schema.py`
- `overcooked-agent-eval/telemetry/logger.py`
- `overcooked-agent-eval/experiments/run_multi_episode_baseline.py`
- `overcooked-agent-eval/tests/test_telemetry.py`
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

The terminal prints a short summary with the layout, episode count, timesteps logged, average episode length, average reward, and output path. The runner saves structured per-timestep telemetry to `results/multi_episode_random_baseline_cramped_room.csv`.

## Notes / assumptions

- The project uses Python 3.10 and expects the dependencies in `requirements.txt` to be installed.
- The Overcooked-AI submodule must be initialized and installed through the project requirements.
- The default seed makes the sequence of random actions repeatable, while each run gets a unique `run_id` for tracking saved data.
- Generated result CSV files remain ignored by Git.

## Next steps

- Add core performance and coordination metrics.
- Add blocking and interference metrics.
- Run experiments across multiple layouts.
- Start implementing deterministic baseline agents.
