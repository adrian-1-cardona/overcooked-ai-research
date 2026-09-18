# Overcooked Agent Evaluation Framework

This folder contains my custom framework for evaluating cooperative AI agents. The upstream Overcooked-AI environment lives in `../external/overcooked_ai` as a Git submodule and should be treated as read-only. Custom agents, experiments, telemetry, metrics, tests, and analysis should be added here.

## Setup

Overcooked-AI currently requires Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run `git submodule update --init --recursive` from the repository root first if `../external/overcooked_ai` is empty.

## Experiments

### Gameplay rendering

Render Overcooked-AI gameplay live in an interactive Pygame desktop window, play interactively as Chef 0, export to an MP4 video, or replay existing CSV telemetry.

```bash
# 1. Live Pygame desktop window (default when GUI is available)
python experiments/render_gameplay.py

# 2. Interactive human mode: Play as Chef 0 with keyboard against an AI partner
python experiments/render_gameplay.py --agent-0 human

# 3. Export episode to an MP4 video (headless-safe)
python experiments/render_gameplay.py --mode video --horizon 200 --output results/gameplay.mp4

# 4. Replay a previously recorded CSV telemetry run
python experiments/render_gameplay.py --replay-csv results/random_baseline_cramped_room.csv

# 5. Render on different kitchen layouts at custom FPS
python experiments/render_gameplay.py --layout asymmetric_advantages --fps 15

# 6. Automatically generate and display Matplotlib performance graphs when done
python experiments/render_gameplay.py --plot
python experiments/render_gameplay.py --agent-0 human --plot

# 7. Plot dashboard from any existing telemetry CSV
python experiments/plot_run.py results/random_baseline_cramped_room.csv
```

**Window Controls:**
- `SPACE` / `P`: Pause / Resume simulation
- `RIGHT ARROW`: Step 1 timestep forward (when paused)
- `UP` / `DOWN`: Increase / Decrease FPS playback speed
- `R`: Restart episode
- `ESC` / `Q`: Exit window cleanly
- **Human Player Controls (`--agent-0 human`):** `WASD` / `Arrow Keys` to move; `SPACE` / `ENTER` / `F` to interact (pick up, drop, chop, cook).

### Random baseline

Runs two built-in random agents on `cramped_room`. Both agents sample from every available action, including `interact`, using a fixed seed so the run can be repeated.

```bash
python experiments/run_random_baseline.py
```

The script prints a short summary and creates `results/random_baseline_cramped_room.csv`. The CSV contains the episode number, timestep, each agent's action, timestep sparse reward, per-agent shaped rewards, cumulative sparse reward, and both player positions.

Optional settings are available with `python experiments/run_random_baseline.py --help`.

### Pretrained BC/PPO comparison

Compare the pretrained behavior-cloning partner with its PPO_BC teammate against
the PPO_SP self-play baseline. The experiment evaluates BC in both player seats,
runs the same total number of self-play episodes, and writes one summary row per
episode.

```bash
# Install the pretrained-agent dependencies once (Python 3.10)
python -m pip install -r requirements-harl.txt

# Apple Silicon only: Ray 2.0's declared gRPC wheel is Intel-only
python -m pip install --no-deps grpcio==1.51.3
```

#### Running the pretrained comparison

Run the standard comparison with the bundled `cramped_room` checkpoints. This
runs BC + PPO_BC in both player orders and the PPO_SP self-play baseline:

```bash
python experiments/run_pretrained_policy_comparison.py \
  --layout cramped_room \
  --episodes 20
```

Run fewer episodes while keeping the full 400-step gridworld horizon:

```bash
python experiments/run_pretrained_policy_comparison.py \
  --layout cramped_room \
  --episodes 5 \
  --horizon 400 \
  --seed 42 \
  --output results/cramped_room_bc_ppo_comparison.csv
```

Enforce the original paper-era 64-feature BC and 20-channel PPO observation
contract when using matching older checkpoints:

```bash
python experiments/run_pretrained_policy_comparison.py \
  --layout cramped_room \
  --episodes 20 \
  --strict-paper-observations
```

Run against custom PPO_BC and PPO_SP checkpoint directories:

```bash
python experiments/run_pretrained_policy_comparison.py \
  --layout cramped_room \
  --episodes 20 \
  --ppo-bc-checkpoint /path/to/ppo_bc_run \
  --ppo-sp-checkpoint /path/to/ppo_sp_run \
  --output results/custom_pretrained_comparison.csv
```

BC and PPO_SP intentionally use different observation adapters. BC receives the
handcrafted player-centric feature vector (distances to pots, dishes, onions,
and related state), while PPO_SP receives lossless spatial grid masks. The
paper-era checkpoints used 64 features and 20 channels; the current bundled
Overcooked-AI revision emits 96 features and 26 channels, so the runner records
the actual shapes. Pass `--strict-paper-observations` when evaluating an older
checkpoint that must enforce the original 64/20 contract.

Use `--ppo-bc-checkpoint` and `--ppo-sp-checkpoint` to evaluate other saved Ray
runs. Results default to `results/pretrained_policy_comparison.csv`.

#### Testing the pretrained comparison

Run the focused unit tests first:

```bash
python -m unittest discover -s tests -p 'test_pretrained_policy_comparison.py' -v
```

Then run a short end-to-end smoke test that restores both pretrained
checkpoints and exercises both observation encoders:

```bash
python experiments/run_pretrained_policy_comparison.py \
  --layout cramped_room \
  --episodes 1 \
  --horizon 20 \
  --output results/pretrained_policy_smoke_test.csv
```

For a full-length validation episode, change `--horizon 20` to
`--horizon 400`. To run every project test, use:

```bash
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  python -m unittest discover -s tests -v
```

### Human-proxy coordination analysis

The compatibility analysis pairs the frozen BC human-behavior proxy with both
its PPO_BC teammate and the PPO_SP policy. It also runs PPO_SP self-play as the
in-distribution baseline, evaluates BC in both player seats, and keeps each
condition balanced at the same total number of episodes.

Run the complete analysis, including full-horizon rollouts, numerical
coordination diagnostics, a comparison graph, timestep telemetry, and MP4
replays of the exact evaluated trajectories:

```bash
python experiments/analyze_human_model_coordination.py \
  --layout cramped_room \
  --episodes 5 \
  --horizon 400 \
  --seed 42 \
  --output-dir results/human_model_coordination
```

Run a faster end-to-end smoke test without video export:

```bash
python experiments/analyze_human_model_coordination.py \
  --layout cramped_room \
  --episodes 1 \
  --horizon 20 \
  --no-video \
  --output-dir results/human_model_coordination_smoke
```

Test the analysis logic directly:

```bash
python -m unittest discover -s tests -p 'test_human_model_coordination.py' -v
```

The output directory contains `episode_metrics.csv`, full `telemetry.csv`,
`comparison.png`, a concise generated report, and one representative MP4 per
condition unless `--no-video` is used. The graph reports return alongside joint
stationary time, blocked movement, and longest delivery drought. These are
transparent diagnostics of coordination lock, not causal proof; the BC policy
is a repeatable human-model proxy rather than a new human-subject study.

This same episode-metrics schema is the comparison boundary for future
evolutionary or LLM planning agents: add their condition rows, then compare
returns and coordination diagnostics against `BC+PPO_BC`, `BC+PPO_SP`, and
`PPO_SP+PPO_SP` in the generated graph.


### Summarising results

`summarize_episode_metrics.py` reads any CSV produced by this framework and prints a formatted summary to stdout. It auto-detects the file format.

```bash
# Summarise all CSVs in results/
python experiments/summarize_episode_metrics.py

# Summarise a specific file
python experiments/summarize_episode_metrics.py results/multi_episode_random_baseline_cramped_room_episode_metrics.csv
```

Three CSV formats are supported:

| Format | Typical filename pattern | Contents |
|---|---|---|
| Single-episode timestep telemetry | `random_baseline_*.csv` | One row per timestep; score, actions, positions |
| Multi-episode timestep telemetry | `multi_episode_*.csv` | One row per timestep across multiple episodes; richer agent state |
| Episode-level metrics | `*_episode_metrics.csv` | One row per episode; score, movement, idle, and collision stats |

For episode-level metrics files the summary includes mean, std, min, and max across all episodes.

Generated CSV files are ignored by Git, so running experiments does not add result data to a commit by default.

## Folder structure

- `agents/` - custom cooperative agent strategies
- `experiments/` - repeatable experiment runners
- `metrics/` - coordination and performance metrics
- `results/` - generated experiment output
- `notebooks/` - exploratory analysis
- `dashboard/` - future visualization tools
- `tests/` - automated checks

## Future milestones

Future work will add meaningful baseline strategies, coordination metrics, experiments across layouts and pairings, and partner compatibility analysis. The framework may later support reinforcement learning, evolutionary methods, or quality-diversity approaches, but the immediate goal is a clean and reliable evaluation foundation.
