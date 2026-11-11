# PPO frame stacking

`OvercookedPartnerMultiAgent` always supplies four consecutive PPO frames,
concatenated along the last axis, oldest first. Each PPO seat owns a bounded
FIFO history. Reset discards the previous episode and repeats the initial
frame four times. The base encoder remains `(W, H, 26)`; PPO observations and
their Box bounds are `(W, H, 104)`. Greedy continues using the raw game state,
and BC continues using its 96-feature vector.

The convolutional model takes its shape from the observation space. Custom
checkpoint playback uses the same four-frame history, advancing once per
simulation timestep. Older custom checkpoints with 26 input channels require
retraining; bundled pretrained PPO and BC agents use their existing loader.

From the repository root, using the project environment:

```bash
overcooked-agent-eval/.venv/bin/python -u -m training.train_agent \
  --recipe ppo_greedy --layout cramped_room --iters 650 --num-workers 4 \
  --out-dir results/frame_stacking/ppo_greedy
overcooked-agent-eval/.venv/bin/python -u -m training.train_agent \
  --recipe ppo_sp --layout cramped_room --iters 650 --num-workers 4 \
  --out-dir results/frame_stacking/ppo_sp
```

Each output directory contains RLlib metrics in `logs/`, its own
`training_status.json`, checkpoints every 50 iterations, and a final checkpoint
with `metadata.json`. The shared GUI status shows the latest reporting run.
Generated frame-stacking results are ignored by Git.

`./run_tests.sh` checks FIFO ordering, reset padding, all five benchmark maps,
independent seats and environments, unchanged partner inputs, and checkpoint
playback.

The history follows the temporal-input approach of
[Mnih et al. (2013)](https://arxiv.org/abs/1312.5602) and
[Mnih et al. (2015)](https://www.nature.com/articles/nature14236), with the
reset/FIFO/channel-concatenation pattern used in
[OpenAI Baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py).
