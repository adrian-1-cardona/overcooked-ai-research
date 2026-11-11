# PPO frame stacking

`OvercookedPartnerMultiAgent` always supplies four consecutive PPO frames,
concatenated along the last axis, oldest first. Each PPO seat owns a bounded
FIFO history. Reset discards the previous episode and repeats the initial
frame four times. The base encoder remains `(W, H, 26)`; PPO observations and
their Box bounds are `(W, H, 104)`. Greedy continues using the raw game state,
and BC continues using its 96-feature vector.

The convolutional model takes its shape from the observation space. Custom
checkpoint playback uses the same four-frame history, advancing once per
simulation timestep. Custom checkpoint loading restores each saved input specification, so older
26-channel checkpoints remain usable alongside new 104-channel checkpoints;
bundled pretrained PPO and BC agents use their existing loader.

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

## Reference map

- **Mnih et al. (Nature 2015)** — `training/env_wrapper.py` gives PPO four observations along the channel axis; this supplies recent movement context for the coordination experiment ([paper](https://www.nature.com/articles/nature14236), [free PDF](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)).
- **Mnih et al. (2013)** — the same wrapper supplies a short history to the convolutional policy, letting this experiment study temporal inputs without adding recurrent layers ([paper](https://arxiv.org/abs/1312.5602)).
- **Daniel Takeshi (2016)** — the preprocessing discussion distinguishes frame stacking from frame skipping; this implementation keeps every Overcooked decision step and changes only observation history ([article](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)).
- **OpenAI Baselines** — `training/env_wrapper.py` follows the `FrameStack` FIFO, repeated-reset-frame and channel-concatenation pattern, adapted to independent PPO agent histories ([code](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)).
- **DeepMind Acme** — its frame-history wrapper is a reference for bounded history, while this project's multi-agent adaptation stays in `training/env_wrapper.py` ([code](https://github.com/google-deepmind/acme/blob/master/acme/wrappers/frame_stacking.py)).
- **DeepMind Dopamine** — `NATURE_DQN_STACK_SIZE = 4` provides reference context for the fixed history length; its Atari resizing, frame skipping and color pooling are not used for Overcooked ([code](https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py)).
- **Farama Gymnasium** — `FrameStackObservation` is a reference for reset padding and history semantics; this legacy RLlib/Gym multi-agent environment uses its own small wrapper rather than importing a single-agent wrapper ([docs](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStackObservation)).
- **Stable-Baselines3** — `VecFrameStack` provides reference context for independent environment histories; `training/env_wrapper.py` keeps equivalent ownership boundaries per environment and PPO seat ([docs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecframestack)).

These sources motivate the experiment; they do not establish that four frames
make Overcooked fully Markovian or guarantee better cooperation. A single encoded
frame already includes player orientation and pot state. Four frames add recent
position and object changes, which the policy may learn to use. Measure sparse
soup-delivery reward separately from shaped training reward, use both seats and
paired random seeds, and report recipe/checkpoint differences in comparisons.

## Final checkpoint evaluation

After both runs complete, run from the repository root:

```bash
TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 OMP_NUM_THREADS=1 \
  overcooked-agent-eval/.venv/bin/python -m evaluation.frame_stacking_eval
```

The experiment loads checkpoint specifications directly, with no trainer startup
or fallback policy. It evaluates the final frame-stacked PPO–Greedy and self-play
checkpoints, the existing 650-iteration single-frame PPO–Greedy and PPO mixed
partner checkpoints, and the exact GreedySymbolSearchAgent used during training.
Nine pairings × both seat orders × 20 seeds produce 360 episodes of 400 steps on
`cramped_room`. Other layouts are outside this training run's evaluation scope.

`results/frame_stacking/evaluation/episodes.csv` contains sparse delivery reward,
soups, blocked moves toward walkable tiles, blocked moves toward walls, and long
periods where both players remain stationary. Interactions are not counted as
blocked movement. `summary.json` includes per-pair means and paired-seed bootstrap
intervals for the key comparisons. Existing checkpoints are historical baselines;
a causal frame-stacking claim needs matched single-frame retraining across
multiple training seeds. The comparison with the mixed-partner PPO addresses
whether a new agent helps that fixed PPO partner more than plain Greedy does.
