"""Overcooked inference adapter matching the PPO training observation history."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
from human_aware_rl.rllib.rllib import RlLibAgent

from training.env_wrapper import FRAME_STACK_SIZE


class FrameStackedRlLibAgent(RlLibAgent):
    """Give a trained PPO policy the same four-frame history used in training."""

    def __init__(self, policy: Any, agent_index: int, featurize_fn: Callable) -> None:
        self._single_frame_encoding = featurize_fn
        super().__init__(policy, agent_index, self._stack_frames)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._frames = [deque(maxlen=FRAME_STACK_SIZE) for _ in range(2)]
        self._last_timestep: int | None = None
        self._stacked_observations: tuple[np.ndarray, ...] | None = None

    def _stack_frames(self, state: Any, debug: bool = False) -> tuple[np.ndarray, ...]:
        # The GUI can request both probabilities and an action for the same
        # state. Each simulation timestep must contribute exactly one frame.
        if self._last_timestep == state.timestep:
            return self._stacked_observations

        observations = self._single_frame_encoding(state)
        for frames, observation in zip(self._frames, observations):
            frame = np.asarray(observation, dtype=np.float32)
            if not frames:
                frames.extend([frame] * FRAME_STACK_SIZE)
            else:
                frames.append(frame)
        self._last_timestep = state.timestep
        self._stacked_observations = tuple(
            np.concatenate(frames, axis=-1) for frames in self._frames
        )
        return self._stacked_observations
