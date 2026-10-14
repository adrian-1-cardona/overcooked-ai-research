"""RLLib Policy wrapper for GreedySymbolSearchAgent.

Allows RLLib multi-agent trainers (like PPOTrainer) to rollout Overcooked
episodes with GreedySymbolSearchAgent acting as a fixed, non-learning partner.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gym as gymnasium
from ray.rllib.policy import Policy as RllibPolicy
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from agents.greedy_symbol_search import GreedySymbolSearchAgent


class WorkerStateContext:
    """Thread/process-local storage for the latest OvercookedState and MDP in a worker."""
    _current_state: OvercookedState | None = None
    _current_mdp: OvercookedGridworld | None = None
    _current_agent_index: int = 1

    @classmethod
    def set(cls, state: OvercookedState | None, mdp: OvercookedGridworld | None, agent_index: int = 1) -> None:
        cls._current_state = state
        cls._current_mdp = mdp
        cls._current_agent_index = agent_index

    @classmethod
    def get(cls) -> tuple[OvercookedState | None, OvercookedGridworld | None, int]:
        return cls._current_state, cls._current_mdp, cls._current_agent_index


class GreedyRllibPolicy(RllibPolicy):
    """RLLib compatible non-learning Policy wrapping GreedySymbolSearchAgent."""

    def __init__(self, observation_space: Any, action_space: Any, config: dict[str, Any]) -> None:
        super().__init__(observation_space, action_space, config)
        self.greedy_agent = GreedySymbolSearchAgent()
        self._mdp: OvercookedGridworld | None = None

    def compute_actions(
        self,
        obs_batch: Any,
        state_batches: Any = None,
        prev_action_batch: Any = None,
        prev_reward_batch: Any = None,
        info_batch: Any = None,
        episodes: Any = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
        state, mdp, agent_index = WorkerStateContext.get()
        if mdp is not None and (self._mdp is None or self._mdp is not mdp):
            self._mdp = mdp
            self.greedy_agent.set_mdp(mdp)

        self.greedy_agent.set_agent_index(agent_index)

        actions: list[int] = []
        batch_size = len(obs_batch)
        for _ in range(batch_size):
            if state is not None:
                act, _ = self.greedy_agent.action(state)
                act_idx = Action.ACTION_TO_INDEX.get(act, Action.ACTION_TO_INDEX[Action.STAY])
            else:
                act_idx = Action.ACTION_TO_INDEX[Action.STAY]
            actions.append(act_idx)

        return np.array(actions, dtype=np.int64), [], {}

    def get_weights(self) -> dict[str, Any]:
        return {}

    def set_weights(self, weights: Any) -> None:
        pass

    def learn_on_batch(self, samples: Any) -> dict[str, Any]:
        return {}

    def compute_gradients(self, postprocessed_batch: Any) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def apply_gradients(self, gradients: Any) -> None:
        pass
