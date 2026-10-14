"""Test GreedyRllibPolicy integration with Ray/RLLib.

Verifies:
1. GreedyRllibPolicy inherits from ray.rllib.policy.Policy
2. compute_actions() accurately queries GreedySymbolSearchAgent
3. Returns valid action indices (0 to 5) compatible with Overcooked action space
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gymnasium
from ray.rllib.policy import Policy as RllibPolicy
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from agents.greedy_symbol_search import GreedySymbolSearchAgent


class WorkerStateContext:
    """Thread/process-local storage for the latest OvercookedState in a worker."""
    _current_state = None
    _current_mdp = None
    _current_agent_index = 1

    @classmethod
    def set(cls, state, mdp, agent_index=1):
        cls._current_state = state
        cls._current_mdp = mdp
        cls._current_agent_index = agent_index

    @classmethod
    def get(cls):
        return cls._current_state, cls._current_mdp, cls._current_agent_index


class GreedyRllibPolicy(RllibPolicy):
    """RLLib compatible non-learning Policy that wraps GreedySymbolSearchAgent."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.greedy_agent = GreedySymbolSearchAgent()
        self._mdp = None

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        state, mdp, agent_index = WorkerStateContext.get()
        if mdp is not None and (self._mdp is None or self._mdp is not mdp):
            self._mdp = mdp
            self.greedy_agent.set_mdp(mdp)

        self.greedy_agent.set_agent_index(agent_index)
        
        actions = []
        batch_size = len(obs_batch)
        for _ in range(batch_size):
            if state is not None:
                act, _ = self.greedy_agent.action(state)
                act_idx = Action.ACTION_TO_INDEX[act]
            else:
                act_idx = Action.ACTION_TO_INDEX[Action.STAY]
            actions.append(act_idx)

        return np.array(actions, dtype=np.int64), [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        return {}

    def compute_gradients(self, postprocessed_batch):
        return [], {}

    def apply_gradients(self, gradients):
        pass


def test_greedy_policy():
    print("Testing GreedyRllibPolicy initialization and compute_actions...")
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=400)
    env.reset()

    obs_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = gymnasium.spaces.Discrete(len(Action.ALL_ACTIONS))

    policy = GreedyRllibPolicy(obs_space, act_space, {})
    WorkerStateContext.set(env.state, mdp, agent_index=1)

    dummy_obs_batch = [np.zeros(10, dtype=np.float32)]
    actions, states, infos = policy.compute_actions(dummy_obs_batch)

    print(f"Action computed: {actions[0]} -> {Action.INDEX_TO_ACTION[actions[0]]}")
    assert actions[0] in range(len(Action.ALL_ACTIONS)), "Invalid action index"
    print("GreedyRllibPolicy verified successfully!")


if __name__ == "__main__":
    test_greedy_policy()
