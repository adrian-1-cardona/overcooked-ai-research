"""MultiAgent Environment wrapper supporting Greedy partner training.

Repositories & Upstream Sources:
- HumanCompatibleAI/human_aware_rl:
  https://github.com/HumanCompatibleAI/human_aware_rl
  Extends `human_aware_rl.rllib.rllib.OvercookedMultiAgent` to support custom non-learning
  partner injection (such as GreedySymbolSearchAgent) alongside standard PPO and BC agents.
- HumanCompatibleAI/overcooked_ai:
  https://github.com/HumanCompatibleAI/overcooked_ai
  Provides the base game simulation environment, state transitions, and reward structures.
- Ray RLlib (ray-project/ray):
  https://github.com/ray-project/ray
  MultiAgentEnv interface for coordinating joint actions across heterogeneous policies.

Academic Papers & Citations:
- Carroll et al. (NeurIPS 2019):
  "On the Utility of Learning about Humans for Human-AI Coordination"
  https://arxiv.org/abs/1910.05789
  Pioneered the multi-agent Gym wrapping methodology for Overcooked-AI rollouts.
- Mnih et al. (2013, 2015):
  https://arxiv.org/abs/1312.5602
  https://www.nature.com/articles/nature14236
  Four consecutive observations provide temporal context to a feed-forward policy.
  FIFO history, repeated initial-frame padding, and channel concatenation follow
  OpenAI Baselines' FrameStack wrapper:
  https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from __future__ import annotations

import sys
from collections import deque
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
from overcooked_ai_py.mdp.actions import Action
from human_aware_rl.rllib.rllib import OvercookedMultiAgent

from training.greedy_policy import WorkerStateContext

FRAME_STACK_SIZE = 4


class OvercookedPartnerMultiAgent(OvercookedMultiAgent):
    """Multi-agent environment with four-frame PPO histories and raw partner inputs."""

    supported_agents = ["ppo", "bc", "greedy"]

    def __init__(
        self,
        base_env: Any,
        reward_shaping_factor: float = 0.0,
        reward_shaping_horizon: float = 0,
        bc_schedule: Any = None,
        use_phi: bool = True,
        partner_type: str = "greedy",
        partner_prob: float = 1.0,
    ) -> None:
        self.partner_type = partner_type
        self.partner_prob = partner_prob
        super().__init__(
            base_env,
            reward_shaping_factor=reward_shaping_factor,
            reward_shaping_horizon=reward_shaping_horizon,
            bc_schedule=bc_schedule,
            use_phi=use_phi,
        )
        self.greedy_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def _setup_observation_space(self, agents: list[str]) -> None:
        super()._setup_observation_space(agents)
        single_frame_space = self.ppo_observation_space
        self.ppo_observation_space = gymnasium.spaces.Box(
            low=np.concatenate([single_frame_space.low] * FRAME_STACK_SIZE, axis=-1),
            high=np.concatenate([single_frame_space.high] * FRAME_STACK_SIZE, axis=-1),
            dtype=single_frame_space.dtype,
        )
        self.greedy_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        all_possible_agents = [
            "ppo_0", "ppo_1", "greedy_0", "greedy_1", "bc_0", "bc_1", "random_0", "random_1"
        ]
        ob_space = {}
        for a in all_possible_agents:
            if a.startswith("ppo"):
                ob_space[a] = self.ppo_observation_space
            elif a.startswith("bc"):
                ob_space[a] = self.bc_observation_space
            else:
                ob_space[a] = self.greedy_observation_space
        self.observation_space = gymnasium.spaces.Dict(ob_space)

    def _setup_action_space(self, agents: list[str]) -> None:
        super()._setup_action_space(agents)
        all_possible_agents = [
            "ppo_0", "ppo_1", "greedy_0", "greedy_1", "bc_0", "bc_1", "random_0", "random_1"
        ]
        action_sp = {
            a: gymnasium.spaces.Discrete(len(Action.ALL_ACTIONS))
            for a in all_possible_agents
        }
        self.action_space = gymnasium.spaces.Dict(action_sp)
        self.shared_action_space = gymnasium.spaces.Discrete(len(Action.ALL_ACTIONS))

    def observation_space_sample(self, agent_ids: list[str] | None = None) -> dict[str, Any]:
        if agent_ids is None:
            agent_ids = self.curr_agents
        obs = {}
        for a in agent_ids:
            if a.startswith("ppo"):
                obs[a] = self.ppo_observation_space.sample()
            elif a.startswith("bc"):
                obs[a] = self.bc_observation_space.sample()
            else:
                obs[a] = self.greedy_observation_space.sample()
        return obs

    def _get_featurize_fn(self, agent_id: str) -> Any:
        if agent_id.startswith("ppo"):
            return lambda state: self.base_env.lossless_state_encoding_mdp(state)
        if agent_id.startswith("bc"):
            return lambda state: self.base_env.featurize_state_mdp(state)
        if agent_id.startswith("greedy"):
            return lambda state: (np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32))
        raise ValueError(f"Unsupported agent type {agent_id}")

    def _update_worker_context(self) -> None:
        # Find which seat greedy is occupying, if any
        greedy_idx = 1
        for i, agent_name in enumerate(self.curr_agents):
            if agent_name.startswith("greedy"):
                greedy_idx = i
                break
        WorkerStateContext.set(
            self.base_env.state,
            self.base_env.mdp,
            agent_index=greedy_idx,
        )

    def _populate_agents(self) -> list[str]:
        agents = ["ppo"]
        if np.random.uniform() < self.partner_prob:
            agents.append(self.partner_type)
        else:
            agents.append("ppo")

        np.random.shuffle(agents)
        agents[0] = agents[0] + "_0"
        agents[1] = agents[1] + "_1"

        self._setup_action_space(agents)
        self._setup_observation_space(agents)
        return agents

    def reset(self, regen_mdp: bool = True) -> dict[str, Any]:
        obs = super().reset(regen_mdp=regen_mdp)
        # Reset also runs during the superclass constructor. Rebuild histories so
        # no frames survive an episode boundary or a change in partner/seat.
        self._ppo_frame_buffers: dict[str, deque[np.ndarray]] = {
            agent_id: deque([frame] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)
            for agent_id, frame in obs.items()
            if agent_id.startswith("ppo")
        }
        for agent_id, frames in self._ppo_frame_buffers.items():
            obs[agent_id] = np.concatenate(frames, axis=-1)
        self._update_worker_context()
        return obs

    def step(self, action_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        obs, rewards, dones, infos = super().step(action_dict)
        for agent_id, frames in self._ppo_frame_buffers.items():
            frames.append(obs[agent_id])
            # Oldest to newest; only PPO observations contain temporal history.
            obs[agent_id] = np.concatenate(frames, axis=-1)
        self._update_worker_context()
        return obs, rewards, dones, infos
