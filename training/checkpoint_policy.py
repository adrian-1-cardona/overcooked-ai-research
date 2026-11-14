"""Restore a PPO policy from its saved specification without starting a trainer."""
from __future__ import annotations

import pickle
from pathlib import Path

import ray.cloudpickle as cloudpickle
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.policy import create_policy_for_framework
from human_aware_rl.ppo.ppo_rllib import RllibPPOModel
from human_aware_rl.rllib.rllib import RlLibAgent

from training.frame_stacked_agent import FrameStackedRlLibAgent


def load_checkpoint_policy(checkpoint: str | Path):
    """Use the checkpoint's own input shape, preserving older 26-channel assets."""
    ModelCatalog.register_custom_model("MyPPOModel", RllibPPOModel)
    with Path(checkpoint).open("rb") as stream:
        saved = pickle.load(stream)
    worker = cloudpickle.loads(saved["worker"])
    spec = worker["policy_specs"]["ppo"]
    config = merge_dicts(worker["policy_config"], spec.config or {})
    policy = create_policy_for_framework(
        "ppo", spec.policy_class, config, spec.observation_space, spec.action_space,
        seed=0,
    )
    policy.set_state(worker["state"]["ppo"])
    return policy


def agent_from_policy(policy, seat, env):
    """Choose the inference adapter from saved model dimensions, never a toggle."""
    channels = policy.model.base_model.input_shape[-1]
    classes = {26: RlLibAgent, 104: FrameStackedRlLibAgent}
    if channels not in classes:
        raise ValueError(f"Unsupported PPO input channel count: {channels}")
    agent = classes[channels](policy, seat, env.lossless_state_encoding_mdp)
    agent.reset()
    agent.set_agent_index(seat)
    agent.set_mdp(env.mdp)
    return agent
