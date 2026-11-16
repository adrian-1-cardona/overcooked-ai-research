"""Restore a PPO policy from its saved specification without starting a trainer."""
from __future__ import annotations

import pickle
from pathlib import Path

import ray.cloudpickle as cloudpickle
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.policy import create_policy_for_framework
from human_aware_rl.ppo.ppo_rllib import RllibLSTMPPOModel, RllibPPOModel
from human_aware_rl.rllib.rllib import RlLibAgent

from training.frame_stacked_agent import FrameStackedRlLibAgent


def load_checkpoint_policy(checkpoint: str | Path):
    """Use the checkpoint's own input shape, preserving older 26-channel assets."""
    with Path(checkpoint).open("rb") as stream:
        saved = pickle.load(stream)
    worker = cloudpickle.loads(saved["worker"])
    spec = worker["policy_specs"]["ppo"]
    config = merge_dicts(worker["policy_config"], spec.config or {})

    custom_config = config.get("model", {}).get("custom_model_config", {})
    use_lstm = custom_config.get("use_lstm", False)
    custom_cls = RllibLSTMPPOModel if use_lstm else RllibPPOModel
    model_name = config.get("model", {}).get("custom_model", "MyPPOModel")

    ModelCatalog.register_custom_model(model_name, custom_cls)
    ModelCatalog.register_custom_model("MyPPOModel", custom_cls)
    ModelCatalog.register_custom_model("MyLSTMPPOModel", RllibLSTMPPOModel)
    ModelCatalog.register_custom_model("RllibLSTMPPOModel", RllibLSTMPPOModel)
    ModelCatalog.register_custom_model("RllibPPOModel", RllibPPOModel)

    policy = create_policy_for_framework(
        "ppo", spec.policy_class, config, spec.observation_space, spec.action_space,
        seed=0,
    )
    policy.set_state(worker["state"]["ppo"])
    return policy


def agent_from_policy(policy, seat, env):
    """Choose the inference adapter from saved model dimensions, never a toggle."""
    if hasattr(policy, "observation_space") and policy.observation_space is not None:
        channels = policy.observation_space.shape[-1]
    elif hasattr(policy.model, "obs_space") and policy.model.obs_space is not None:
        channels = policy.model.obs_space.shape[-1]
    else:
        shape = policy.model.base_model.input_shape
        channels = shape[0][-1] if isinstance(shape, list) else shape[-1]

    classes = {26: RlLibAgent, 104: FrameStackedRlLibAgent}
    if channels not in classes:
        raise ValueError(f"Unsupported PPO input channel count: {channels}")
    agent = classes[channels](policy, seat, env.lossless_state_encoding_mdp)
    agent.reset()
    agent.set_agent_index(seat)
    agent.set_mdp(env.mdp)
    return agent
