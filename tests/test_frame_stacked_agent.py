"""Inference histories must match the histories supplied during PPO training."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "overcooked-agent-eval", REPO_ROOT / "external/overcooked_ai/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from training.frame_stacked_agent import FrameStackedRlLibAgent


class RecordingPolicy:
    model = SimpleNamespace()

    def __init__(self):
        self.observations = []

    def compute_actions(self, obs, state):
        self.observations.append(np.asarray(obs).copy())
        return np.zeros(len(obs), dtype=int), [], {
            "action_dist_inputs": np.zeros((len(obs), len(Action.ALL_ACTIONS)))
        }


def test_inference_fifo_matches_each_players_raw_encodings():
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=20)
    policies = [RecordingPolicy(), RecordingPolicy()]
    agents = [
        FrameStackedRlLibAgent(policy, seat, env.lossless_state_encoding_mdp)
        for seat, policy in enumerate(policies)
    ]
    initial = env.lossless_state_encoding_mdp(env.state)
    histories = [[frame.astype(np.float32)] * 4 for frame in initial]
    for step in range(7):
        for seat, agent in enumerate(agents):
            action, _ = agent.action(env.state)
            assert action in Action.ALL_ACTIONS
            expected = np.concatenate(histories[seat][-4:], axis=-1)
            actual = policies[seat].observations[-1][0]
            assert actual.shape == (5, 4, 104)
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual, expected)
            agent.action_probabilities(env.state)
            agent.action(env.state)
            np.testing.assert_array_equal(policies[seat].observations[-1][0], expected)
        env.step((Direction.NORTH if step % 2 else Direction.SOUTH, Direction.EAST))
        for history, frame in zip(histories, env.lossless_state_encoding_mdp(env.state)):
            history.append(frame.astype(np.float32))

    # Each agent owns its history even when another instance advances/resets.
    agents[0].reset()
    agents[0].action(env.state)
    current = env.lossless_state_encoding_mdp(env.state)[0]
    np.testing.assert_array_equal(
        policies[0].observations[-1][0], np.concatenate([current] * 4, axis=-1)
    )
    agents[1].action(env.state)
    np.testing.assert_array_equal(
        policies[1].observations[-1][0], np.concatenate(histories[1][-4:], axis=-1)
    )


def test_reset_repeats_initial_frame_after_a_new_episode():
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=20)
    policy = RecordingPolicy()
    agent = FrameStackedRlLibAgent(policy, 0, env.lossless_state_encoding_mdp)
    agent.action(env.state)
    env.step((Direction.SOUTH, Direction.WEST))
    agent.action(env.state)
    env.reset()
    agent.reset()
    agent.action(env.state)
    initial = env.lossless_state_encoding_mdp(env.state)[0]
    np.testing.assert_array_equal(
        policy.observations[-1][0], np.concatenate([initial] * 4, axis=-1)
    )
