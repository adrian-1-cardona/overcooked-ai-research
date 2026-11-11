"""Test OvercookedPartnerMultiAgent environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from training.env_wrapper import OvercookedPartnerMultiAgent
from training.greedy_policy import GreedyRllibPolicy, WorkerStateContext


BENCHMARK_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]


def make_env(layout="cramped_room", partner_type="ppo", horizon=20):
    mdp = OvercookedGridworld.from_layout_name(layout)
    return OvercookedPartnerMultiAgent(
        base_env=OvercookedEnv.from_mdp(mdp, horizon=horizon),
        partner_type=partner_type,
        partner_prob=1.0,
        use_phi=False,
    )


def assert_primed_observations(env, observations):
    frames = env.base_env.lossless_state_encoding_mdp(env.base_env.state)
    for seat, agent_id in enumerate(env.curr_agents):
        if agent_id.startswith("ppo"):
            expected = np.concatenate([frames[seat]] * 4, axis=-1).astype(np.float32)
            np.testing.assert_array_equal(observations[agent_id], expected)
            assert observations[agent_id].dtype == np.float32
            assert env.ppo_observation_space.contains(observations[agent_id])
            assert env.observation_space[agent_id].contains(observations[agent_id])


@pytest.mark.parametrize("layout", BENCHMARK_LAYOUTS)
@pytest.mark.parametrize("partner_type", ["ppo", "greedy"])
def test_reset_primes_four_frames_and_space_bounds(layout, partner_type):
    env = make_env(layout, partner_type)
    observations = env.reset()
    raw_shape = env.base_env.lossless_state_encoding_mdp(env.base_env.state)[0].shape
    expected_shape = (*raw_shape[:-1], raw_shape[-1] * 4)

    assert env.ppo_observation_space.shape == expected_shape
    assert expected_shape[-1] == 104
    np.testing.assert_array_equal(env.ppo_observation_space.low, np.zeros(expected_shape))
    assert np.all(np.isposinf(env.ppo_observation_space.high))
    assert_primed_observations(env, observations)
    for agent_id, sample in env.observation_space_sample().items():
        assert env.observation_space[agent_id].contains(sample)


@pytest.mark.parametrize("layout", BENCHMARK_LAYOUTS)
def test_self_play_fifo_keeps_separate_seat_histories(layout):
    env = make_env(layout, horizon=8)
    observations = env.reset()
    initial_frames = env.base_env.lossless_state_encoding_mdp(env.base_env.state)
    histories = {
        agent_id: [initial_frames[seat].copy()] * 4
        for seat, agent_id in enumerate(env.curr_agents)
    }
    assert not np.array_equal(initial_frames[0], initial_frames[1])

    actions = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    for timestep in range(8):
        previous_observations = {key: value.copy() for key, value in observations.items()}
        joint_action = {
            agent_id: Action.ACTION_TO_INDEX[actions[(timestep + seat) % len(actions)]]
            for seat, agent_id in enumerate(env.curr_agents)
        }
        observations, rewards, dones, infos = env.step(joint_action)
        frames = env.base_env.lossless_state_encoding_mdp(env.base_env.state)
        for seat, agent_id in enumerate(env.curr_agents):
            histories[agent_id].append(frames[seat].copy())
            histories[agent_id] = histories[agent_id][-4:]
            expected = np.concatenate(histories[agent_id], axis=-1).astype(np.float32)
            np.testing.assert_array_equal(observations[agent_id], expected)
            np.testing.assert_array_equal(
                observations[agent_id][..., :-26],
                previous_observations[agent_id][..., 26:],
            )
            assert env.ppo_observation_space.contains(observations[agent_id])
        assert set(rewards) == set(observations)
        assert set(infos) == set(observations)
        assert dones["__all__"] == (timestep == 7)

    # Terminal history must not leak into the next episode.
    assert_primed_observations(env, env.reset())


def test_reset_clears_history_when_ppo_changes_seat(monkeypatch):
    monkeypatch.setattr(np.random, "shuffle", lambda agents: None)
    env = make_env(partner_type="greedy")
    observations = env.reset()
    assert env.curr_agents == ["ppo_0", "greedy_1"]
    for _ in range(5):
        observations, _, _, _ = env.step({
            agent_id: Action.ACTION_TO_INDEX[Direction.NORTH]
            for agent_id in observations
        })

    monkeypatch.setattr(np.random, "shuffle", lambda agents: agents.reverse())
    observations = env.reset()
    assert env.curr_agents == ["greedy_0", "ppo_1"]
    assert_primed_observations(env, observations)
    for _ in range(5):
        observations, _, _, _ = env.step({
            agent_id: Action.ACTION_TO_INDEX[Direction.EAST]
            for agent_id in observations
        })
    assert_primed_observations(env, env.reset())


def test_frame_history_is_owned_by_each_environment():
    env = make_env()
    other_env = make_env()
    observations = env.reset()
    other_observations = other_env.reset()
    other_initial = {key: value.copy() for key, value in other_observations.items()}
    for _ in range(5):
        observations, _, _, _ = env.step({
            agent_id: Action.ACTION_TO_INDEX[Direction.NORTH]
            for agent_id in observations
        })

    # Advancing another worker's environment must not change this environment's history.
    other_observations, _, _, _ = other_env.step({
        agent_id: Action.ACTION_TO_INDEX[Direction.EAST]
        for agent_id in other_observations
    })
    fresh_frames = other_env.base_env.lossless_state_encoding_mdp(other_env.base_env.state)
    for seat, agent_id in enumerate(other_env.curr_agents):
        expected = np.concatenate(
            [other_initial[agent_id][..., 26:], fresh_frames[seat]], axis=-1
        ).astype(np.float32)
        np.testing.assert_array_equal(other_observations[agent_id], expected)


def test_returned_observations_cannot_modify_retained_history():
    env = make_env()
    observations = env.reset()
    originals = {key: value.copy() for key, value in observations.items()}
    for observation in observations.values():
        observation.fill(-999)
    observations, _, _, _ = env.step({
        agent_id: Action.ACTION_TO_INDEX[Direction.NORTH]
        for agent_id in env.curr_agents
    })
    for agent_id in observations:
        np.testing.assert_array_equal(observations[agent_id][..., :-26], originals[agent_id][..., 26:])


def test_bc_partner_keeps_single_feature_vector():
    env = make_env(partner_type="bc")
    observations = env.reset()
    for _ in range(5):
        features = env.base_env.featurize_state_mdp(env.base_env.state)
        for seat, agent_id in enumerate(env.curr_agents):
            if agent_id.startswith("bc"):
                assert observations[agent_id].shape == (96,)
                np.testing.assert_array_equal(observations[agent_id], features[seat].astype(np.float32))
                assert env.bc_observation_space.contains(observations[agent_id])
            else:
                assert observations[agent_id].shape == (5, 4, 104)
        observations, _, _, _ = env.step({
            agent_id: Action.ACTION_TO_INDEX[Direction.NORTH]
            for agent_id in observations
        })


def test_partner_multiagent():
    print("Testing OvercookedPartnerMultiAgent...")
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(mdp, horizon=400)

    env = OvercookedPartnerMultiAgent(
        base_env=base_env,
        partner_type="greedy",
        partner_prob=1.0,
        use_phi=False,
    )

    obs = env.reset()
    print(f"Agents populated: {env.curr_agents}")
    assert any(a.startswith("ppo") for a in env.curr_agents)
    assert any(a.startswith("greedy") for a in env.curr_agents)
    assert_primed_observations(env, obs)
    greedy_seat = next(i for i, agent_id in enumerate(env.curr_agents) if agent_id.startswith("greedy"))
    state, context_mdp, context_seat = WorkerStateContext.get()
    assert state is base_env.state
    assert context_mdp is base_env.mdp
    assert context_seat == greedy_seat
    np.testing.assert_array_equal(obs[env.curr_agents[greedy_seat]], np.zeros(1, dtype=np.float32))

    # Instantiate greedy policy
    greedy_policy = GreedyRllibPolicy(env.greedy_observation_space, env.shared_action_space, {})

    # Compute actions
    actions_dict = {}
    for agent_id in env.curr_agents:
        if agent_id.startswith("ppo"):
            actions_dict[agent_id] = Action.ACTION_TO_INDEX[Direction.NORTH]
        else:
            act, _, _ = greedy_policy.compute_actions([obs[agent_id]])
            actions_dict[agent_id] = int(act[0])

    print(f"Action dict: {actions_dict}")
    next_obs, rewards, dones, infos = env.step(actions_dict)
    state, context_mdp, context_seat = WorkerStateContext.get()
    assert state is base_env.state
    assert context_mdp is base_env.mdp
    assert context_seat == greedy_seat
    for agent_id, observation in next_obs.items():
        if agent_id.startswith("ppo"):
            assert observation.shape == (5, 4, 104)
        else:
            np.testing.assert_array_equal(observation, np.zeros(1, dtype=np.float32))
            action, _, _ = greedy_policy.compute_actions([observation])
            assert env.shared_action_space.contains(int(action[0]))
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print("OvercookedPartnerMultiAgent test passed!")


if __name__ == "__main__":
    test_partner_multiagent()
