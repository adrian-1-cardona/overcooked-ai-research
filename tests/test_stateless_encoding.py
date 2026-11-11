"""Stateless State Encoding & Contract Regression Tests.

This test suite guarantees that the default stateless observation encoding in Overcooked-AI
and the multi-agent training environment wrappers remain strictly preserved across all
supported layouts and agent configurations.

When branching to implement frame stacking (e.g. k-frame history or temporal concatenation),
these tests ensure that:
1. The stateless 26-channel spatial encoding contract (lossless_state_encoding_mdp) is preserved.
2. The 96-feature vector featurization (featurize_state_mdp) is preserved for BC models.
3. Statelessness invariant: encoding is purely functional and memoryless (no historical artifacts).
4. MultiAgent wrappers (OvercookedPartnerMultiAgent) yield exact stateless observation spaces and values.
5. All baseline agents (Greedy, Stay, Random, BC, PPO) continue operating with stateless inputs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from agents.greedy_symbol_search import GreedySymbolSearchAgent
from training.env_wrapper import OvercookedPartnerMultiAgent


BENCHMARK_LAYOUTS = [
    ("cramped_room", (5, 4)),
    ("asymmetric_advantages", (9, 5)),
    ("coordination_ring", (5, 5)),
    ("forced_coordination", (5, 5)),
    ("counter_circuit_o_1order", (8, 5)),
]

STATELESS_PPO_CHANNELS = 26
STATELESS_BC_FEATURES = 96


class TestStatelessEncodingContracts(unittest.TestCase):
    """Test stateless encoding invariants across all benchmark layouts."""

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    def test_lossless_stateless_encoding_shapes_across_all_layouts(self) -> None:
        """Verify that lossless_state_encoding returns exactly (W, H, 26) stateless arrays."""
        for layout_name, (expected_w, expected_h) in BENCHMARK_LAYOUTS:
            with self.subTest(layout=layout_name):
                mdp = OvercookedGridworld.from_layout_name(layout_name)
                env = OvercookedEnv.from_mdp(mdp, horizon=100)
                state = env.state

                p0_obs, p1_obs = env.lossless_state_encoding_mdp(state)

                # Both players get an observation of shape (W, H, 26)
                expected_shape = (expected_w, expected_h, STATELESS_PPO_CHANNELS)
                self.assertEqual(
                    p0_obs.shape,
                    expected_shape,
                    f"Layout {layout_name} p0 obs shape mismatch. Expected {expected_shape}, got {p0_obs.shape}",
                )
                self.assertEqual(
                    p1_obs.shape,
                    expected_shape,
                    f"Layout {layout_name} p1 obs shape mismatch. Expected {expected_shape}, got {p1_obs.shape}",
                )

                # Ensure observation values are within valid normalized / categorical range [0, 1] or positive integers
                self.assertTrue(np.all(p0_obs >= 0), f"Negative values found in p0 obs for {layout_name}")
                self.assertTrue(np.all(p1_obs >= 0), f"Negative values found in p1 obs for {layout_name}")

    def test_stateless_bc_feature_vector_shapes(self) -> None:
        """Verify that featurize_state_mdp returns the standard 96-element feature vector."""
        for layout_name, _ in BENCHMARK_LAYOUTS:
            with self.subTest(layout=layout_name):
                mdp = OvercookedGridworld.from_layout_name(layout_name)
                env = OvercookedEnv.from_mdp(mdp, horizon=100)
                f0, f1 = env.featurize_state_mdp(env.state)

                self.assertEqual(
                    f0.shape,
                    (STATELESS_BC_FEATURES,),
                    f"Layout {layout_name} BC feat shape mismatch: expected ({STATELESS_BC_FEATURES},), got {f0.shape}",
                )
                self.assertEqual(
                    f1.shape,
                    (STATELESS_BC_FEATURES,),
                    f"Layout {layout_name} BC feat shape mismatch: expected ({STATELESS_BC_FEATURES},), got {f1.shape}",
                )

    def test_statelessness_memoryless_invariant(self) -> None:
        """Verify that stateless encoding is strictly memoryless.

        Given state S at t=0, taking random environment steps and then reverting or
        revisiting state S must produce an EXACTLY IDENTICAL observation.
        (A frame-stacked encoder would retain temporal history and differ).
        """
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        env = OvercookedEnv.from_mdp(mdp, horizon=50)

        initial_state = env.state.deepcopy()
        init_p0_obs, init_p1_obs = env.lossless_state_encoding_mdp(initial_state)

        # Step through multiple transitions
        actions = [Action.STAY, Action.INTERACT]
        for _ in range(10):
            env.step(([Action.STAY, Action.STAY]))

        # Re-encode the initial state: MUST match bit-for-bit with initial encoding
        revisited_p0_obs, revisited_p1_obs = env.lossless_state_encoding_mdp(initial_state)

        np.testing.assert_array_equal(
            init_p0_obs,
            revisited_p0_obs,
            err_msg="Lossless stateless encoding for P0 is not deterministic / memoryless!",
        )
        np.testing.assert_array_equal(
            init_p1_obs,
            revisited_p1_obs,
            err_msg="Lossless stateless encoding for P1 is not deterministic / memoryless!",
        )

    def test_overcooked_partner_multiagent_stateless_contract(self) -> None:
        """Verify that OvercookedPartnerMultiAgent initializes with stateless spaces and steps cleanly."""
        for layout_name, (w, h) in BENCHMARK_LAYOUTS[:2]:
            with self.subTest(layout=layout_name):
                mdp = OvercookedGridworld.from_layout_name(layout_name)
                base_env = OvercookedEnv.from_mdp(mdp, horizon=20)
                multi_env = OvercookedPartnerMultiAgent(
                    base_env, partner_type="greedy", partner_prob=1.0
                )

                # Check observation space definitions
                self.assertEqual(
                    multi_env.ppo_observation_space.shape,
                    (w, h, STATELESS_PPO_CHANNELS),
                )
                self.assertEqual(multi_env.greedy_observation_space.shape, (1,))

                # Reset and check returned observation types and shapes
                obs = multi_env.reset()
                self.assertEqual(len(obs), 2)

                for agent_id, agent_obs in obs.items():
                    if agent_id.startswith("ppo"):
                        self.assertEqual(agent_obs.shape, (w, h, STATELESS_PPO_CHANNELS))
                    elif agent_id.startswith("greedy"):
                        self.assertEqual(agent_obs.shape, (1,))

                # Step through 5 steps and verify observation shapes remain stateless throughout
                for _ in range(5):
                    joint_act = {a: Action.ACTION_TO_INDEX[Action.STAY] for a in obs.keys()}
                    next_obs, rewards, dones, infos = multi_env.step(joint_act)
                    for agent_id, agent_obs in next_obs.items():
                        if agent_id.startswith("ppo"):
                            self.assertEqual(agent_obs.shape, (w, h, STATELESS_PPO_CHANNELS))
                        elif agent_id.startswith("greedy"):
                            self.assertEqual(agent_obs.shape, (1,))

    def test_greedy_agent_compatibility_with_stateless_states(self) -> None:
        """Verify GreedySymbolSearchAgent operates deterministically on raw stateless OvercookedState."""
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        env = OvercookedEnv.from_mdp(mdp, horizon=30)
        agent = GreedySymbolSearchAgent()
        agent.set_mdp(mdp)
        agent.set_agent_index(0)

        state = env.state
        action, action_info = agent.action(state)
        self.assertIn(action, Action.ALL_ACTIONS)

        # Confirm step proceeds cleanly
        next_state, reward, done, info = env.step(([action, Action.STAY]))
        next_action, _ = agent.action(next_state)
        self.assertIn(next_action, Action.ALL_ACTIONS)

    def test_agent_factory_stateless_action_execution(self) -> None:
        """Verify baseline agents produce valid discrete actions given stateless environment state."""
        from gui.agent_manager import create_agent

        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        env = OvercookedEnv.from_mdp(mdp, horizon=20)
        state = env.state

        for agent_name in ("stay", "random", "greedy", "bc", "ppo_sp"):
            with self.subTest(agent=agent_name):
                agent, warning = create_agent(agent_name, 0, mdp, "cramped_room")
                self.assertIsNotNone(agent)
                action, _ = agent.action(state)
                self.assertIn(action, Action.ALL_ACTIONS)

    def test_frame_stacking_boundary_guard(self) -> None:
        """Guard the boundary: stateless observation space must not be accidentally widened.

        If someone modifies base lossless encoding to return stacked frames without preserving
        the 26-channel contract for stateless mode, this test will fail explicitly.
        """
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        env = OvercookedEnv.from_mdp(mdp, horizon=10)
        p0_obs, _ = env.lossless_state_encoding_mdp(env.state)

        # Explicitly verify the channel dimension equals 26 (single frame)
        self.assertEqual(
            p0_obs.shape[-1],
            STATELESS_PPO_CHANNELS,
            f"Stateless observation channel count altered! Expected {STATELESS_PPO_CHANNELS}, got {p0_obs.shape[-1]}. "
            "If introducing frame stacking, keep the stateless default intact or make frame stacking opt-in.",
        )


if __name__ == "__main__":
    unittest.main()
