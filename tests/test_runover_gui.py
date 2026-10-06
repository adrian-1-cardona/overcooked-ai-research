"""Automated tests for the Overcooked-AI Interactive Runner GUI."""

from __future__ import annotations

import os
import unittest
import warnings
from pathlib import Path

# Ensure headless execution for testing and suppress library noise
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import pygame
pygame.init()

from gui.app import OvercookedApp, AppState
from gui.agent_manager import (
    get_available_layouts,
    is_ppo_supported,
    create_agent,
    InteractiveHumanAgent,
    StayAgent,
)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


class TestOvercookedGUI(unittest.TestCase):
    def setUp(self) -> None:
        self.app = OvercookedApp(initial_layout="cramped_room", horizon=10)

    def tearDown(self) -> None:
        pygame.display.quit()

    def test_initial_state(self) -> None:
        self.assertEqual(self.app.state, AppState.SETUP)
        self.assertEqual(self.app.current_layout, "cramped_room")
        self.assertIsNotNone(self.app.env)
        self.assertIsNotNone(self.app.visualizer)
        # Red hat for chef 0, blue hat for chef 1
        self.assertEqual(self.app.visualizer.player_colors, ["red", "blue"])
        # In SETUP, dropdowns are enabled and view graph is hidden
        self.assertTrue(self.app.dropdown_layout.is_enabled)
        self.assertFalse(self.app.btn_view_graph.is_visible)

    def test_layout_discovery(self) -> None:
        layouts = get_available_layouts()
        self.assertIn("cramped_room", layouts)
        self.assertIn("asymmetric_advantages", layouts)
        self.assertIn("coordination_ring", layouts)

    def test_live_layout_change(self) -> None:
        """Verify layout changes immediately update the environment when in SETUP."""
        self.app._on_layout_changed("coordination_ring")
        self.assertEqual(self.app.current_layout, "coordination_ring")
        self.assertEqual(self.app.state, AppState.SETUP)
        self.assertEqual(self.app.step_count, 0)

    def test_agent_factory(self) -> None:
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        human, warn_h = create_agent("human", 0, mdp, "cramped_room")
        self.assertIsInstance(human, InteractiveHumanAgent)
        self.assertIsNone(warn_h)

        stay, warn_s = create_agent("stay", 1, mdp, "cramped_room")
        self.assertIsInstance(stay, StayAgent)
        self.assertIsNone(warn_s)

    def test_mid_run_locks_dropdowns(self) -> None:
        """Verify that dropdowns are locked mid-run and cannot change maps or agents."""
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_game()
        self.assertEqual(self.app.state, AppState.RUNNING)
        self.assertTrue(self.app.is_mid_run)

        # Trigger event cycle to apply mid-run lock
        self.app.handle_events()
        self.assertFalse(self.app.dropdown_agent_0.is_enabled)
        self.assertFalse(self.app.dropdown_agent_1.is_enabled)
        self.assertFalse(self.app.dropdown_layout.is_enabled)

        # Attempt to change layout while mid-run must be ignored
        self.app._on_layout_changed("corridor")
        self.assertEqual(self.app.current_layout, "cramped_room")

        # Pause mid-run: must remain locked
        self.app.toggle_pause()
        self.assertEqual(self.app.state, AppState.PAUSED)
        self.assertTrue(self.app.is_mid_run)
        self.app.handle_events()
        self.assertFalse(self.app.dropdown_layout.is_enabled)

    def test_view_graph_only_visible_when_done(self) -> None:
        """Verify that View Graph (Streamlit) button ONLY appears when DONE, not mid-run."""
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_game()

        # Step 5 steps (mid-run)
        for _ in range(5):
            self.app._step_simulation()
        self.app.handle_events()
        self.app._draw_frame()
        self.assertFalse(self.app.btn_view_graph.is_visible)

        # Pause (mid-run)
        self.app.toggle_pause()
        self.app.handle_events()
        self.app._draw_frame()
        self.assertFalse(self.app.btn_view_graph.is_visible)

        # Step remaining steps to reach horizon (DONE)
        self.app.toggle_pause()
        for _ in range(5):
            self.app._step_simulation()

        self.assertEqual(self.app.state, AppState.DONE)
        self.app.handle_events()
        self.app._draw_frame()
        # Now and only now is View Graph visible!
        self.assertTrue(self.app.btn_view_graph.is_visible)

    def test_game_over_reset_clears_everything(self) -> None:
        """Verify that hitting RESET at game over clears everything and returns to SETUP."""
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_game()

        # Step until horizon is reached (DONE)
        for _ in range(10):
            self.app._step_simulation()
        self.assertEqual(self.app.state, AppState.DONE)
        self.app.handle_events()
        self.app._draw_frame()

        # Both RESET and RUN AGAIN buttons are visible, and dropdowns are unlocked
        self.assertTrue(self.app.btn_reset.is_visible)
        self.assertTrue(self.app.btn_run.is_visible)
        self.assertEqual(self.app.btn_reset.text, "Reset")
        self.assertEqual(self.app.btn_run.text, "Run Again")

        # Click the RESET button
        self.app.btn_reset.on_click()

        # Check full reset to SETUP state
        self.assertEqual(self.app.state, AppState.SETUP)
        self.assertEqual(self.app.step_count, 0)
        self.assertEqual(self.app.cumulative_score, 0.0)
        self.assertEqual(len(self.app.recorded_rows), 0)
        self.assertFalse(self.app.btn_view_graph.is_visible)

        # Dropdowns are completely unlocked so user can pick different agents and layouts
        self.app.handle_events()
        self.assertTrue(self.app.dropdown_layout.is_enabled)
        self.assertTrue(self.app.dropdown_agent_0.is_enabled)
        self.assertTrue(self.app.dropdown_agent_1.is_enabled)

        # User chooses different agent and layout
        self.app.dropdown_agent_0.selected_value = "stay"
        self.app._on_agent_0_changed("stay")
        self.assertEqual(self.app.selected_agent_0_type, "stay")

        self.app.dropdown_layout.selected_value = "coordination_ring"
        self.app._on_layout_changed("coordination_ring")
        self.assertEqual(self.app.current_layout, "coordination_ring")

    def test_restart_mid_run_returns_to_setup(self) -> None:
        """Verify clicking Restart/Reset mid-run resets to SETUP and unlocks dropdowns."""
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_game()
        self.assertEqual(self.app.state, AppState.RUNNING)

        for _ in range(3):
            self.app._step_simulation()
        self.assertEqual(self.app.step_count, 3)

        # Click Reset mid-run
        self.app.reset_to_setup()
        self.assertEqual(self.app.state, AppState.SETUP)
        self.assertEqual(self.app.step_count, 0)
        self.app.handle_events()
        self.assertTrue(self.app.dropdown_layout.is_enabled)
        self.assertTrue(self.app.dropdown_agent_0.is_enabled)
        self.assertTrue(self.app.dropdown_agent_1.is_enabled)

    def test_draw_frame_all_states(self) -> None:
        """Verify rendering frames completes without error across all application states."""
        # 1. SETUP
        self.app._draw_frame()

        # 2. RUNNING
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_game()
        self.app._draw_frame()

        # 3. PAUSED
        self.app.toggle_pause()
        self.app._draw_frame()

        # 4. DONE
        self.app.state = AppState.DONE
        self.app.step_count = self.app.horizon
        self.app.cumulative_score = 40.0
    def test_dual_ppo_agents_forced_coordination(self) -> None:
        """Verify that two PPO agents can play together seamlessly on forced_coordination."""
        self.app._on_layout_changed("forced_coordination")
        self.app.selected_agent_0_type = "ppo"
        self.app.selected_agent_1_type = "ppo"
        self.app.start_game()
        self.assertEqual(self.app.state, AppState.RUNNING)
        self.assertEqual(len(self.app.agents), 2)

        # Step 5 simulation steps
        for _ in range(5):
            self.app._step_simulation()

        self.assertEqual(self.app.step_count, 5)

    def test_matrix_view_graphs_only_visible_when_done(self) -> None:
        """Verify that View Graphs in Matrix mode is ONLY visible when the benchmark is done."""
        self.app._on_mode_changed("matrix")
        self.app._draw_frame()
        # Initially, View Graphs must be hidden
        self.assertFalse(self.app.btn_matrix_view_graphs.is_visible)

        # While running, View Graphs must remain hidden
        self.app.matrix_running = True
        self.app.matrix_done = False
        self.app._draw_frame()
        self.assertFalse(self.app.btn_matrix_view_graphs.is_visible)

        # When done, View Graphs must become visible
        self.app.matrix_running = False
        self.app.matrix_done = True
        self.app._draw_frame()
        self.assertTrue(self.app.btn_matrix_view_graphs.is_visible)


if __name__ == "__main__":
    unittest.main()



