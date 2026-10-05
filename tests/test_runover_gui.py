"""Automated tests for the Overcooked-AI Interactive Runner GUI."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

# Ensure headless execution for testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

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

    def test_layout_discovery(self) -> None:
        layouts = get_available_layouts()
        self.assertIn("cramped_room", layouts)
        self.assertIn("asymmetric_advantages", layouts)
        self.assertIn("coordination_ring", layouts)

    def test_live_layout_change(self) -> None:
        """Verify layout changes immediately update the environment."""
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

    def test_simulation_run_to_done(self) -> None:
        """Verify simulation execution, step count, and transition to DONE."""
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_or_restart_game()
        self.assertEqual(self.app.state, AppState.RUNNING)

        for _ in range(10):
            self.app._step_simulation()

        self.assertEqual(self.app.state, AppState.DONE)
        self.assertEqual(self.app.step_count, 10)
        self.assertEqual(len(self.app.recorded_rows), 10)

    def test_pause_resume_toggle(self) -> None:
        self.app.selected_agent_0_type = "greedy"
        self.app.selected_agent_1_type = "stay"
        self.app.start_or_restart_game()
        self.assertEqual(self.app.state, AppState.RUNNING)

        self.app.toggle_pause()
        self.assertEqual(self.app.state, AppState.PAUSED)

        self.app.toggle_pause()
        self.assertEqual(self.app.state, AppState.RUNNING)


if __name__ == "__main__":
    unittest.main()
