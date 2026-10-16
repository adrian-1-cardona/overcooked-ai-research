"""Unit tests for training recipes, dropdown selection, BC model integration, and slider syncing."""

import os
import sys
import unittest
from pathlib import Path
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gui.app import OvercookedApp
from gui.agent_manager import create_agent


class TestRecipeDropdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        from gui.agent_manager import evict_trainer_cache
        evict_trainer_cache()

    def setUp(self):
        self.app = OvercookedApp(initial_layout="cramped_room", horizon=50, fps=30)

    def test_dropdown_train_recipe_exists_with_all_options(self):
        self.assertTrue(hasattr(self.app, "dropdown_train_recipe"))
        options = [val for _, val in self.app.dropdown_train_recipe.options]
        expected = ["ppo", "ppo_sp", "ppo_bc", "greedy", "ppo_50_50", "custom"]
        for opt in expected:
            self.assertIn(opt, options)

    def test_bc_option_in_all_dropdowns(self):
        # Live game dropdowns
        live_0_vals = [val for _, val in self.app.dropdown_agent_0.options]
        live_1_vals = [val for _, val in self.app.dropdown_agent_1.options]
        self.assertIn("bc", live_0_vals)
        self.assertIn("bc", live_1_vals)

        # Train setup dropdowns
        train_1_vals = [val for _, val in self.app.dropdown_train_agent_1.options]
        train_2_vals = [val for _, val in self.app.dropdown_train_agent_2.options]
        self.assertIn("bc", train_1_vals)
        self.assertIn("bc", train_2_vals)

    def test_recipe_preset_ppo(self):
        self.app._on_recipe_changed("ppo")
        self.assertEqual(self.app.train_recipe, "ppo")
        self.assertEqual(self.app.train_agent_1, "ppo")
        self.assertEqual(self.app.train_agent_1_pct, 100)
        self.assertEqual(self.app.slider_agent_1.value, 100)
        self.assertEqual(self.app.train_agent_2, "ppo")
        self.assertEqual(self.app.train_agent_2_pct, 0)
        self.assertEqual(self.app.slider_agent_2.value, 0)

    def test_recipe_preset_ppo_sp(self):
        self.app._on_recipe_changed("ppo_sp")
        self.assertEqual(self.app.train_recipe, "ppo_sp")
        self.assertEqual(self.app.train_agent_1, "ppo")
        self.assertEqual(self.app.train_agent_1_pct, 100)
        self.assertEqual(self.app.slider_agent_1.value, 100)
        self.assertEqual(self.app.train_agent_2, "ppo")
        self.assertEqual(self.app.train_agent_2_pct, 0)
        self.assertEqual(self.app.slider_agent_2.value, 0)

    def test_recipe_preset_ppo_bc(self):
        self.app._on_recipe_changed("ppo_bc")
        self.assertEqual(self.app.train_recipe, "ppo_bc")
        self.assertEqual(self.app.train_agent_1, "ppo")
        self.assertEqual(self.app.train_agent_1_pct, 50)
        self.assertEqual(self.app.slider_agent_1.value, 50)
        self.assertEqual(self.app.train_agent_2, "bc")
        self.assertEqual(self.app.train_agent_2_pct, 50)
        self.assertEqual(self.app.slider_agent_2.value, 50)

    def test_recipe_preset_greedy(self):
        self.app._on_recipe_changed("greedy")
        self.assertEqual(self.app.train_recipe, "greedy")
        self.assertEqual(self.app.train_agent_1, "ppo")
        self.assertEqual(self.app.train_agent_1_pct, 0)
        self.assertEqual(self.app.slider_agent_1.value, 0)
        self.assertEqual(self.app.train_agent_2, "greedy")
        self.assertEqual(self.app.train_agent_2_pct, 100)
        self.assertEqual(self.app.slider_agent_2.value, 100)

    def test_recipe_preset_ppo_greedy(self):
        self.app._on_recipe_changed("ppo_greedy")
        self.assertEqual(self.app.train_recipe, "ppo_greedy")
        self.assertEqual(self.app.train_agent_1, "ppo")
        self.assertEqual(self.app.train_agent_1_pct, 0)
        self.assertEqual(self.app.slider_agent_1.value, 0)
        self.assertEqual(self.app.train_agent_2, "greedy")
        self.assertEqual(self.app.train_agent_2_pct, 100)
        self.assertEqual(self.app.slider_agent_2.value, 100)

    def test_recipe_preset_ppo_50_50(self):
        self.app._on_recipe_changed("ppo_50_50")
        self.assertEqual(self.app.train_recipe, "ppo_50_50")
        self.assertEqual(self.app.train_agent_1, "ppo")
        self.assertEqual(self.app.train_agent_1_pct, 50)
        self.assertEqual(self.app.slider_agent_1.value, 50)
        self.assertEqual(self.app.train_agent_2, "greedy")
        self.assertEqual(self.app.train_agent_2_pct, 50)
        self.assertEqual(self.app.slider_agent_2.value, 50)

    def test_custom_slider_tweak_switches_to_custom(self):
        self.app._on_recipe_changed("ppo_bc")
        self.assertEqual(self.app.train_recipe, "ppo_bc")

        # User tweaks slider 1 to 70% and slider 2 to 30%
        self.app._on_slider_1_changed(70)
        self.app._on_slider_2_changed(30)
        self.assertEqual(self.app.train_recipe, "custom")
        self.assertEqual(self.app.dropdown_train_recipe.selected_value, "custom")

    def test_recipe_bottom_descriptions(self):
        expected_descriptions = {
            "ppo": ("PPO (Standard) (ppo)", "100% PPO self-play baseline"),
            "ppo_sp": ("PPO-SP (Self-Play) (ppo_sp)", "PPO self-play training"),
            "ppo_bc": ("PPO-BC (Human Model) (ppo_bc)", "PPO trained alongside the behavioral cloning human model (50/50 mix)"),
            "greedy": ("Greedy Specialist (greedy)", "PPO trained against the heuristic greedy chef"),
            "ppo_50_50": ("PPO 50/50 Mix (SP + Greedy) (ppo_50_50)", "Half self-play, half greedy specialist rollouts"),
            "custom": ("Custom Blend (Sliders) (custom)", "Custom partner allocation via sliders"),
        }

        self.app.active_mode = "train"
        self.app.train_view = "setup"

        for recipe_key, (exp_title, exp_desc) in expected_descriptions.items():
            self.app._on_recipe_changed(recipe_key)
            info = self.app.get_recipe_info()
            self.assertEqual(info["title"], exp_title)
            self.assertEqual(info["desc"], exp_desc)
            # Render frame to verify visual drawing works without error
            self.app._draw_frame()

    def test_bc_agent_creation_and_action(self):
        mdp = self.app.mdp
        agent, warning = create_agent("bc", agent_index=1, mdp=mdp, layout_name="cramped_room")
        self.assertIsNotNone(agent)
        self.assertIsNone(warning)
        action, _ = agent.action(self.app.env.state)
        self.assertIn(action, [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), "interact"])

    def test_custom_agent_missing_checkpoint_falls_back_to_greedy(self):
        from pathlib import Path
        mdp = self.app.mdp
        custom_model = Path(__file__).resolve().parents[1] / "models" / "ppo_v_greedy"
        if custom_model.is_dir():
            agent, warning = create_agent(f"custom:{custom_model}", agent_index=1, mdp=mdp, layout_name="bonus_order_test")
            self.assertIsNotNone(agent)
            self.assertIn("has no checkpoint", warning)
            action, _ = agent.action(self.app.env.state)
            self.assertIn(action, [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), "interact"])

    def test_gui_handles_incompatible_custom_agent_live_without_crashing(self):
        from pathlib import Path
        custom_model = Path(__file__).resolve().parents[1] / "models" / "ppo_v_greedy"
        if custom_model.is_dir():
            self.app._on_layout_changed("bonus_order_test")
            self.app._on_agent_0_changed("ppo")
            self.app._on_agent_1_changed(f"custom:{custom_model}")
            self.assertIsNotNone(self.app.warning_message)
            self.assertIn("ppo_v_greedy", self.app.warning_message)

            # Start game should start cleanly without crashing
            self.app.handle_run_or_restart_click()
            self.assertEqual(self.app.state.name, "RUNNING")
            self.assertEqual(len(self.app.agents), 2)
            self.app._step_simulation()
            self.assertGreaterEqual(self.app.step_count, 1)


if __name__ == "__main__":
    unittest.main()

