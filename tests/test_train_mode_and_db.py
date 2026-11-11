"""Unit tests for Train Mode and Agent Database in OvercookedApp."""

import json
import os
import shutil
import sys
import unittest
from unittest.mock import patch, MagicMock
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
from gui.agent_database import AgentDatabase, MODELS_DIR, REGISTRY_FILE


class TestTrainModeAndDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        for p in MODELS_DIR.glob("TestJeff*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        if REGISTRY_FILE.is_file():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                recs = [r for r in recs if not r.get("name", "").startswith("TestJeff")]
                with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                    json.dump(recs, f, indent=2)
            except Exception:
                pass
        self.app = OvercookedApp(initial_layout="cramped_room", horizon=50, fps=30)

    def tearDown(self):
        for p in MODELS_DIR.glob("TestJeff*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        if REGISTRY_FILE.is_file():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                recs = [r for r in recs if not r.get("name", "").startswith("TestJeff")]
                with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                    json.dump(recs, f, indent=2)
            except Exception:
                pass

    def test_agent_database_crud(self):
        # 1. Add agent
        test_dir = MODELS_DIR / "TestJeff_001"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "config.pkl").write_text("dummy")

        rec = AgentDatabase.add_agent(
            name="TestJeff",
            layout="cramped_room",
            checkpoint_path=test_dir,
            agent_1="ppo",
            agent_1_pct=50,
            agent_2="greedy",
            agent_2_pct=50,
        )
        self.assertEqual(rec["name"], "TestJeff")

        # 2. Search
        found = AgentDatabase.search("TestJeff")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["name"], "TestJeff")

        # 3. Delete
        success = AgentDatabase.delete_agent(rec["id"])
        self.assertTrue(success)
        self.assertFalse(test_dir.exists())

        found_after = AgentDatabase.search("TestJeff")
        self.assertEqual(len(found_after), 0)

    def test_train_views_and_modal_confirmation(self):
        # Switch to Train Mode
        self.app.active_mode = "train"
        self.assertEqual(self.app.active_mode, "train")

        # 1. Setup view (Image 4)
        self.app.train_view = "setup"
        self.assertEqual(self.app.train_agent_name, "Jeff")
        self.assertEqual(self.app.train_agent_1_pct, 50)
        self.assertEqual(self.app.train_agent_2_pct, 50)

        # Draw frame in setup view
        self.app._draw_frame()

        # 2. Active view (Image 3)
        self.app.train_view = "active"
        self.app.train_progress_cur = 67
        self.app.train_progress_total = 100
        self.app.train_eta_str = "0hr 40min"
        self.app._draw_frame()

        # 3. Complete view (Image 2)
        self.app.train_view = "complete"
        self.app.train_last_trained_name = "Jeff"
        self.app._draw_frame()

        # 4. Database view (Image 1)
        self.app.train_view = "db"
        self.app._draw_frame()

        # Test "Are you sure?" modal popup
        test_dir = MODELS_DIR / "TestJeff_Modal"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "config.pkl").write_text("dummy")
        rec = AgentDatabase.add_agent("TestJeff_Modal", "cramped_room", test_dir)

        # Trigger confirmation modal
        self.app._confirm_delete_agent(rec["id"], "TestJeff_Modal")
        self.assertTrue(self.app.modal_confirm.is_visible)
        self.assertEqual(self.app.modal_confirm.title, "Are you Sure?")
        self.assertIn("TestJeff_Modal", self.app.modal_confirm.message)

        # Drawing frame renders modal on top
        self.app._draw_frame()

        # Click Yes on modal
        self.app.modal_confirm.btn_yes.on_click()
        self.assertFalse(self.app.modal_confirm.is_visible)
        self.assertFalse(test_dir.exists())

    def test_full_training_and_permanent_deletion_lifecycle(self):
        from training.train_agent import train
        from gui.agent_manager import create_agent, evict_trainer_cache

        test_model_dir = MODELS_DIR / "TestJeff_Bot"
        if test_model_dir.exists():
            shutil.rmtree(test_model_dir, ignore_errors=True)

        # 1. Train 1 iteration
        ckpt_path = train(
            recipe="ppo_greedy",
            layout_name="cramped_room",
            num_iters=1,
            num_workers=0,
            out_dir=test_model_dir,
        )
        self.assertTrue((test_model_dir / "metadata.json").is_file())

        # 2. Register agent in DB
        rec = AgentDatabase.add_agent(
            name="TestJeff_Bot",
            layout="cramped_room",
            checkpoint_path=test_model_dir,
            agent_1="ppo",
            agent_1_pct=50,
            agent_2="greedy",
            agent_2_pct=50,
        )
        self.assertIn("TestJeff_Bot", [r["name"] for r in AgentDatabase.search()])

        # 3. Load agent into Overcooked and step
        agent, warn = create_agent(f"custom:{test_model_dir}", 0, self.app.mdp, "cramped_room")
        self.assertIsNotNone(agent)
        state = self.app.mdp.get_standard_start_state()
        action, info = agent.action(state)
        self.assertIn(action, [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), "interact"])

        # 4. Delete agent ("gone forever")
        deleted = AgentDatabase.delete_agent(rec["id"])
        evict_trainer_cache(rec["id"])
        self.assertTrue(deleted)

        # Assert files on disk are completely gone
        self.assertFalse(test_model_dir.exists())

        # Assert removed from registry
        self.assertNotIn("TestJeff_Bot", [r["name"] for r in AgentDatabase.search()])

    def test_training_live_terminal_pause_and_stop_controls(self):
        from unittest.mock import MagicMock, patch
        import signal

        self.app.active_mode = "train"
        self.app.train_view = "active"

        # Verify buttons exist with correct attributes
        self.assertEqual(self.app.btn_view_training_live.text, "View Training Live")
        self.assertEqual(self.app.btn_pause_training.text, "Pause training")
        self.assertEqual(self.app.btn_stop_training.text, "Stop training")

        # Verify drawing while active and paused
        self.app.train_is_paused = False
        self.app._draw_frame()

        self.app.train_is_paused = True
        self.app._draw_frame()

        # Mock training process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        self.app.train_process = mock_proc
        self.app.train_is_paused = False

        # Test pause toggle -> sends SIGSTOP, updates button text to Resume training
        with patch("os.kill") as mock_kill:
            self.app._toggle_pause_training()
            mock_kill.assert_called_once_with(99999, signal.SIGSTOP)
            self.assertTrue(self.app.train_is_paused)
            self.assertEqual(self.app.btn_pause_training.text, "Resume training")

        # Test resume toggle -> sends SIGCONT, updates button text to Pause training
        with patch("os.kill") as mock_kill:
            self.app._toggle_pause_training()
            mock_kill.assert_called_once_with(99999, signal.SIGCONT)
            self.assertFalse(self.app.train_is_paused)
            self.assertEqual(self.app.btn_pause_training.text, "Pause training")

        # Test stop training -> terminates process, cleans up and returns to setup view
        self.app._stop_training()
        mock_proc.terminate.assert_called_once()
        self.assertIsNone(self.app.train_process)
        self.assertFalse(self.app.train_is_paused)
        self.assertEqual(self.app.train_view, "setup")

        # Test opening live terminal uses osascript with tail -f
        with patch("subprocess.Popen") as mock_sub:
            self.app._open_live_training_terminal()
            mock_sub.assert_called_once()
            args = mock_sub.call_args[0][0]
            self.assertEqual(args[0], "osascript")
            self.assertIn("tail -f", args[2])

    def test_all_maps_training_dropdown_and_execution(self):
        # 1. Verify train layout dropdown
        self.assertTrue(hasattr(self.app, "dropdown_train_layout"))
        options = self.app.dropdown_train_layout.options
        self.assertEqual(options[0], ("All Maps", "all_layouts"))
        self.assertEqual(self.app.train_layout, "all_layouts")

        # 2. Select layout
        self.app._on_train_layout_changed("cramped_room")
        self.assertEqual(self.app.train_layout, "cramped_room")

        self.app._on_train_layout_changed("all_layouts")
        self.assertEqual(self.app.train_layout, "all_layouts")

        # 3. Test starting training via btn_train_start.on_click()
        self.app.train_agent_name = "TestJeff_AllMaps"
        self.app.train_iters = 100
        with patch("subprocess.Popen") as mock_pop:
            mock_sub = MagicMock()
            mock_sub.poll.return_value = None
            mock_pop.return_value = mock_sub

            self.app.btn_train_start.on_click()

            mock_pop.assert_called_once()
            cmd = mock_pop.call_args[0][0]
            self.assertIn("--layout", cmd)
            self.assertIn("all_layouts", cmd)
            self.assertIn("--iters", cmd)
            self.assertIn("100", cmd)
            self.assertEqual(self.app.train_view, "active")
            self.assertEqual(self.app.train_progress_total, 500)

        # 4. Test polling progress with multi-layout status
        status_file = REPO_ROOT / "results" / "training_status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_data = {
            "status": "training",
            "recipe": "ppo_greedy",
            "layout": "all_layouts",
            "current_layout": "asymmetric_advantages",
            "layout_index": 2,
            "total_layouts": 5,
            "current_iter": 150,
            "total_iters": 500,
            "reward_mean": 12.5,
            "percent": 30.0,
            "eta_seconds": 700,
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        self.app._poll_training_progress()
        self.assertEqual(self.app.train_progress_cur, 150)
        self.assertEqual(self.app.train_progress_total, 500)
        self.assertEqual(self.app.train_reward_mean, 12.5)
        self.assertEqual(self.app.train_current_layout, "asymmetric_advantages")

        # 5. Draw active frame with all maps status
        self.app._draw_frame()

        # Clean up status file
        if status_file.exists():
            status_file.unlink()


if __name__ == "__main__":
    unittest.main()

