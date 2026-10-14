"""Unit tests for Train Mode and Agent Database in OvercookedApp."""

import json
import os
import shutil
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


if __name__ == "__main__":
    unittest.main()
