"""Unit tests for accurate timer, progress bar, and ETA countdown calculations."""

import json
import os
import sys
import time
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
from training.train_agent import STATUS_FILE


class TestTimerAndProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        if STATUS_FILE.is_file():
            try:
                STATUS_FILE.unlink()
            except Exception:
                pass
        self.app = OvercookedApp(initial_layout="cramped_room", horizon=50, fps=30)

    def tearDown(self):
        if STATUS_FILE.is_file():
            try:
                STATUS_FILE.unlink()
            except Exception:
                pass

    def test_initial_timer_estimate_is_accurate_prior(self):
        # Default initialization: 1000s -> 16min 40s
        self.assertEqual(self.app.train_eta_str, "16min 40s")
        self.assertEqual(self.app.train_eta_seconds, 1000)

    def test_poll_updates_progress_and_eta_from_status_file(self):
        self.app.train_view = "active"

        # Simulate 10 iterations done, 90 remaining at 20s/iter = 1800s (30 mins)
        status_data = {
            "status": "training",
            "recipe": "ppo_greedy",
            "layout": "cramped_room",
            "current_iter": 10,
            "total_iters": 100,
            "reward_mean": 4.5,
            "percent": 10.0,
            "eta_seconds": 1800,
        }
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        self.app._poll_training_progress()

        self.assertEqual(self.app.train_progress_cur, 10)
        self.assertEqual(self.app.train_progress_total, 100)
        self.assertEqual(self.app.train_eta_seconds, 1800)
        self.assertEqual(self.app.train_eta_str, "30min 00s")

    def test_smooth_live_countdown_between_polls(self):
        from unittest.mock import patch

        self.app.train_view = "active"
        self.app.train_eta_seconds = 600  # 10 minutes
        # Simulate last poll exactly 65 seconds ago
        fixed_now = 1000.0
        self.app.train_last_poll_time = fixed_now - 65.0

        # Empty status file or no new update
        if STATUS_FILE.is_file():
            STATUS_FILE.unlink()

        with patch("time.time", return_value=fixed_now):
            self.app._poll_training_progress()

        # 600 - 65 = 535 seconds = 8 minutes 55 seconds remaining
        self.assertEqual(self.app.train_eta_str, "8min 55s")

    def test_completion_sets_eta_to_zero(self):
        self.app.train_view = "active"
        status_data = {
            "status": "completed",
            "recipe": "ppo_greedy",
            "layout": "cramped_room",
            "current_iter": 100,
            "total_iters": 100,
            "reward_mean": 12.0,
            "percent": 100.0,
            "eta_seconds": 0,
        }
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        self.app._poll_training_progress()

        self.assertEqual(self.app.train_view, "complete")
        self.assertEqual(self.app.train_eta_str, "0s")
        self.assertEqual(self.app.train_eta_seconds, 0)


if __name__ == "__main__":
    unittest.main()
