"""Unit tests for the Overcooked-AI run dashboard plotting module."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
AGENT_EVAL_DIR = TEST_DIR.parent
EXPERIMENTS_DIR = AGENT_EVAL_DIR / "experiments"
EXTERNAL_OVERCOOKED = AGENT_EVAL_DIR.parent / "external" / "overcooked_ai" / "src"

if str(EXTERNAL_OVERCOOKED) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_OVERCOOKED))
if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiments.plot_run import (
    parse_position,
    parse_telemetry_csv,
    plot_run_dashboard,
)


class TestPlotRun(unittest.TestCase):
    def test_parse_position(self) -> None:
        self.assertEqual(parse_position((2, 3)), (2, 3))
        self.assertEqual(parse_position([1, 4]), (1, 4))
        self.assertEqual(parse_position("(2, 1)"), (2, 1))
        self.assertEqual(parse_position(" [0, 5] "), (0, 5))
        self.assertIsNone(parse_position(None))
        self.assertIsNone(parse_position("invalid"))

    def test_parse_telemetry_csv(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "timestep",
                    "agent_0_action",
                    "agent_1_action",
                    "sparse_reward",
                    "agent_0_shaped_reward",
                    "agent_1_shaped_reward",
                    "cumulative_sparse_reward",
                    "agent_0_position",
                    "agent_1_position",
                    "done",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "episode": 1,
                "timestep": 1,
                "agent_0_action": "north",
                "agent_1_action": "interact",
                "sparse_reward": 0.0,
                "agent_0_shaped_reward": 3.0,
                "agent_1_shaped_reward": 0.0,
                "cumulative_sparse_reward": 0.0,
                "agent_0_position": "(1, 1)",
                "agent_1_position": "(2, 1)",
                "done": False,
            })

        try:
            rows = parse_telemetry_csv(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["timestep"], 1)
            self.assertEqual(rows[0]["agent_0_action"], "north")
            self.assertEqual(rows[0]["agent_0_shaped_reward"], 3.0)
            self.assertEqual(rows[0]["agent_0_position"], (1, 1))
            self.assertEqual(rows[0]["agent_1_position"], (2, 1))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_plot_run_dashboard(self) -> None:
        rows = [
            {
                "timestep": 1,
                "sparse_reward": 0.0,
                "cumulative_sparse_reward": 0.0,
                "agent_0_shaped_reward": 0.0,
                "agent_1_shaped_reward": 0.0,
                "agent_0_action": "north",
                "agent_1_action": "stay",
                "agent_0_position": (1, 1),
                "agent_1_position": (2, 1),
                "layout_name": "cramped_room",
            },
            {
                "timestep": 2,
                "sparse_reward": 20.0,
                "cumulative_sparse_reward": 20.0,
                "agent_0_shaped_reward": 3.0,
                "agent_1_shaped_reward": 1.0,
                "agent_0_action": "interact",
                "agent_1_action": "south",
                "agent_0_position": (1, 2),
                "agent_1_position": (2, 2),
                "layout_name": "cramped_room",
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_img = Path(f.name)

        try:
            saved = plot_run_dashboard(
                rows=rows,
                layout_name="cramped_room",
                output_path=out_img,
                show=False,
            )
            self.assertIsNotNone(saved)
            self.assertTrue(out_img.exists())
            self.assertGreater(out_img.stat().st_size, 1000)
        finally:
            out_img.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
