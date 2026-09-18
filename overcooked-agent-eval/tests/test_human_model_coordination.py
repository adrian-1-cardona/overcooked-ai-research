"""Tests for the human-proxy coordination diagnostics."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
EXTERNAL_SRC = PROJECT_ROOT.parent / "external" / "overcooked_ai" / "src"
for path in (PROJECT_ROOT, EXTERNAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.analyze_human_model_coordination import (
    EpisodeRecord,
    compute_metrics,
    condition_summary,
)
from overcooked_ai_py.mdp.actions import Action, Direction


@dataclass
class _Player:
    position: tuple[int, int]


class _State:
    def __init__(self, first: tuple[int, int], second: tuple[int, int]) -> None:
        self.players = (_Player(first), _Player(second))


class TestHumanModelCoordination(unittest.TestCase):
    def test_metrics_detect_stalls_and_blocked_moves(self) -> None:
        states = [
            _State((0, 0), (1, 0)),
            _State((0, 0), (1, 0)),
            _State((0, 0), (1, 1)),
            _State((0, 0), (1, 1)),
            _State((0, 0), (1, 1)),
        ]
        actions = [
            (Direction.EAST, Action.STAY),
            (Direction.EAST, Direction.SOUTH),
            (Action.STAY, Action.INTERACT),
            (Action.STAY, Action.STAY),
        ]
        metrics = compute_metrics(
            "BC+PPO_SP", "bc_0+ppo_sp_1", 1, 42, states, actions, [0, 20, 0, 0]
        )
        self.assertEqual(metrics.sparse_return, 20)
        self.assertEqual(metrics.deliveries, 1)
        self.assertAlmostEqual(metrics.joint_stationary_rate, 0.75)
        self.assertAlmostEqual(metrics.blocked_move_rate, 2 / 3)
        self.assertEqual(metrics.longest_joint_stationary_streak, 2)
        self.assertEqual(metrics.max_delivery_drought, 2)
        self.assertEqual(metrics.unique_joint_positions, 2)
        self.assertAlmostEqual(metrics.repeated_joint_position_rate, 0.5)

    def test_summary_aggregates_conditions(self) -> None:
        states = [_State((0, 0), (1, 0)), _State((0, 0), (1, 0))]
        metrics = compute_metrics(
            "BC+PPO_BC", "bc_0+ppo_bc_1", 1, 3, states, [(Action.STAY, Action.STAY)], [20]
        )
        record = EpisodeRecord(metrics, states, [(Action.STAY, Action.STAY)], [20], [{}])
        summary = condition_summary([record])
        self.assertEqual(summary["BC+PPO_BC"]["sparse_return"], 20)
        self.assertEqual(summary["BC+PPO_BC"]["joint_stationary_rate"], 1)


if __name__ == "__main__":
    unittest.main()
