"""Tests for the pretrained BC/PPO_BC versus PPO_SP rollout runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
EXTERNAL_SRC = PROJECT_ROOT.parent / "external" / "overcooked_ai" / "src"
for path in (PROJECT_ROOT, EXTERNAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.run_pretrained_policy_comparison import (
    ObservationContract,
    bundled_checkpoint,
    evaluate_condition,
    inspect_observation_contract,
    resolve_checkpoint_files,
)


class _FakeEnv:
    class _Mdp:
        @staticmethod
        def get_standard_start_state() -> object:
            return object()

    mdp = _Mdp()

    @staticmethod
    def featurize_state_mdp(state: object) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(64), np.zeros(64)

    @staticmethod
    def lossless_state_encoding_mdp(state: object) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((5, 4, 20)), np.zeros((5, 4, 20))


class _FakeEvaluator:
    env = _FakeEnv()

    @staticmethod
    def evaluate_agent_pair(pair: object, num_games: int, info: bool) -> dict[str, list[int]]:
        return {
            "ep_returns": [10 * (i + 1) for i in range(num_games)],
            "ep_lengths": [400] * num_games,
        }


class TestPretrainedPolicyComparison(unittest.TestCase):
    def test_observation_encoders_remain_distinct(self) -> None:
        contract = inspect_observation_contract(_FakeEvaluator())
        self.assertEqual(contract.bc_shape, (64,))
        self.assertEqual(contract.sp_shape, (5, 4, 20))
        contract.validate_paper_contract()

    def test_strict_contract_rejects_newer_encoder_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "64-value"):
            ObservationContract((96,), (5, 4, 26)).validate_paper_contract()

    def test_resolves_bundled_asset_layout(self) -> None:
        files = resolve_checkpoint_files(bundled_checkpoint("cramped_room", "BC"))
        self.assertEqual(files.config.name, "config.pkl")
        self.assertEqual(files.checkpoint.name, "checkpoint-550")

    def test_balanced_condition_rows(self) -> None:
        contract = ObservationContract((64,), (5, 4, 20))
        rows = evaluate_condition(
            _FakeEvaluator(),
            trainer=object(),
            policy_order=("bc", "ppo"),
            condition="BC+PPO_BC",
            episodes=2,
            seed=7,
            contract=contract,
            pair_factory=lambda trainer, first, second: (first, second),
        )
        self.assertEqual([row.sparse_return for row in rows], [10.0, 10.0])
        self.assertEqual(rows[0].seat_order, "bc_0+ppo_1")
        self.assertEqual(rows[1].seed, 8)


if __name__ == "__main__":
    unittest.main()
