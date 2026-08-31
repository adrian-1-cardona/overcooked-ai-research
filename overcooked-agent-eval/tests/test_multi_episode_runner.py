"""Tests for multi-episode configuration, multi-layout execution, and reproducibility."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from experiments.run_multi_episode_baseline import (
    deterministic_run_id,
    events_at_timestep,
    execute_batch,
    parse_args,
    run_experiment,
    validate_layouts,
)
from telemetry import load_batch_manifest


class MultiEpisodeRunnerTests(unittest.TestCase):
    def test_argument_defaults_and_layout_specific_output(self) -> None:
        args = parse_args([])
        self.assertEqual(args.episodes, 5)
        self.assertEqual(args.layouts, ["cramped_room"])
        self.assertEqual(args.base_seed, 42)

    def test_multiple_layouts_argument_parsing(self) -> None:
        args = parse_args([
            "--layouts",
            "cramped_room",
            "asymmetric_advantages",
            "coordination_ring",
        ])
        self.assertEqual(
            args.layouts,
            ["cramped_room", "asymmetric_advantages", "coordination_ring"],
        )

    def test_comma_separated_layouts_parsing(self) -> None:
        args = parse_args(["--layout", "cramped_room,asymmetric_advantages"])
        self.assertEqual(
            args.layouts,
            ["cramped_room", "asymmetric_advantages"],
        )

    def test_invalid_arguments_exit_with_clear_error(self) -> None:
        for arguments in (
            ["--episodes", "0"],
            ["--horizon", "0"],
            ["--base-seed", "-1"],
        ):
            with self.subTest(arguments=arguments):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments)

    def test_layout_validation_rejects_unknown_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid layout"):
            validate_layouts(["cramped_room", "non_existent_layout_xyz"])

    def test_seed_alias_remains_available(self) -> None:
        args = parse_args(["--seed", "100"])
        self.assertEqual(args.base_seed, 100)

    def test_episode_seeds_and_results_are_reproducible(self) -> None:
        first, _, _ = run_experiment("cramped_room", episodes=2, horizon=6, seed=20)
        second, _, _ = run_experiment("cramped_room", episodes=2, horizon=6, seed=20)

        self.assertEqual(
            [row.to_dict() for row in first.rows],
            [row.to_dict() for row in second.rows],
        )
        self.assertEqual({row.episode_seed for row in first.rows}, {20, 21})
        self.assertEqual(
            {row.run_id for row in first.rows},
            {deterministic_run_id("cramped_room", 2, 6, 20)},
        )

    def test_ten_episode_batch_has_unique_episode_ids_and_seeds(self) -> None:
        logger, rewards, lengths = run_experiment(
            "cramped_room", episodes=10, horizon=1, seed=42
        )

        self.assertEqual([row.episode_id for row in logger.rows], list(range(1, 11)))
        self.assertEqual([row.episode_seed for row in logger.rows], list(range(42, 52)))
        self.assertEqual(len(rewards), 10)
        self.assertEqual(lengths, [1] * 10)

    def test_run_id_changes_with_configuration(self) -> None:
        self.assertNotEqual(
            deterministic_run_id("cramped_room", 5, 400, 42),
            deterministic_run_id("cramped_room", 5, 400, 43),
        )

    def test_events_are_attributed_to_the_correct_agent_and_timestep(self) -> None:
        game_stats = {
            "onion_pickup": [[0, 2], [1]],
            "soup_delivery": [[], [2]],
            "cumulative_sparse_rewards_by_agent": [0, 0],
        }

        self.assertEqual(events_at_timestep(game_stats, 2, 0), "onion_pickup")
        self.assertEqual(events_at_timestep(game_stats, 2, 1), "soup_delivery")

    def test_execute_batch_creates_csv_and_manifest_across_multiple_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            custom_out = temp_path / "batch.csv"
            layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]

            results = execute_batch(
                layouts=layouts,
                episodes=2,
                horizon=5,
                base_seed=42,
                custom_output=custom_out,
            )

            self.assertEqual(len(results), 3)
            for csv_path, manifest_path in results:
                self.assertTrue(csv_path.exists())
                self.assertTrue(manifest_path.exists())

                manifest = load_batch_manifest(manifest_path)
                self.assertEqual(manifest.episodes, 2)
                self.assertEqual(manifest.horizon, 5)
                self.assertIn(manifest.layout, layouts)


if __name__ == "__main__":
    unittest.main()
