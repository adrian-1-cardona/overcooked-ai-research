"""Hand-checked trajectory fixtures for episode, coordination, role, duplication, and aggregate metrics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from metrics import (
    aggregate_episode_metrics,
    summarize_episodes,
    write_aggregate_metrics_csv,
    write_episode_metrics_csv,
)
from telemetry import TelemetryRow


def telemetry_row(
    *,
    episode_id: int = 1,
    episode_seed: int = 42,
    timestep: int = 1,
    layout_name: str = "fixture_layout",
    previous_0: tuple[int, int] = (1, 1),
    previous_1: tuple[int, int] = (3, 1),
    position_0: tuple[int, int] = (1, 1),
    position_1: tuple[int, int] = (3, 1),
    action_0: str = "stay",
    action_1: str = "stay",
    held_object_0: str = "none",
    held_object_1: str = "none",
    reward: float = 0,
    events_0: str = "",
    events_1: str = "",
    done: bool = True,
) -> TelemetryRow:
    return TelemetryRow(
        run_id="fixture-run",
        episode_id=episode_id,
        episode_seed=episode_seed,
        timestep=timestep,
        layout_name=layout_name,
        agent_0_id=0,
        agent_1_id=1,
        agent_0_name="FixtureAgent0",
        agent_1_name="FixtureAgent1",
        agent_0_action=action_0,
        agent_1_action=action_1,
        reward=reward,
        agent_0_sparse_reward=reward,
        agent_1_sparse_reward=0,
        agent_0_shaped_reward=0,
        agent_1_shaped_reward=0,
        done=done,
        agent_0_previous_position=json.dumps(previous_0),
        agent_1_previous_position=json.dumps(previous_1),
        agent_0_position=json.dumps(position_0),
        agent_1_position=json.dumps(position_1),
        agent_0_orientation="north",
        agent_1_orientation="south",
        agent_0_held_object=held_object_0,
        agent_1_held_object=held_object_1,
        agent_0_events=events_0,
        agent_1_events=events_1,
    )


class EpisodePerformanceMetricTests(unittest.TestCase):
    def test_performance_fixture_has_hand_checked_values(self) -> None:
        rows = [
            telemetry_row(
                timestep=1,
                previous_0=(1, 1),
                position_0=(2, 1),
                action_0="east",
                previous_1=(3, 2),
                position_1=(3, 2),
                action_1="stay",
                reward=20,
                events_0="soup_delivery",
                done=False,
            ),
            telemetry_row(
                timestep=2,
                previous_0=(2, 1),
                position_0=(2, 1),
                action_0="stay",
                previous_1=(3, 2),
                position_1=(3, 1),
                action_1="north",
            ),
        ]

        summary = summarize_episodes(rows)[0]

        self.assertEqual(summary.team_score, 20)
        self.assertEqual(summary.soups_delivered, 1)
        self.assertEqual(summary.episode_length, 2)
        self.assertEqual(summary.agent_0_distance_traveled, 1)
        self.assertEqual(summary.agent_1_distance_traveled, 1)
        self.assertEqual(summary.team_distance_traveled, 2)
        self.assertEqual(summary.agent_0_idle_timesteps, 1)
        self.assertEqual(summary.agent_1_idle_timesteps, 1)
        self.assertEqual(summary.team_idle_rate, 0.5)

    def test_summary_csv_preserves_run_and_episode_join_keys(self) -> None:
        summary = summarize_episodes([telemetry_row()])[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "episode_metrics.csv"
            write_episode_metrics_csv([summary], output)
            with output.open(newline="", encoding="utf-8") as csv_file:
                saved = list(csv.DictReader(csv_file))[0]

        self.assertEqual(saved["run_id"], "fixture-run")
        self.assertEqual(saved["episode_id"], "1")


class CoordinationMetricTests(unittest.TestCase):
    def test_teammate_block_is_not_counted_as_wall_collision(self) -> None:
        summary = summarize_episodes(
            [
                telemetry_row(
                    previous_0=(1, 1),
                    position_0=(1, 1),
                    action_0="east",
                    previous_1=(2, 1),
                    position_1=(2, 1),
                    action_1="stay",
                )
            ]
        )[0]

        self.assertEqual(summary.agent_0_blocked_by_teammate, 1)
        self.assertEqual(summary.agent_1_blocking_teammate, 1)
        self.assertEqual(summary.team_blocking_events, 1)
        self.assertEqual(summary.team_wall_collision_attempts, 0)

    def test_wall_collision_is_not_counted_as_teammate_block(self) -> None:
        summary = summarize_episodes(
            [
                telemetry_row(
                    previous_0=(1, 1),
                    position_0=(1, 1),
                    action_0="west",
                    previous_1=(3, 1),
                    position_1=(3, 1),
                    action_1="stay",
                )
            ]
        )[0]

        self.assertEqual(summary.agent_0_wall_collision_attempts, 1)
        self.assertEqual(summary.team_blocking_events, 0)
        self.assertEqual(summary.team_interference_timesteps, 0)

    def test_same_target_collision_counts_both_agents_once(self) -> None:
        summary = summarize_episodes(
            [
                telemetry_row(
                    previous_0=(1, 1),
                    position_0=(1, 1),
                    action_0="east",
                    previous_1=(3, 1),
                    position_1=(3, 1),
                    action_1="west",
                )
            ]
        )[0]

        self.assertEqual(summary.same_target_collision_events, 1)
        self.assertEqual(summary.agent_0_collision_attempts, 1)
        self.assertEqual(summary.agent_1_collision_attempts, 1)
        self.assertEqual(summary.team_collision_events, 1)

    def test_swap_attempt_is_a_distinct_collision(self) -> None:
        summary = summarize_episodes(
            [
                telemetry_row(
                    previous_0=(1, 1),
                    position_0=(1, 1),
                    action_0="east",
                    previous_1=(2, 1),
                    position_1=(2, 1),
                    action_1="west",
                )
            ]
        )[0]

        self.assertEqual(summary.swap_collision_events, 1)
        self.assertEqual(summary.team_collision_events, 1)
        self.assertEqual(summary.team_blocking_events, 0)

    def test_idle_action_is_separate_from_failed_movement(self) -> None:
        summary = summarize_episodes(
            [
                telemetry_row(
                    previous_0=(1, 1),
                    position_0=(1, 1),
                    action_0="stay",
                    previous_1=(3, 1),
                    position_1=(3, 1),
                    action_1="north",
                )
            ]
        )[0]

        self.assertEqual(summary.agent_0_idle_timesteps, 1)
        self.assertEqual(summary.agent_1_idle_timesteps, 0)
        self.assertEqual(summary.agent_1_wall_collision_attempts, 1)

    def test_consecutive_interference_counts_as_repeated(self) -> None:
        rows = [
            telemetry_row(
                timestep=1,
                previous_0=(1, 1),
                position_0=(1, 1),
                action_0="east",
                previous_1=(2, 1),
                position_1=(2, 1),
                action_1="stay",
                done=False,
            ),
            telemetry_row(
                timestep=2,
                previous_0=(1, 1),
                position_0=(1, 1),
                action_0="east",
                previous_1=(2, 1),
                position_1=(2, 1),
                action_1="stay",
            ),
        ]

        summary = summarize_episodes(rows)[0]

        self.assertEqual(summary.team_interference_timesteps, 2)
        self.assertEqual(summary.agent_0_repeated_interference_timesteps, 1)
        self.assertEqual(summary.agent_1_repeated_interference_timesteps, 1)
        self.assertEqual(summary.team_repeated_interference_timesteps, 1)


class RoleProxyAndDuplicationMetricTests(unittest.TestCase):
    def test_held_object_time_shares_and_role_proxies(self) -> None:
        # Agent 0 acts as prep (holds onion, pots onion)
        # Agent 1 acts as deliverer (holds dish, holds soup, delivers)
        rows = [
            telemetry_row(
                timestep=1,
                held_object_0="onion",
                held_object_1="dish",
                events_0="onion_pickup",
                events_1="dish_pickup",
                done=False,
            ),
            telemetry_row(
                timestep=2,
                held_object_0="none",
                held_object_1="soup",
                events_0="potting_onion",
                events_1="soup_pickup",
                done=False,
            ),
            telemetry_row(
                timestep=3,
                held_object_0="none",
                held_object_1="none",
                events_0="",
                events_1="soup_delivery",
                reward=20,
                done=True,
            ),
        ]

        summary = summarize_episodes(rows)[0]

        # Check held object counts & shares
        self.assertEqual(summary.agent_0_held_onion_timesteps, 1)
        self.assertEqual(summary.agent_0_held_none_timesteps, 2)
        self.assertAlmostEqual(summary.agent_0_held_onion_share, 1 / 3)
        self.assertAlmostEqual(summary.agent_0_held_none_share, 2 / 3)

        # Sum of shares must equal 1.0
        total_share_0 = (
            summary.agent_0_held_none_share
            + summary.agent_0_held_onion_share
            + summary.agent_0_held_tomato_share
            + summary.agent_0_held_dish_share
            + summary.agent_0_held_soup_share
        )
        self.assertAlmostEqual(total_share_0, 1.0)

        self.assertEqual(summary.agent_1_held_dish_timesteps, 1)
        self.assertEqual(summary.agent_1_held_soup_timesteps, 1)
        self.assertEqual(summary.agent_1_held_none_timesteps, 1)
        total_share_1 = (
            summary.agent_1_held_none_share
            + summary.agent_1_held_onion_share
            + summary.agent_1_held_tomato_share
            + summary.agent_1_held_dish_share
            + summary.agent_1_held_soup_share
        )
        self.assertAlmostEqual(total_share_1, 1.0)

        # Check event counts & rates
        self.assertEqual(summary.agent_0_potting_onion_count, 1)
        self.assertEqual(summary.agent_0_potting_event_count, 1)
        self.assertEqual(summary.agent_1_potting_event_count, 0)
        self.assertEqual(summary.team_potting_event_count, 1)
        self.assertEqual(summary.agent_1_dish_pickup_count, 1)
        self.assertEqual(summary.agent_1_soup_pickup_count, 1)
        self.assertEqual(summary.agent_1_soup_delivery_count, 1)
        self.assertEqual(summary.agent_0_soup_delivery_count, 0)
        self.assertEqual(summary.soups_delivered, 1)
        self.assertAlmostEqual(summary.agent_1_soup_delivery_rate, 1 / 3)

    def test_task_duplication_detected_when_both_gather(self) -> None:
        rows = [
            telemetry_row(
                timestep=1,
                held_object_0="onion",
                held_object_1="onion",
                done=False,
            ),
            telemetry_row(
                timestep=2,
                held_object_0="onion",
                held_object_1="dish",
                done=True,
            ),
        ]

        summary = summarize_episodes(rows)[0]
        self.assertEqual(summary.duplicate_gather_timesteps, 1)
        self.assertEqual(summary.team_task_duplication_timesteps, 1)
        self.assertAlmostEqual(summary.team_task_duplication_rate, 0.5)

    def test_unused_pipeline_detected_when_soup_waiting(self) -> None:
        rows = [
            telemetry_row(
                timestep=1,
                held_object_0="soup",
                held_object_1="none",
                action_0="stay",
                action_1="stay",
                events_0="soup_pickup",
                done=False,
            ),
            telemetry_row(
                timestep=2,
                held_object_0="none",
                held_object_1="none",
                action_0="north",
                action_1="stay",
                events_0="",
                done=True,
            ),
        ]

        summary = summarize_episodes(rows)[0]
        # In step 2, soup was picked up in step 1 so pipeline has soup, but in step 2 neither agent is dishing/delivering
        self.assertGreaterEqual(summary.team_unused_pipeline_timesteps, 1)
        self.assertGreater(summary.team_unused_pipeline_rate, 0.0)


class BatchAggregationTests(unittest.TestCase):
    def test_batch_aggregates_compute_correct_statistics(self) -> None:
        # Create two distinct episode summaries with known scores (10 and 20)
        row1 = telemetry_row(episode_id=1, reward=10, done=True)
        row2 = telemetry_row(episode_id=2, reward=20, done=True)

        summaries = summarize_episodes([row1, row2])
        aggregates = aggregate_episode_metrics(summaries)

        self.assertEqual(len(aggregates), 1)
        batch = aggregates[0]
        self.assertEqual(batch.episode_count, 2)
        self.assertEqual(batch.layout_name, "fixture_layout")

        score_agg = batch.metrics["team_score"]
        self.assertEqual(score_agg.sample_size, 2)
        self.assertAlmostEqual(score_agg.mean, 15.0)
        self.assertAlmostEqual(score_agg.min, 10.0)
        self.assertAlmostEqual(score_agg.max, 20.0)
        self.assertAlmostEqual(score_agg.std, 7.0710678, places=4)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_csv = Path(temp_dir) / "aggregates.csv"
            write_aggregate_metrics_csv(aggregates, out_csv)
            self.assertTrue(out_csv.exists())
            with out_csv.open(newline="", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 1)
                self.assertEqual(reader[0]["layout_name"], "fixture_layout")
                self.assertEqual(reader[0]["episode_count"], "2")
                self.assertAlmostEqual(float(reader[0]["team_score_mean"]), 15.0)


if __name__ == "__main__":
    unittest.main()
