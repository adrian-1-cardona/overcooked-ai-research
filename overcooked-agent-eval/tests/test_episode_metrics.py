"""Hand-checked trajectory fixtures for episode and coordination metrics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from metrics import summarize_episodes, write_episode_metrics_csv
from telemetry import TelemetryRow


def telemetry_row(
    *,
    timestep: int = 1,
    previous_0: tuple[int, int] = (1, 1),
    previous_1: tuple[int, int] = (3, 1),
    position_0: tuple[int, int] = (1, 1),
    position_1: tuple[int, int] = (3, 1),
    action_0: str = "stay",
    action_1: str = "stay",
    reward: float = 0,
    events_0: str = "",
    events_1: str = "",
    done: bool = True,
) -> TelemetryRow:
    return TelemetryRow(
        run_id="fixture-run",
        episode_id=1,
        episode_seed=42,
        timestep=timestep,
        layout_name="fixture_layout",
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
        agent_0_held_object="none",
        agent_1_held_object="none",
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


if __name__ == "__main__":
    unittest.main()
