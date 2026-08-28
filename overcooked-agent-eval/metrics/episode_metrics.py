"""Compute episode-level performance and coordination metrics from telemetry."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from telemetry import TelemetryLogger, TelemetryRow


DIRECTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


@dataclass(frozen=True)
class EpisodeMetrics:
    """One summary row for one completed episode."""

    run_id: str
    episode_id: int
    episode_seed: int
    layout_name: str
    agent_0_id: int
    agent_1_id: int
    agent_0_name: str
    agent_1_name: str
    team_score: float
    soups_delivered: int
    episode_length: int
    agent_0_distance_traveled: int
    agent_1_distance_traveled: int
    team_distance_traveled: int
    agent_0_idle_timesteps: int
    agent_1_idle_timesteps: int
    team_idle_timesteps: int
    agent_0_idle_rate: float
    agent_1_idle_rate: float
    team_idle_rate: float
    agent_0_wall_collision_attempts: int
    agent_1_wall_collision_attempts: int
    team_wall_collision_attempts: int
    agent_0_blocked_by_teammate: int
    agent_1_blocked_by_teammate: int
    agent_0_blocking_teammate: int
    agent_1_blocking_teammate: int
    team_blocking_events: int
    agent_0_collision_attempts: int
    agent_1_collision_attempts: int
    team_collision_events: int
    same_target_collision_events: int
    swap_collision_events: int
    agent_0_interference_timesteps: int
    agent_1_interference_timesteps: int
    team_interference_timesteps: int
    agent_0_repeated_interference_timesteps: int
    agent_1_repeated_interference_timesteps: int
    team_repeated_interference_timesteps: int
    agent_0_interference_rate: float
    agent_1_interference_rate: float
    team_interference_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def load_telemetry_csv(input_path: Path | str) -> list[TelemetryRow]:
    """Load a saved telemetry CSV after validating its schema and values."""
    path = Path(input_path)
    TelemetryLogger.validate_csv(path)
    with path.open(newline="", encoding="utf-8") as input_file:
        return [TelemetryRow.from_dict(row) for row in csv.DictReader(input_file)]


def write_episode_metrics_csv(
    summaries: list[EpisodeMetrics], output_path: Path | str
) -> Path:
    """Write one metrics row per episode to a separate CSV."""
    if not summaries:
        raise ValueError("Cannot save an empty episode summary")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EpisodeMetrics.fieldnames())
        writer.writeheader()
        writer.writerows(summary.to_dict() for summary in summaries)
    return path


def summarize_episodes(rows: list[TelemetryRow]) -> list[EpisodeMetrics]:
    """Group telemetry by run and episode and compute each summary."""
    if not rows:
        raise ValueError("Cannot summarize empty telemetry")
    grouped: dict[tuple[str, int], list[TelemetryRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.run_id, row.episode_id)].append(row)
    return [
        _summarize_episode(grouped[key])
        for key in sorted(grouped, key=lambda value: (value[0], value[1]))
    ]


def _position(value: str) -> tuple[int, int]:
    coordinates = json.loads(value)
    return coordinates[0], coordinates[1]


def _target(position: tuple[int, int], action: str) -> tuple[int, int] | None:
    if action not in DIRECTION_DELTAS:
        return None
    dx, dy = DIRECTION_DELTAS[action]
    return position[0] + dx, position[1] + dy


def _distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def _event_count(value: str, event_name: str) -> int:
    return sum(event == event_name for event in value.split(";") if event)


def _validate_episode(rows: list[TelemetryRow]) -> list[TelemetryRow]:
    ordered = sorted(rows, key=lambda row: row.timestep)
    first = ordered[0]
    identity = (
        first.run_id,
        first.episode_id,
        first.episode_seed,
        first.layout_name,
        first.agent_0_id,
        first.agent_1_id,
        first.agent_0_name,
        first.agent_1_name,
    )
    for index, row in enumerate(ordered, start=1):
        row_identity = (
            row.run_id,
            row.episode_id,
            row.episode_seed,
            row.layout_name,
            row.agent_0_id,
            row.agent_1_id,
            row.agent_0_name,
            row.agent_1_name,
        )
        if row_identity != identity:
            raise ValueError("Episode telemetry contains inconsistent identifiers")
        if row.timestep != index:
            raise ValueError("Episode timesteps must be contiguous and start at 1")
        if index > 1:
            previous = ordered[index - 2]
            if (
                _position(row.agent_0_previous_position)
                != _position(previous.agent_0_position)
                or _position(row.agent_1_previous_position)
                != _position(previous.agent_1_position)
            ):
                raise ValueError("Episode positions are not continuous between timesteps")
        if row.done != (index == len(ordered)):
            raise ValueError("Only the final telemetry row may have done=True")
    return ordered


def _summarize_episode(rows: list[TelemetryRow]) -> EpisodeMetrics:
    ordered = _validate_episode(rows)
    first = ordered[0]
    episode_length = len(ordered)
    distance = [0, 0]
    idle = [0, 0]
    wall_collisions = [0, 0]
    blocked_by_teammate = [0, 0]
    blocking_teammate = [0, 0]
    collision_attempts = [0, 0]
    interference = [0, 0]
    repeated_interference = [0, 0]
    previous_interference = [False, False]
    team_blocking_events = 0
    team_collision_events = 0
    same_target_events = 0
    swap_events = 0
    team_interference_timesteps = 0
    team_repeated_interference_timesteps = 0
    soups_delivered = 0

    for row in ordered:
        previous = [
            _position(row.agent_0_previous_position),
            _position(row.agent_1_previous_position),
        ]
        current = [_position(row.agent_0_position), _position(row.agent_1_position)]
        actions = [row.agent_0_action, row.agent_1_action]
        targets = [_target(previous[0], actions[0]), _target(previous[1], actions[1])]
        directional = [targets[0] is not None, targets[1] is not None]
        failed = [
            directional[0] and current[0] == previous[0],
            directional[1] and current[1] == previous[1],
        ]

        for agent_index in (0, 1):
            distance[agent_index] += _distance(
                previous[agent_index], current[agent_index]
            )
            idle[agent_index] += int(actions[agent_index] == "stay")

        same_target = bool(
            directional[0]
            and directional[1]
            and failed[0]
            and failed[1]
            and targets[0] == targets[1]
        )
        swap = bool(
            directional[0]
            and directional[1]
            and failed[0]
            and failed[1]
            and targets[0] == previous[1]
            and targets[1] == previous[0]
        )
        joint_collision = same_target or swap
        blocked = [False, False]
        if joint_collision:
            collision_attempts[0] += 1
            collision_attempts[1] += 1
            team_collision_events += 1
            same_target_events += int(same_target)
            swap_events += int(swap)
        else:
            blocked[0] = bool(
                failed[0]
                and targets[0] == previous[1]
                and current[1] == previous[1]
            )
            blocked[1] = bool(
                failed[1]
                and targets[1] == previous[0]
                and current[0] == previous[0]
            )
            for agent_index in (0, 1):
                if blocked[agent_index]:
                    teammate_index = 1 - agent_index
                    blocked_by_teammate[agent_index] += 1
                    blocking_teammate[teammate_index] += 1
                    team_blocking_events += 1

        for agent_index in (0, 1):
            if failed[agent_index] and not joint_collision and not blocked[agent_index]:
                wall_collisions[agent_index] += 1

        interference_event = joint_collision or blocked[0] or blocked[1]
        current_interference = [interference_event, interference_event]
        any_repeated = False
        for agent_index in (0, 1):
            if current_interference[agent_index]:
                interference[agent_index] += 1
                if previous_interference[agent_index]:
                    repeated_interference[agent_index] += 1
                    any_repeated = True
        if any(current_interference):
            team_interference_timesteps += 1
        if any_repeated:
            team_repeated_interference_timesteps += 1
        previous_interference = current_interference

        soups_delivered += _event_count(row.agent_0_events, "soup_delivery")
        soups_delivered += _event_count(row.agent_1_events, "soup_delivery")

    team_idle = sum(idle)
    team_distance = sum(distance)
    team_wall_collisions = sum(wall_collisions)
    return EpisodeMetrics(
        run_id=first.run_id,
        episode_id=first.episode_id,
        episode_seed=first.episode_seed,
        layout_name=first.layout_name,
        agent_0_id=first.agent_0_id,
        agent_1_id=first.agent_1_id,
        agent_0_name=first.agent_0_name,
        agent_1_name=first.agent_1_name,
        team_score=sum(row.reward for row in ordered),
        soups_delivered=soups_delivered,
        episode_length=episode_length,
        agent_0_distance_traveled=distance[0],
        agent_1_distance_traveled=distance[1],
        team_distance_traveled=team_distance,
        agent_0_idle_timesteps=idle[0],
        agent_1_idle_timesteps=idle[1],
        team_idle_timesteps=team_idle,
        agent_0_idle_rate=idle[0] / episode_length,
        agent_1_idle_rate=idle[1] / episode_length,
        team_idle_rate=team_idle / (2 * episode_length),
        agent_0_wall_collision_attempts=wall_collisions[0],
        agent_1_wall_collision_attempts=wall_collisions[1],
        team_wall_collision_attempts=team_wall_collisions,
        agent_0_blocked_by_teammate=blocked_by_teammate[0],
        agent_1_blocked_by_teammate=blocked_by_teammate[1],
        agent_0_blocking_teammate=blocking_teammate[0],
        agent_1_blocking_teammate=blocking_teammate[1],
        team_blocking_events=team_blocking_events,
        agent_0_collision_attempts=collision_attempts[0],
        agent_1_collision_attempts=collision_attempts[1],
        team_collision_events=team_collision_events,
        same_target_collision_events=same_target_events,
        swap_collision_events=swap_events,
        agent_0_interference_timesteps=interference[0],
        agent_1_interference_timesteps=interference[1],
        team_interference_timesteps=team_interference_timesteps,
        agent_0_repeated_interference_timesteps=repeated_interference[0],
        agent_1_repeated_interference_timesteps=repeated_interference[1],
        team_repeated_interference_timesteps=team_repeated_interference_timesteps,
        agent_0_interference_rate=interference[0] / episode_length,
        agent_1_interference_rate=interference[1] / episode_length,
        team_interference_rate=team_interference_timesteps / episode_length,
    )
