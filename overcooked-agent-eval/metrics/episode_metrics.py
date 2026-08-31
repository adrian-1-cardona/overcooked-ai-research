"""Compute episode-level performance, coordination, role, and duplication metrics."""

from __future__ import annotations

import csv
import json
import math
import statistics
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
    """One comprehensive summary row for one completed episode."""

    # Identity and layout
    run_id: str
    episode_id: int
    episode_seed: int
    layout_name: str
    agent_0_id: int
    agent_1_id: int
    agent_0_name: str
    agent_1_name: str

    # Core performance (Issue #2)
    team_score: float
    soups_delivered: int
    episode_length: int

    # Movement and idle time (Issue #2)
    agent_0_distance_traveled: int
    agent_1_distance_traveled: int
    team_distance_traveled: int
    agent_0_idle_timesteps: int
    agent_1_idle_timesteps: int
    team_idle_timesteps: int
    agent_0_idle_rate: float
    agent_1_idle_rate: float
    team_idle_rate: float

    # Blocking, collisions, and interference (Issue #3)
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

    # Held-object role proxies (Issue #17)
    agent_0_held_none_timesteps: int
    agent_0_held_onion_timesteps: int
    agent_0_held_tomato_timesteps: int
    agent_0_held_dish_timesteps: int
    agent_0_held_soup_timesteps: int
    agent_1_held_none_timesteps: int
    agent_1_held_onion_timesteps: int
    agent_1_held_tomato_timesteps: int
    agent_1_held_dish_timesteps: int
    agent_1_held_soup_timesteps: int
    agent_0_held_none_share: float
    agent_0_held_onion_share: float
    agent_0_held_tomato_share: float
    agent_0_held_dish_share: float
    agent_0_held_soup_share: float
    agent_1_held_none_share: float
    agent_1_held_onion_share: float
    agent_1_held_tomato_share: float
    agent_1_held_dish_share: float
    agent_1_held_soup_share: float

    # Task-event role proxies (Issue #17)
    agent_0_potting_onion_count: int
    agent_1_potting_onion_count: int
    team_potting_onion_count: int
    agent_0_potting_tomato_count: int
    agent_1_potting_tomato_count: int
    team_potting_tomato_count: int
    agent_0_potting_event_count: int
    agent_1_potting_event_count: int
    team_potting_event_count: int
    agent_0_dish_pickup_count: int
    agent_1_dish_pickup_count: int
    team_dish_pickup_count: int
    agent_0_soup_pickup_count: int
    agent_1_soup_pickup_count: int
    team_soup_pickup_count: int
    agent_0_soup_delivery_count: int
    agent_1_soup_delivery_count: int
    team_soup_delivery_count: int
    agent_0_potting_event_rate: float
    agent_1_potting_event_rate: float
    agent_0_soup_delivery_rate: float
    agent_1_soup_delivery_rate: float

    # Task duplication & unused pipeline work (Issue #13)
    duplicate_gather_timesteps: int
    duplicate_dish_timesteps: int
    duplicate_deliver_timesteps: int
    team_task_duplication_timesteps: int
    team_task_duplication_rate: float
    team_unused_pipeline_timesteps: int
    team_unused_pipeline_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


@dataclass(frozen=True)
class AggregateMetric:
    """Statistical summary for one numeric metric across multiple episodes."""

    metric_name: str
    sample_size: int
    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchSummary:
    """Summary of all aggregated metrics for a specific layout and agent pairing."""

    layout_name: str
    agent_0_name: str
    agent_1_name: str
    episode_count: int
    metrics: dict[str, AggregateMetric]

    def to_flat_dict(self) -> dict[str, object]:
        flat: dict[str, object] = {
            "layout_name": self.layout_name,
            "agent_0_name": self.agent_0_name,
            "agent_1_name": self.agent_1_name,
            "episode_count": self.episode_count,
        }
        for metric_name, agg in self.metrics.items():
            flat[f"{metric_name}_mean"] = agg.mean
            flat[f"{metric_name}_std"] = agg.std
            flat[f"{metric_name}_min"] = agg.min
            flat[f"{metric_name}_max"] = agg.max
        return flat


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


def aggregate_episode_metrics(
    summaries: list[EpisodeMetrics],
) -> list[BatchSummary]:
    """Aggregate per-episode metrics into batch summaries grouped by layout and pairing."""
    if not summaries:
        return []

    grouped: dict[tuple[str, str, str], list[EpisodeMetrics]] = defaultdict(list)
    for s in summaries:
        grouped[(s.layout_name, s.agent_0_name, s.agent_1_name)].append(s)

    non_numeric_fields = {
        "run_id",
        "episode_id",
        "episode_seed",
        "layout_name",
        "agent_0_id",
        "agent_1_id",
        "agent_0_name",
        "agent_1_name",
    }
    numeric_fieldnames = [
        f.name for f in fields(EpisodeMetrics) if f.name not in non_numeric_fields
    ]

    batch_summaries: list[BatchSummary] = []
    for (layout, a0, a1), ep_list in sorted(grouped.items()):
        n = len(ep_list)
        metrics_dict: dict[str, AggregateMetric] = {}
        for fname in numeric_fieldnames:
            values = [float(getattr(ep, fname)) for ep in ep_list]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if n > 1 else 0.0
            min_val = min(values)
            max_val = max(values)
            metrics_dict[fname] = AggregateMetric(
                metric_name=fname,
                sample_size=n,
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
            )
        batch_summaries.append(
            BatchSummary(
                layout_name=layout,
                agent_0_name=a0,
                agent_1_name=a1,
                episode_count=n,
                metrics=metrics_dict,
            )
        )
    return batch_summaries


def write_aggregate_metrics_csv(
    batch_summaries: list[BatchSummary], output_path: Path | str
) -> Path:
    """Write batch aggregates to a CSV file."""
    if not batch_summaries:
        raise ValueError("Cannot save empty batch aggregates")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [b.to_flat_dict() for b in batch_summaries]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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


def _classify_task(held_object: str, events: str, action: str) -> str:
    """Classify agent observable task into gather, dish, deliver, idle, or other."""
    if (
        _event_count(events, "soup_delivery") > 0
        or _event_count(events, "soup_pickup") > 0
        or held_object == "soup"
    ):
        return "deliver"
    if (
        _event_count(events, "dish_pickup") > 0
        or _event_count(events, "useful_dish_pickup") > 0
        or held_object == "dish"
    ):
        return "dish"
    if (
        _event_count(events, "potting_onion") > 0
        or _event_count(events, "potting_tomato") > 0
        or _event_count(events, "onion_pickup") > 0
        or _event_count(events, "tomato_pickup") > 0
        or held_object in ("onion", "tomato")
    ):
        return "gather"
    if action == "stay" and held_object == "none":
        return "idle"
    return "other"


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

    # Movement and idle counters
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

    # Role proxy held object counters
    held_none = [0, 0]
    held_onion = [0, 0]
    held_tomato = [0, 0]
    held_dish = [0, 0]
    held_soup = [0, 0]

    # Role proxy event counters
    potting_onion_count = [0, 0]
    potting_tomato_count = [0, 0]
    potting_event_count = [0, 0]
    dish_pickup_count = [0, 0]
    soup_pickup_count = [0, 0]
    soup_delivery_count = [0, 0]

    # Task duplication counters
    duplicate_gather_timesteps = 0
    duplicate_dish_timesteps = 0
    duplicate_deliver_timesteps = 0
    team_task_duplication_timesteps = 0

    # Unused pipeline state tracking
    # Pipeline active when soup is cooked/picked up and awaiting delivery or dish
    soup_in_pipeline = 0
    team_unused_pipeline_timesteps = 0

    for row in ordered:
        previous = [
            _position(row.agent_0_previous_position),
            _position(row.agent_1_previous_position),
        ]
        current = [_position(row.agent_0_position), _position(row.agent_1_position)]
        actions = [row.agent_0_action, row.agent_1_action]
        held_objects = [row.agent_0_held_object, row.agent_1_held_object]
        events = [row.agent_0_events, row.agent_1_events]

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

            # Track held objects
            obj = held_objects[agent_index]
            if obj == "none":
                held_none[agent_index] += 1
            elif obj == "onion":
                held_onion[agent_index] += 1
            elif obj == "tomato":
                held_tomato[agent_index] += 1
            elif obj == "dish":
                held_dish[agent_index] += 1
            elif obj == "soup":
                held_soup[agent_index] += 1
            else:
                # Fallback for unexpected object name to none
                held_none[agent_index] += 1

            # Track events
            p_onion = _event_count(events[agent_index], "potting_onion")
            p_tomato = _event_count(events[agent_index], "potting_tomato")
            p_tot = p_onion + p_tomato
            d_pick = _event_count(events[agent_index], "dish_pickup") + _event_count(
                events[agent_index], "useful_dish_pickup"
            )
            s_pick = _event_count(events[agent_index], "soup_pickup")
            s_deliv = _event_count(events[agent_index], "soup_delivery")

            potting_onion_count[agent_index] += p_onion
            potting_tomato_count[agent_index] += p_tomato
            potting_event_count[agent_index] += p_tot
            dish_pickup_count[agent_index] += d_pick
            soup_pickup_count[agent_index] += s_pick
            soup_delivery_count[agent_index] += s_deliv

        # Joint collisions & blocking
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

        # Task classification & duplication
        task_0 = _classify_task(held_objects[0], events[0], actions[0])
        task_1 = _classify_task(held_objects[1], events[1], actions[1])

        if task_0 == task_1 and task_0 not in ("idle", "other"):
            team_task_duplication_timesteps += 1
            if task_0 == "gather":
                duplicate_gather_timesteps += 1
            elif task_0 == "dish":
                duplicate_dish_timesteps += 1
            elif task_0 == "deliver":
                duplicate_deliver_timesteps += 1

        # Unused pipeline tracking:
        # If soup pickup occurred or an agent holds soup or potted onion >= 3
        # but neither agent is actively working on dish/deliver
        step_soup_picked = sum(
            _event_count(events[i], "soup_pickup") for i in (0, 1)
        )
        step_soup_delivered = sum(
            _event_count(events[i], "soup_delivery") for i in (0, 1)
        )
        soup_in_pipeline += step_soup_picked - step_soup_delivered
        if soup_in_pipeline < 0:
            soup_in_pipeline = 0

        # Pipeline is waiting if soup is in pipeline or anyone holds soup/dish,
        # but neither agent is actively delivering or advancing soup
        if soup_in_pipeline > 0 or held_objects[0] == "soup" or held_objects[1] == "soup":
            is_servicing_pipeline = (
                task_0 in ("dish", "deliver") or task_1 in ("dish", "deliver")
            )
            if not is_servicing_pipeline:
                team_unused_pipeline_timesteps += 1

    team_idle = sum(idle)
    team_distance = sum(distance)
    team_wall_collisions = sum(wall_collisions)
    soups_delivered = sum(soup_delivery_count)

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
        agent_0_held_none_timesteps=held_none[0],
        agent_0_held_onion_timesteps=held_onion[0],
        agent_0_held_tomato_timesteps=held_tomato[0],
        agent_0_held_dish_timesteps=held_dish[0],
        agent_0_held_soup_timesteps=held_soup[0],
        agent_1_held_none_timesteps=held_none[1],
        agent_1_held_onion_timesteps=held_onion[1],
        agent_1_held_tomato_timesteps=held_tomato[1],
        agent_1_held_dish_timesteps=held_dish[1],
        agent_1_held_soup_timesteps=held_soup[1],
        agent_0_held_none_share=held_none[0] / episode_length,
        agent_0_held_onion_share=held_onion[0] / episode_length,
        agent_0_held_tomato_share=held_tomato[0] / episode_length,
        agent_0_held_dish_share=held_dish[0] / episode_length,
        agent_0_held_soup_share=held_soup[0] / episode_length,
        agent_1_held_none_share=held_none[1] / episode_length,
        agent_1_held_onion_share=held_onion[1] / episode_length,
        agent_1_held_tomato_share=held_tomato[1] / episode_length,
        agent_1_held_dish_share=held_dish[1] / episode_length,
        agent_1_held_soup_share=held_soup[1] / episode_length,
        agent_0_potting_onion_count=potting_onion_count[0],
        agent_1_potting_onion_count=potting_onion_count[1],
        team_potting_onion_count=sum(potting_onion_count),
        agent_0_potting_tomato_count=potting_tomato_count[0],
        agent_1_potting_tomato_count=potting_tomato_count[1],
        team_potting_tomato_count=sum(potting_tomato_count),
        agent_0_potting_event_count=potting_event_count[0],
        agent_1_potting_event_count=potting_event_count[1],
        team_potting_event_count=sum(potting_event_count),
        agent_0_dish_pickup_count=dish_pickup_count[0],
        agent_1_dish_pickup_count=dish_pickup_count[1],
        team_dish_pickup_count=sum(dish_pickup_count),
        agent_0_soup_pickup_count=soup_pickup_count[0],
        agent_1_soup_pickup_count=soup_pickup_count[1],
        team_soup_pickup_count=sum(soup_pickup_count),
        agent_0_soup_delivery_count=soup_delivery_count[0],
        agent_1_soup_delivery_count=soup_delivery_count[1],
        team_soup_delivery_count=soups_delivered,
        agent_0_potting_event_rate=potting_event_count[0] / episode_length,
        agent_1_potting_event_rate=potting_event_count[1] / episode_length,
        agent_0_soup_delivery_rate=soup_delivery_count[0] / episode_length,
        agent_1_soup_delivery_rate=soup_delivery_count[1] / episode_length,
        duplicate_gather_timesteps=duplicate_gather_timesteps,
        duplicate_dish_timesteps=duplicate_dish_timesteps,
        duplicate_deliver_timesteps=duplicate_deliver_timesteps,
        team_task_duplication_timesteps=team_task_duplication_timesteps,
        team_task_duplication_rate=team_task_duplication_timesteps / episode_length,
        team_unused_pipeline_timesteps=team_unused_pipeline_timesteps,
        team_unused_pipeline_rate=team_unused_pipeline_timesteps / episode_length,
    )
