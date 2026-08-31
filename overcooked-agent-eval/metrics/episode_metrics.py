"""Compute episode-level performance, coordination, role, and duplication metrics.

What this file does (in simple terms):
In Overcooked, two robot chefs have to cook soup together.
Just looking at their final score doesn't tell the full story!
For example:
  - Did they bump into each other in the hallway? (Collisions & blocking)
  - Did one chef prepare ingredients while the other served soup? (Roles)
  - Were both chefs trying to grab the same onion at the same time? (Task duplication)
  - Did a cooked soup sit cold in the pot while nobody dished it? (Unused pipeline)

This module reads the step-by-step game history (telemetry) and calculates
all of these interesting statistics for every single game without rerunning the game.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from telemetry import TelemetryLogger, TelemetryRow


# How positions change on a 2D grid when moving in each direction:
# (x is horizontal: right is positive east, left is negative west)
# (y is vertical: down is positive south, up is negative north)
DIRECTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


@dataclass(frozen=True)
class EpisodeMetrics:
    """One comprehensive summary row for one completed episode (game)."""

    # --- Identity and layout ---
    run_id: str
    episode_id: int
    episode_seed: int
    layout_name: str
    agent_0_id: int
    agent_1_id: int
    agent_0_name: str
    agent_1_name: str

    # --- Core performance (Issue #2) ---
    team_score: float                # Total soup points earned by the team
    soups_delivered: int             # Number of bowls of soup delivered to customers
    episode_length: int              # Total number of game steps (e.g. 400 steps)

    # --- Movement and idle time (Issue #2) ---
    agent_0_distance_traveled: int   # Grid squares walked by chef 0
    agent_1_distance_traveled: int   # Grid squares walked by chef 1
    team_distance_traveled: int      # Total grid squares walked by both chefs
    agent_0_idle_timesteps: int      # Steps where chef 0 chose to do nothing ('stay')
    agent_1_idle_timesteps: int      # Steps where chef 1 chose to do nothing ('stay')
    team_idle_timesteps: int         # Total steps where either chef chose 'stay'
    agent_0_idle_rate: float         # Fraction of time chef 0 stood still (idle / total steps)
    agent_1_idle_rate: float         # Fraction of time chef 1 stood still (idle / total steps)
    team_idle_rate: float            # Fraction of team time spent standing still

    # --- Blocking, collisions, and interference (Issue #3) ---
    agent_0_wall_collision_attempts: int  # Chef 0 bumped into a kitchen counter or wall
    agent_1_wall_collision_attempts: int  # Chef 1 bumped into a kitchen counter or wall
    team_wall_collision_attempts: int     # Total wall/counter bumps for the team
    agent_0_blocked_by_teammate: int      # Chef 0 wanted to move where chef 1 was standing
    agent_1_blocked_by_teammate: int      # Chef 1 wanted to move where chef 0 was standing
    agent_0_blocking_teammate: int        # Chef 0 was standing in chef 1's way
    agent_1_blocking_teammate: int        # Chef 1 was standing in chef 0's way
    team_blocking_events: int             # Total teammate blocking incidents
    agent_0_collision_attempts: int       # Chef 0 was in a joint collision (same square or swap)
    agent_1_collision_attempts: int       # Chef 1 was in a joint collision
    team_collision_events: int            # Total joint collision events
    same_target_collision_events: int     # Both chefs tried to step into the exact same square
    swap_collision_events: int            # Both chefs tried to swap squares directly past each other
    agent_0_interference_timesteps: int   # Steps where chef 0 had a path conflict with chef 1
    agent_1_interference_timesteps: int   # Steps where chef 1 had a path conflict with chef 0
    team_interference_timesteps: int      # Steps where the team experienced movement interference
    agent_0_repeated_interference_timesteps: int  # Interference happening two or more steps in a row
    agent_1_repeated_interference_timesteps: int
    team_repeated_interference_timesteps: int     # Persistent gridlock / hallway jams
    agent_0_interference_rate: float      # Interference steps / episode length
    agent_1_interference_rate: float
    team_interference_rate: float

    # --- Held-object role proxies (Issue #17) ---
    # How many steps each chef spent holding each kind of item
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

    # Fraction of the game spent holding each item (these always add up to 1.0 = 100%!)
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

    # --- Task-event role proxies (Issue #17) ---
    # Counts of specific game actions performed by each chef
    agent_0_potting_onion_count: int      # Onions dropped into soup pots by chef 0
    agent_1_potting_onion_count: int      # Onions dropped into soup pots by chef 1
    team_potting_onion_count: int
    agent_0_potting_tomato_count: int     # Tomatoes dropped into soup pots by chef 0
    agent_1_potting_tomato_count: int     # Tomatoes dropped into soup pots by chef 1
    team_potting_tomato_count: int
    agent_0_potting_event_count: int      # Total ingredients potted by chef 0
    agent_1_potting_event_count: int      # Total ingredients potted by chef 1
    team_potting_event_count: int
    agent_0_dish_pickup_count: int        # Clean plates grabbed by chef 0
    agent_1_dish_pickup_count: int        # Clean plates grabbed by chef 1
    team_dish_pickup_count: int
    agent_0_soup_pickup_count: int        # Bowls of hot soup scooped from pots by chef 0
    agent_1_soup_pickup_count: int        # Bowls of hot soup scooped from pots by chef 1
    team_soup_pickup_count: int
    agent_0_soup_delivery_count: int      # Soups served to customer counter by chef 0
    agent_1_soup_delivery_count: int      # Soups served to customer counter by chef 1
    team_soup_delivery_count: int
    agent_0_potting_event_rate: float     # Ingredients potted per game step
    agent_1_potting_event_rate: float
    agent_0_soup_delivery_rate: float     # Soups delivered per game step
    agent_1_soup_delivery_rate: float

    # --- Task duplication & unused pipeline work (Issue #13) ---
    duplicate_gather_timesteps: int       # Steps where both chefs were gathering ingredients
    duplicate_dish_timesteps: int         # Steps where both chefs were holding/grabbing plates
    duplicate_deliver_timesteps: int      # Steps where both chefs were holding/serving soup
    team_task_duplication_timesteps: int  # Steps where both chefs did the same job simultaneously
    team_task_duplication_rate: float     # Fraction of game spent doing redundant work
    team_unused_pipeline_timesteps: int   # Steps where soup was ready but neither chef picked it up
    team_unused_pipeline_rate: float      # Fraction of game where finished soup sat waiting

    def to_dict(self) -> dict[str, object]:
        """Convert metrics to a Python dictionary."""
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        """Return the names of all 68 metric columns in order."""
        return [field.name for field in fields(cls)]


@dataclass(frozen=True)
class AggregateMetric:
    """Statistical summary for one numeric metric across multiple games."""

    metric_name: str    # e.g. 'team_score' or 'team_blocking_events'
    sample_size: int    # Number of games in the batch (N)
    mean: float         # Average value across all games
    std: float          # Standard deviation (how much the scores varied between games)
    min: float          # Lowest value in any game
    max: float          # Highest value in any game

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchSummary:
    """Summary of all metrics for a specific kitchen layout and chef pairing."""

    layout_name: str
    agent_0_name: str
    agent_1_name: str
    episode_count: int
    metrics: dict[str, AggregateMetric]

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten into a single dictionary for exporting to CSV."""
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
    """Load a saved telemetry CSV after verifying that every row is valid."""
    path = Path(input_path)
    TelemetryLogger.validate_csv(path)
    with path.open(newline="", encoding="utf-8") as input_file:
        return [TelemetryRow.from_dict(row) for row in csv.DictReader(input_file)]


def write_episode_metrics_csv(
    summaries: list[EpisodeMetrics], output_path: Path | str
) -> Path:
    """Write one metrics summary row per episode into a CSV file."""
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
    """Group games by map and chef pairing, and calculate mean, std, min, and max."""
    if not summaries:
        return []

    # Group episodes by (layout, agent_0, agent_1)
    grouped: dict[tuple[str, str, str], list[EpisodeMetrics]] = defaultdict(list)
    for s in summaries:
        grouped[(s.layout_name, s.agent_0_name, s.agent_1_name)].append(s)

    # All columns that are numbers rather than names or IDs
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
    """Write the statistical summary (mean, std, min, max) to a CSV file."""
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
    """Group step-by-step telemetry by game, and compute the summary for each game."""
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
    """Turn a JSON string coordinate like '[1, 2]' into a Python tuple (1, 2)."""
    coordinates = json.loads(value)
    return coordinates[0], coordinates[1]


def _target(position: tuple[int, int], action: str) -> tuple[int, int] | None:
    """Figure out which square the chef was trying to walk into."""
    if action not in DIRECTION_DELTAS:
        return None
    dx, dy = DIRECTION_DELTAS[action]
    return position[0] + dx, position[1] + dy


def _distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    """Calculate Manhattan grid distance between two positions (|x1-x2| + |y1-y2|)."""
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def _event_count(value: str, event_name: str) -> int:
    """Count how many times a specific event happened in this step's event string."""
    return sum(event == event_name for event in value.split(";") if event)


def _classify_task(held_object: str, events: str, action: str) -> str:
    """Classify what task the chef appears to be doing right now.
    
    Categories:
      - 'gather': grabbing or holding raw ingredients (onions, tomatoes)
      - 'dish': grabbing or holding clean plates
      - 'deliver': holding or serving bowls of soup
      - 'idle': standing still with empty hands
      - 'other': walking around empty-handed
    """
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
    """Check that all rows for a game are in chronological order with no gaps."""
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
    """Calculate all performance, movement, collision, role, and duplication metrics for one game."""
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

    # Role proxy held-object counters
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

    # Unused pipeline state tracking: soup ready in pot or waiting on counter
    soup_in_pipeline = 0
    team_unused_pipeline_timesteps = 0

    # Loop through every timestep in the game
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

        # Process each chef's movement, idle, and item holding
        for agent_index in (0, 1):
            distance[agent_index] += _distance(
                previous[agent_index], current[agent_index]
            )
            idle[agent_index] += int(actions[agent_index] == "stay")

            # Track held object
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
                held_none[agent_index] += 1

            # Track task events
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

        # Check for joint collisions:
        # Case A: Same-target collision (both wanted to walk into the same square)
        same_target = bool(
            directional[0]
            and directional[1]
            and failed[0]
            and failed[1]
            and targets[0] == targets[1]
        )
        # Case B: Swap collision (both wanted to step through each other)
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
            # Check for teammate blocking (one chef was in the other's way)
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

        # Check for wall collisions (bumping into counter/wall, not a teammate)
        for agent_index in (0, 1):
            if failed[agent_index] and not joint_collision and not blocked[agent_index]:
                wall_collisions[agent_index] += 1

        # Check for overall interference
        interference_event = joint_collision or blocked[0] or blocked[1]
        current_interference = [interference_event, interference_event]
        any_repeated = False
        for agent_index in (0, 1):
            if current_interference[agent_index]:
                interference[agent_index] += 1
                # If interference happened on the previous step too, it's repeated!
                if previous_interference[agent_index]:
                    repeated_interference[agent_index] += 1
                    any_repeated = True
        if any(current_interference):
            team_interference_timesteps += 1
        if any_repeated:
            team_repeated_interference_timesteps += 1
        previous_interference = current_interference

        # Check for task duplication (are both doing the same job right now?)
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

        # Check for unused pipeline (soup is ready or scooped, but nobody is serving)
        step_soup_picked = sum(
            _event_count(events[i], "soup_pickup") for i in (0, 1)
        )
        step_soup_delivered = sum(
            _event_count(events[i], "soup_delivery") for i in (0, 1)
        )
        soup_in_pipeline += step_soup_picked - step_soup_delivered
        if soup_in_pipeline < 0:
            soup_in_pipeline = 0

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

    # Package all 68 fields into the EpisodeMetrics object
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
