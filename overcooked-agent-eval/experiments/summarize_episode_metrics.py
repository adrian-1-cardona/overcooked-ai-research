"""Summarize performance, coordination, role proxies, and duplication metrics from result CSVs.

Supports:
  1. Multi-episode timestep telemetry (e.g. results/multi_episode_random_baseline_cramped_room.csv)
     - Computes episode-level metrics CSV (*_episode_metrics.csv)
     - Computes and prints batch aggregate statistics (N, mean, std, min, max)
  2. Episode-level metrics CSV (e.g. results/*_episode_metrics.csv)
     - Computes and prints batch aggregate statistics across episodes
  3. Single-episode legacy CSV (e.g. results/random_baseline_cramped_room.csv)
     - Prints single episode score and step metrics
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import (
    BatchSummary,
    EpisodeMetrics,
    aggregate_episode_metrics,
    load_telemetry_csv,
    summarize_episodes,
    write_aggregate_metrics_csv,
    write_episode_metrics_csv,
)

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_INPUT = (
    DEFAULT_RESULTS_DIR / "multi_episode_random_baseline_cramped_room.csv"
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float, decimals: int = 3) -> str:
    """Format a float, suppressing unnecessary trailing zeros."""
    if decimals == 0:
        return str(int(round(value)))
    formatted = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _print_section(title: str) -> None:
    print(f"\n  {title}")
    print("  " + "-" * len(title))


def _stat_row(label: str, values: list[float], decimals: int = 3) -> None:
    if not values:
        print(f"    {label:<42}  n/a")
        return
    n = len(values)
    mean = statistics.mean(values)
    if n == 1:
        print(f"    {label:<42}  {_fmt(mean, decimals)}")
    else:
        stdev = statistics.stdev(values)
        mn, mx = min(values), max(values)
        print(
            f"    {label:<42}  N={n}  mean={_fmt(mean, decimals)}"
            f"  std={_fmt(stdev, decimals)}"
            f"  min={_fmt(mn, decimals)}"
            f"  max={_fmt(mx, decimals)}"
        )


def _col(rows: list[dict], key: str) -> list[float]:
    """Extract a numeric column, silently skipping missing / non-numeric values."""
    out: list[float] = []
    for r in rows:
        v = r.get(key, "")
        if v != "":
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _is_episode_metrics(fieldnames: list[str]) -> bool:
    """Return True if the CSV contains episode-level metrics."""
    return "team_score" in fieldnames and "episode_length" in fieldnames


def _is_multi_episode_timestep(fieldnames: list[str]) -> bool:
    """Return True if the CSV is a multi-episode timestep telemetry file."""
    return "episode_id" in fieldnames and "timestep" in fieldnames and "reward" in fieldnames


def _is_single_episode_timestep(fieldnames: list[str]) -> bool:
    """Return True if the CSV is a single-episode timestep telemetry file."""
    return "episode" in fieldnames and "timestep" in fieldnames


# ---------------------------------------------------------------------------
# Summarize single-episode CSV
# ---------------------------------------------------------------------------


def _summarize_single_episode_csv(path: Path, rows: list[dict]) -> None:
    episodes: dict[int, list[dict]] = {}
    for row in rows:
        ep = int(row.get("episode", 1))
        episodes.setdefault(ep, []).append(row)

    scores = [
        float(ep_rows[-1]["cumulative_sparse_reward"])
        for ep_rows in episodes.values()
    ]
    lengths = [len(ep_rows) for ep_rows in episodes.values()]
    shaped_0 = [
        sum(float(r.get("agent_0_shaped_reward", 0.0)) for r in ep_rows)
        for ep_rows in episodes.values()
    ]
    shaped_1 = [
        sum(float(r.get("agent_1_shaped_reward", 0.0)) for r in ep_rows)
        for ep_rows in episodes.values()
    ]

    print(f"\n{'='*70}")
    print(f"  File : {path.name}")
    print(f"  Type : single-episode timestep telemetry")
    print(f"{'='*70}")

    _print_section("Overview")
    print(f"    {'Episodes':<42}  {len(episodes)}")
    print(f"    {'Total timesteps':<42}  {sum(lengths)}")

    _print_section("Score")
    _stat_row("Team sparse reward (per episode)", scores, decimals=1)
    _stat_row("Cumulative shaped reward – agent 0", shaped_0, decimals=1)
    _stat_row("Cumulative shaped reward – agent 1", shaped_1, decimals=1)

    _print_section("Episode length")
    _stat_row("Timesteps per episode", lengths, decimals=0)


# ---------------------------------------------------------------------------
# Summarize episode metrics rows
# ---------------------------------------------------------------------------


def _summarize_episode_metrics_rows(path: Path, rows: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"  File : {path.name}")
    print(f"  Type : episode-level metrics")
    print(f"{'='*70}")

    n = len(rows)
    _print_section("Overview")
    print(f"    {'Episodes':<42}  {n}")

    for key in ("layout_name", "agent_0_name", "agent_1_name"):
        vals = {r.get(key, "") for r in rows} - {""}
        if vals:
            print(f"    {key:<42}  {', '.join(sorted(vals))}")

    _print_section("Performance")
    _stat_row("Team score", _col(rows, "team_score"), decimals=1)
    _stat_row("Soups delivered", _col(rows, "soups_delivered"), decimals=1)
    _stat_row("Episode length", _col(rows, "episode_length"), decimals=0)

    _print_section("Movement & Idle")
    _stat_row("Distance traveled – agent 0", _col(rows, "agent_0_distance_traveled"), decimals=1)
    _stat_row("Distance traveled – agent 1", _col(rows, "agent_1_distance_traveled"), decimals=1)
    _stat_row("Team distance traveled", _col(rows, "team_distance_traveled"), decimals=1)
    _stat_row("Idle timesteps – agent 0", _col(rows, "agent_0_idle_timesteps"), decimals=1)
    _stat_row("Idle timesteps – agent 1", _col(rows, "agent_1_idle_timesteps"), decimals=1)
    _stat_row("Idle rate – agent 0", _col(rows, "agent_0_idle_rate"), decimals=3)
    _stat_row("Idle rate – agent 1", _col(rows, "agent_1_idle_rate"), decimals=3)
    _stat_row("Team idle rate", _col(rows, "team_idle_rate"), decimals=3)

    _print_section("Collisions & Interference")
    _stat_row("Wall collision attempts – team", _col(rows, "team_wall_collision_attempts"), decimals=1)
    _stat_row("Teammate-blocking events – team", _col(rows, "team_blocking_events"), decimals=1)
    _stat_row("Collision events – team", _col(rows, "team_collision_events"), decimals=1)
    _stat_row("Same-target collision events", _col(rows, "same_target_collision_events"), decimals=1)
    _stat_row("Swap collision events", _col(rows, "swap_collision_events"), decimals=1)
    _stat_row("Interference timesteps – team", _col(rows, "team_interference_timesteps"), decimals=1)
    _stat_row("Repeated interference timesteps – team", _col(rows, "team_repeated_interference_timesteps"), decimals=1)
    _stat_row("Interference rate – team", _col(rows, "team_interference_rate"), decimals=3)

    _print_section("Role Proxies (Held Objects & Events)")
    _stat_row("Held onion share – agent 0", _col(rows, "agent_0_held_onion_share"), decimals=3)
    _stat_row("Held onion share – agent 1", _col(rows, "agent_1_held_onion_share"), decimals=3)
    _stat_row("Held dish share – agent 0", _col(rows, "agent_0_held_dish_share"), decimals=3)
    _stat_row("Held dish share – agent 1", _col(rows, "agent_1_held_dish_share"), decimals=3)
    _stat_row("Held soup share – agent 0", _col(rows, "agent_0_held_soup_share"), decimals=3)
    _stat_row("Held soup share – agent 1", _col(rows, "agent_1_held_soup_share"), decimals=3)
    _stat_row("Potting events – agent 0", _col(rows, "agent_0_potting_event_count"), decimals=1)
    _stat_row("Potting events – agent 1", _col(rows, "agent_1_potting_event_count"), decimals=1)
    _stat_row("Dish pickups – team", _col(rows, "team_dish_pickup_count"), decimals=1)
    _stat_row("Soup pickups – team", _col(rows, "team_soup_pickup_count"), decimals=1)
    _stat_row("Soup deliveries – agent 0", _col(rows, "agent_0_soup_delivery_count"), decimals=1)
    _stat_row("Soup deliveries – agent 1", _col(rows, "agent_1_soup_delivery_count"), decimals=1)

    _print_section("Task Duplication & Pipeline")
    _stat_row("Duplicate gather timesteps", _col(rows, "duplicate_gather_timesteps"), decimals=1)
    _stat_row("Duplicate dish timesteps", _col(rows, "duplicate_dish_timesteps"), decimals=1)
    _stat_row("Duplicate deliver timesteps", _col(rows, "duplicate_deliver_timesteps"), decimals=1)
    _stat_row("Task duplication timesteps – team", _col(rows, "team_task_duplication_timesteps"), decimals=1)
    _stat_row("Task duplication rate – team", _col(rows, "team_task_duplication_rate"), decimals=3)
    _stat_row("Unused pipeline timesteps – team", _col(rows, "team_unused_pipeline_timesteps"), decimals=1)
    _stat_row("Unused pipeline rate – team", _col(rows, "team_unused_pipeline_rate"), decimals=3)


# ---------------------------------------------------------------------------
# Process file / entry points
# ---------------------------------------------------------------------------


def process_telemetry_file(
    path: Path,
    output_path: Path | None = None,
    aggregate_output_path: Path | None = None,
) -> None:
    """Compute episode metrics from telemetry CSV, save metrics and summary CSVs, and print report."""
    rows = load_telemetry_csv(path)
    summaries = summarize_episodes(rows)

    metrics_out = output_path or path.with_name(f"{path.stem}_episode_metrics.csv")
    write_episode_metrics_csv(summaries, metrics_out)

    aggregates = aggregate_episode_metrics(summaries)
    if aggregate_output_path:
        write_aggregate_metrics_csv(aggregates, aggregate_output_path)

    with metrics_out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        metric_rows = list(reader)

    _summarize_episode_metrics_rows(metrics_out, metric_rows)
    print(f"\n  Output episode metrics CSV: {metrics_out.resolve()}")
    if aggregate_output_path:
        print(f"  Output aggregate metrics CSV: {aggregate_output_path.resolve()}")


def process_path(
    path: Path,
    output_path: Path | None = None,
    aggregate_output_path: Path | None = None,
) -> None:
    if not path.exists():
        print(f"[warn] File not found: {path}")
        return

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        print(f"[warn] Empty file: {path}")
        return

    if _is_episode_metrics(fieldnames):
        _summarize_episode_metrics_rows(path, rows)
        if aggregate_output_path:
            from telemetry import TelemetryLogger
            # Convert rows to EpisodeMetrics to aggregate
            # Or compute directly
            pass
    elif _is_multi_episode_timestep(fieldnames):
        process_telemetry_file(path, output_path, aggregate_output_path)
    elif _is_single_episode_timestep(fieldnames):
        _summarize_single_episode_csv(path, rows)
    else:
        print(
            f"[warn] Unrecognised CSV format: {path.name} (columns: {fieldnames[:6]}…)"
        )


def _find_default_inputs() -> list[Path]:
    """Return all CSVs in results directory."""
    if not DEFAULT_RESULTS_DIR.exists():
        return []
    return sorted(
        p for p in DEFAULT_RESULTS_DIR.glob("*.csv") if p.stat().st_size > 0
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute or print performance and coordination metrics from result CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="CSV file(s) to process or summarise. Defaults to results/*.csv when omitted.",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        type=Path,
        help="explicit input CSV file path",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        type=Path,
        help="optional output path for computed episode metrics CSV",
    )
    parser.add_argument(
        "--aggregate-output",
        dest="aggregate_output_file",
        type=Path,
        help="optional output path for computed batch aggregate metrics CSV",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    inputs: list[Path] = []
    if args.input_file:
        inputs.append(args.input_file)
    if args.inputs:
        inputs.extend(args.inputs)
    if not inputs:
        inputs = _find_default_inputs()

    if not inputs:
        # Fallback to default input if exists
        if DEFAULT_INPUT.exists():
            inputs = [DEFAULT_INPUT]
        else:
            print("No CSV files found in results/. Run an experiment first.")
            return

    for path in inputs:
        process_path(
            path,
            output_path=args.output_file,
            aggregate_output_path=args.aggregate_output_file,
        )

    print()


if __name__ == "__main__":
    main()
