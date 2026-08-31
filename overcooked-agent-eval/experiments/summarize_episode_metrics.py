"""Summarize episode metrics from one or more Overcooked-AI result CSVs.

Accepts two CSV formats produced by this framework:

  1. Timestep-level CSV  (columns: episode, timestep, sparse_reward, …)
     – e.g. results/random_baseline_cramped_room.csv

  2. Episode-level metrics CSV  (columns: run_id, episode_id, team_score, …)
     – e.g. results/multi_episode_random_baseline_cramped_room_episode_metrics.csv

If multiple files are given, each is summarised in turn.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: float, decimals: int = 3) -> str:
    """Format a float, suppressing unnecessary trailing zeros."""
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _print_section(title: str) -> None:
    print(f"\n  {title}")
    print("  " + "-" * len(title))


def _stat_row(label: str, values: list[float], decimals: int = 3) -> None:
    if not values:
        print(f"    {label:<40}  n/a")
        return
    mean = statistics.mean(values)
    n = len(values)
    if n == 1:
        print(f"    {label:<40}  {_fmt(mean, decimals)}")
    else:
        stdev = statistics.stdev(values)
        mn, mx = min(values), max(values)
        print(
            f"    {label:<40}  mean={_fmt(mean, decimals)}"
            f"  std={_fmt(stdev, decimals)}"
            f"  min={_fmt(mn, decimals)}"
            f"  max={_fmt(mx, decimals)}"
        )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_episode_metrics(fieldnames: list[str]) -> bool:
    """Return True if the CSV looks like an episode-level metrics file."""
    return "team_score" in fieldnames


def _is_multi_episode_timestep(fieldnames: list[str]) -> bool:
    """Return True if the CSV is a multi-episode timestep telemetry file."""
    return "episode_id" in fieldnames and "timestep" in fieldnames and "reward" in fieldnames


def _is_single_episode_timestep(fieldnames: list[str]) -> bool:
    """Return True if the CSV is a single-episode timestep telemetry file."""
    return "episode" in fieldnames and "timestep" in fieldnames


# ---------------------------------------------------------------------------
# Summarise timestep-level CSV
# ---------------------------------------------------------------------------

def _summarise_timestep_csv(path: Path, rows: list[dict]) -> None:
    episodes: dict[int, list[dict]] = {}
    for row in rows:
        ep = int(row["episode"])
        episodes.setdefault(ep, []).append(row)

    scores = [float(ep_rows[-1]["cumulative_sparse_reward"]) for ep_rows in episodes.values()]
    lengths = [len(ep_rows) for ep_rows in episodes.values()]
    shaped_0 = [
        sum(float(r["agent_0_shaped_reward"]) for r in ep_rows)
        for ep_rows in episodes.values()
    ]
    shaped_1 = [
        sum(float(r["agent_1_shaped_reward"]) for r in ep_rows)
        for ep_rows in episodes.values()
    ]

    print(f"\n{'='*60}")
    print(f"  File : {path.name}")
    print(f"  Type : timestep-level telemetry")
    print(f"{'='*60}")

    _print_section("Overview")
    print(f"    {'Episodes':<40}  {len(episodes)}")
    print(f"    {'Total timesteps':<40}  {sum(lengths)}")

    _print_section("Score")
    _stat_row("Team sparse reward (per episode)", scores, decimals=1)
    _stat_row("Cumulative shaped reward – agent 0", shaped_0, decimals=1)
    _stat_row("Cumulative shaped reward – agent 1", shaped_1, decimals=1)

    _print_section("Episode length")
    _stat_row("Timesteps per episode", lengths, decimals=0)


# ---------------------------------------------------------------------------
# Summarise multi-episode timestep-level CSV
# ---------------------------------------------------------------------------

def _summarise_multi_episode_timestep_csv(path: Path, rows: list[dict]) -> None:
    episodes: dict[int, list[dict]] = {}
    for row in rows:
        ep = int(row["episode_id"])
        episodes.setdefault(ep, []).append(row)

    # Cumulative reward per episode = sum of per-step rewards
    scores = [sum(float(r["reward"]) for r in ep_rows) for ep_rows in episodes.values()]
    lengths = [len(ep_rows) for ep_rows in episodes.values()]
    shaped_0 = [
        sum(float(r["agent_0_shaped_reward"]) for r in ep_rows)
        for ep_rows in episodes.values()
    ]
    shaped_1 = [
        sum(float(r["agent_1_shaped_reward"]) for r in ep_rows)
        for ep_rows in episodes.values()
    ]

    print(f"\n{'='*60}")
    print(f"  File : {path.name}")
    print(f"  Type : multi-episode timestep telemetry")
    print(f"{'='*60}")

    # Metadata
    layouts = {r.get("layout_name", "") for r in rows} - {""}
    agents_0 = {r.get("agent_0_name", "") for r in rows} - {""}
    agents_1 = {r.get("agent_1_name", "") for r in rows} - {""}
    _print_section("Overview")
    print(f"    {'Episodes':<40}  {len(episodes)}")
    print(f"    {'Total timesteps':<40}  {sum(lengths)}")
    if layouts:
        print(f"    {'layout_name':<40}  {', '.join(sorted(layouts))}")
    if agents_0:
        print(f"    {'agent_0_name':<40}  {', '.join(sorted(agents_0))}")
    if agents_1:
        print(f"    {'agent_1_name':<40}  {', '.join(sorted(agents_1))}")

    _print_section("Score")
    _stat_row("Team sparse reward (per episode)", scores, decimals=1)
    _stat_row("Cumulative shaped reward – agent 0", shaped_0, decimals=1)
    _stat_row("Cumulative shaped reward – agent 1", shaped_1, decimals=1)

    _print_section("Episode length")
    _stat_row("Timesteps per episode", lengths, decimals=0)


# ---------------------------------------------------------------------------
# Summarise episode-level metrics CSV
# ---------------------------------------------------------------------------

def _col(rows: list[dict], key: str) -> list[float]:
    """Extract a numeric column, silently skipping missing / empty values."""
    out = []
    for r in rows:
        v = r.get(key, "")
        if v != "":
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _summarise_episode_metrics_csv(path: Path, rows: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  File : {path.name}")
    print(f"  Type : episode-level metrics")
    print(f"{'='*60}")

    n = len(rows)
    _print_section("Overview")
    print(f"    {'Episodes':<40}  {n}")

    # Metadata (show unique values if homogeneous)
    for key in ("layout_name", "agent_0_name", "agent_1_name"):
        vals = {r.get(key, "") for r in rows} - {""}
        if vals:
            print(f"    {key:<40}  {', '.join(sorted(vals))}")

    _print_section("Score")
    _stat_row("Team score", _col(rows, "team_score"), decimals=1)
    _stat_row("Soups delivered", _col(rows, "soups_delivered"), decimals=1)

    _print_section("Movement")
    _stat_row("Distance traveled – agent 0", _col(rows, "agent_0_distance_traveled"), decimals=1)
    _stat_row("Distance traveled – agent 1", _col(rows, "agent_1_distance_traveled"), decimals=1)
    _stat_row("Team distance traveled", _col(rows, "team_distance_traveled"), decimals=1)

    _print_section("Idle")
    _stat_row("Idle timesteps – agent 0", _col(rows, "agent_0_idle_timesteps"), decimals=1)
    _stat_row("Idle timesteps – agent 1", _col(rows, "agent_1_idle_timesteps"), decimals=1)
    _stat_row("Idle rate – agent 0", _col(rows, "agent_0_idle_rate"), decimals=3)
    _stat_row("Idle rate – agent 1", _col(rows, "agent_1_idle_rate"), decimals=3)

    _print_section("Collisions & interference")
    _stat_row("Wall collision attempts – team", _col(rows, "team_wall_collision_attempts"), decimals=1)
    _stat_row("Teammate-blocking events – team", _col(rows, "team_blocking_events"), decimals=1)
    _stat_row("Interference timesteps – team", _col(rows, "team_interference_timesteps"), decimals=1)
    _stat_row("Interference rate – team", _col(rows, "team_interference_rate"), decimals=3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_default_inputs() -> list[Path]:
    """Return all CSVs in the results directory, excluding .gitkeep."""
    return sorted(
        p for p in DEFAULT_RESULTS_DIR.glob("*.csv") if p.stat().st_size > 0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a summary of one or more Overcooked-AI result CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "CSV file(s) to summarise. "
            "Defaults to all CSVs in results/ when omitted."
        ),
    )
    return parser.parse_args()


def summarise(path: Path) -> None:
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
        _summarise_episode_metrics_csv(path, rows)
    elif _is_multi_episode_timestep(fieldnames):
        _summarise_multi_episode_timestep_csv(path, rows)
    elif _is_single_episode_timestep(fieldnames):
        _summarise_timestep_csv(path, rows)
    else:
        print(f"[warn] Unrecognised CSV format: {path.name} (columns: {fieldnames[:6]}…)")


def main() -> None:
    args = parse_args()
    inputs: list[Path] = args.inputs or _find_default_inputs()

    if not inputs:
        print("No CSV files found. Run an experiment first.")
        return

    for path in inputs:
        summarise(path)

    print()


if __name__ == "__main__":
    main()
