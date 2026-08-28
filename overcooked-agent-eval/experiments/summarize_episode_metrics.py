"""Compute episode summaries from a saved telemetry CSV."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import load_telemetry_csv, summarize_episodes, write_episode_metrics_csv


DEFAULT_INPUT = (
    PROJECT_ROOT / "results" / "multi_episode_random_baseline_cramped_room.csv"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute performance and coordination metrics from saved telemetry."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_episode_metrics.csv")
    return args


def main() -> None:
    args = parse_args()
    rows = load_telemetry_csv(args.input)
    summaries = summarize_episodes(rows)
    output_path = write_episode_metrics_csv(summaries, args.output)
    print("Episode metrics complete")
    print(f"Episodes summarized: {len(summaries)}")
    print(f"Total team score: {sum(summary.team_score for summary in summaries):.1f}")
    print(f"Soups delivered: {sum(summary.soups_delivered for summary in summaries)}")
    print(
        "Blocking events: "
        f"{sum(summary.team_blocking_events for summary in summaries)}"
    )
    print(
        "Collision events: "
        f"{sum(summary.team_collision_events for summary in summaries)}"
    )
    print(
        "Interference timesteps: "
        f"{sum(summary.team_interference_timesteps for summary in summaries)}"
    )
    print(f"Output CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
