"""Run a reproducible two-agent random baseline in Overcooked-AI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_LAYOUT = "cramped_room"
DEFAULT_HORIZON = 400
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_multi_episode_baseline import run_experiment


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "random_baseline_cramped_room.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two random agents and save one CSV row per timestep."
    )
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    return args


def run_episode(layout: str, horizon: int, seed: int) -> list[dict[str, object]]:
    """Run one episode through the reusable runner and keep the legacy CSV shape."""
    logger, _, _ = run_experiment(layout, episodes=1, horizon=horizon, seed=seed)
    cumulative_sparse_reward = 0
    rows: list[dict[str, object]] = []
    for telemetry_row in logger.rows:
        cumulative_sparse_reward += telemetry_row.reward
        rows.append(
            {
                "episode": 1,
                "timestep": telemetry_row.timestep,
                "agent_0_action": telemetry_row.agent_0_action,
                "agent_1_action": telemetry_row.agent_1_action,
                "sparse_reward": telemetry_row.reward,
                "agent_0_shaped_reward": telemetry_row.agent_0_shaped_reward,
                "agent_1_shaped_reward": telemetry_row.agent_1_shaped_reward,
                "cumulative_sparse_reward": cumulative_sparse_reward,
                "agent_0_position": repr(
                    tuple(json.loads(telemetry_row.agent_0_position))
                ),
                "agent_1_position": repr(
                    tuple(json.loads(telemetry_row.agent_1_position))
                ),
                "done": telemetry_row.done,
            }
        )

    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write telemetry rows to a CSV, creating the result directory if needed."""
    if not rows:
        raise ValueError("Cannot write an empty episode")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(layout: str, rows: list[dict[str, object]], output: Path) -> None:
    print("Random baseline complete")
    print(f"Layout: {layout}")
    print(f"Episode length: {len(rows)}")
    print(f"Total reward / score: {rows[-1]['cumulative_sparse_reward']}")
    print(f"Timesteps logged: {len(rows)}")
    print(f"Output CSV: {output.resolve()}")


def main() -> None:
    args = parse_args()
    rows = run_episode(args.layout, args.horizon, args.seed)
    write_csv(rows, args.output)
    print_summary(args.layout, rows, args.output)


if __name__ == "__main__":
    main()
