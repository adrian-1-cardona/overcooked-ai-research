"""Run a reproducible two-agent random baseline in Overcooked-AI."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


DEFAULT_LAYOUT = "cramped_room"
DEFAULT_HORIZON = 400
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def action_name(action: object) -> str:
    """Return a readable, ASCII-safe name for an Overcooked action."""
    names = {
        (0, -1): "north",
        (0, 1): "south",
        (1, 0): "east",
        (-1, 0): "west",
        (0, 0): "stay",
        "interact": "interact",
    }
    return names[action]


def run_episode(layout: str, horizon: int, seed: int) -> list[dict[str, object]]:
    """Run one episode and return per-timestep telemetry rows."""
    np.random.seed(seed)
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    agents = [RandomAgent(all_actions=True), RandomAgent(all_actions=True)]
    for index, agent in enumerate(agents):
        agent.set_agent_index(index)
        agent.set_mdp(mdp)

    state = env.state
    done = False
    cumulative_sparse_reward = 0
    rows: list[dict[str, object]] = []

    while not done:
        actions_and_info = [agent.action(state) for agent in agents]
        joint_action = tuple(item[0] for item in actions_and_info)
        action_info = [item[1] for item in actions_and_info]
        next_state, sparse_reward, done, info = env.step(
            joint_action, joint_agent_action_info=action_info
        )
        cumulative_sparse_reward += sparse_reward
        rows.append(
            {
                "episode": 1,
                "timestep": next_state.timestep,
                "agent_0_action": action_name(joint_action[0]),
                "agent_1_action": action_name(joint_action[1]),
                "sparse_reward": sparse_reward,
                "agent_0_shaped_reward": info["shaped_r_by_agent"][0],
                "agent_1_shaped_reward": info["shaped_r_by_agent"][1],
                "cumulative_sparse_reward": cumulative_sparse_reward,
                "agent_0_position": repr(next_state.players[0].position),
                "agent_1_position": repr(next_state.players[1].position),
                "done": done,
            }
        )
        state = next_state

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
