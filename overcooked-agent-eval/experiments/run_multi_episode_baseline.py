"""Run multiple random-agent episodes and save per-timestep telemetry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetry import TelemetryLogger, TelemetryRow


DEFAULT_LAYOUT = "cramped_room"
DEFAULT_EPISODES = 5
DEFAULT_HORIZON = 400
DEFAULT_SEED = 42
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "multi_episode_random_baseline_cramped_room.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple random-agent episodes and save structured telemetry."
    )
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    return args


def action_name(action: object) -> str:
    names = {
        (0, -1): "north",
        (0, 1): "south",
        (1, 0): "east",
        (-1, 0): "west",
        (0, 0): "stay",
        "interact": "interact",
    }
    return names[action]


def held_object_name(player: object) -> str:
    held_object = player.held_object
    return "none" if held_object is None else held_object.name


def run_experiment(
    layout: str, episodes: int, horizon: int, seed: int
) -> tuple[TelemetryLogger, list[float], list[int]]:
    """Run the requested episodes and return telemetry and episode summaries."""
    np.random.seed(seed)
    mdp = OvercookedGridworld.from_layout_name(layout)
    logger = TelemetryLogger()
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    run_id = uuid4().hex

    for episode_id in range(1, episodes + 1):
        env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        agents = [RandomAgent(all_actions=True), RandomAgent(all_actions=True)]
        for index, agent in enumerate(agents):
            agent.set_agent_index(index)
            agent.set_mdp(mdp)

        state = env.state
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            actions_and_info = [agent.action(state) for agent in agents]
            joint_action = tuple(item[0] for item in actions_and_info)
            action_info = [item[1] for item in actions_and_info]
            next_state, reward, done, _ = env.step(
                joint_action, joint_agent_action_info=action_info
            )
            episode_reward += reward
            episode_length += 1
            players = next_state.players
            logger.log(
                TelemetryRow(
                    run_id=run_id,
                    episode_id=episode_id,
                    timestep=next_state.timestep,
                    layout_name=layout,
                    agent_0_name="RandomAgent",
                    agent_1_name="RandomAgent",
                    agent_0_action=action_name(joint_action[0]),
                    agent_1_action=action_name(joint_action[1]),
                    reward=reward,
                    done=done,
                    agent_0_position=repr(players[0].position),
                    agent_1_position=repr(players[1].position),
                    agent_0_orientation=action_name(players[0].orientation),
                    agent_1_orientation=action_name(players[1].orientation),
                    agent_0_held_object=held_object_name(players[0]),
                    agent_1_held_object=held_object_name(players[1]),
                )
            )
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return logger, episode_rewards, episode_lengths


def print_summary(
    layout: str,
    episode_rewards: list[float],
    episode_lengths: list[int],
    output_path: Path,
) -> None:
    episodes = len(episode_lengths)
    total_timesteps = sum(episode_lengths)
    print("Multi-episode random baseline complete")
    print(f"Layout: {layout}")
    print(f"Episodes: {episodes}")
    print(f"Timesteps logged: {total_timesteps}")
    print(f"Average episode length: {total_timesteps / episodes:.1f}")
    print(f"Average reward / score: {sum(episode_rewards) / episodes:.1f}")
    print(f"Output CSV: {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    logger, rewards, lengths = run_experiment(
        args.layout, args.episodes, args.horizon, args.seed
    )
    output_path = logger.save_csv(args.output)
    print_summary(args.layout, rewards, lengths, output_path)


if __name__ == "__main__":
    main()
