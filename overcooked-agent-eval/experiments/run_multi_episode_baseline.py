"""Run multiple random-agent episodes across one or more layouts and save per-timestep telemetry.

What this script does (in simple terms):
This script runs a batch of Overcooked games between two robot chefs (RandomAgents).
You can tell it:
  - Which kitchen map to use (e.g. cramped_room, asymmetric_advantages, coordination_ring)
  - How many games to play (e.g. 5 or 10 games)
  - How long each game lasts (e.g. 400 steps)
  - What random seed number to start with (for 100% reproducible games)

At every step of every game, it logs what happened (positions, actions, soup points)
into a neat CSV spreadsheet and creates a companion JSON manifest recipe card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetry import (
    BatchManifest,
    TelemetryLogger,
    TelemetryRow,
    create_batch_manifest,
    save_batch_manifest,
)


# Default settings for quick baseline runs:
DEFAULT_LAYOUTS = ["cramped_room"]
DEFAULT_EPISODES = 5
DEFAULT_HORIZON = 400
DEFAULT_SEED = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read command-line arguments passed to this script."""
    parser = argparse.ArgumentParser(
        description="Run multiple random-agent episodes across layouts and save structured telemetry."
    )
    parser.add_argument(
        "--layout",
        "--layouts",
        dest="layouts",
        nargs="+",
        default=DEFAULT_LAYOUTS,
        help="one or more layout names (e.g. cramped_room asymmetric_advantages coordination_ring)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="number of games to play per layout (default: 5)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="number of timesteps per game (default: 400)",
    )
    parser.add_argument(
        "--base-seed",
        "--seed",
        dest="base_seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed for episode 1; later episode seeds increment by one (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional custom output CSV path",
    )
    args = parser.parse_args(argv)

    # Flatten any comma-separated layout strings (e.g. 'cramped_room,asymmetric_advantages')
    flattened_layouts: list[str] = []
    for item in args.layouts:
        for name in item.split(","):
            cleaned = name.strip()
            if cleaned and cleaned not in flattened_layouts:
                flattened_layouts.append(cleaned)
    args.layouts = flattened_layouts or DEFAULT_LAYOUTS

    # Check for invalid numbers
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.base_seed < 0:
        parser.error("--base-seed cannot be negative")

    return args


def validate_layouts(layouts: list[str]) -> None:
    """Pre-validate that all layout names exist in Overcooked-AI before running.
    
    If any map name is misspelled, it will tell you immediately before wasting time.
    """
    invalid_layouts: list[str] = []
    for layout in layouts:
        try:
            OvercookedGridworld.from_layout_name(layout)
        except Exception:
            invalid_layouts.append(layout)
    if invalid_layouts:
        raise ValueError(
            f"Invalid layout(s) specified: {', '.join(invalid_layouts)}. "
            "Please choose valid Overcooked-AI layout names."
        )


def action_name(action: object) -> str:
    """Convert movement coordinate deltas like (0, -1) into human words like 'north'."""
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
    """Return the name of the object in the chef's hands (or 'none')."""
    held_object = player.held_object
    return "none" if held_object is None else held_object.name


def deterministic_run_id(layout: str, episodes: int, horizon: int, seed: int) -> str:
    """Create a unique fingerprint ID for this exact experiment configuration."""
    configuration = {
        "schema_version": 2,
        "layout": layout,
        "episodes": episodes,
        "horizon": horizon,
        "base_seed": seed,
        "agents": ["RandomAgent", "RandomAgent"],
    }
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"random-{digest}"


def events_at_timestep(
    game_stats: dict[str, object], timestep: int, agent_index: int
) -> str:
    """Return a semicolon-separated list of game events (like 'potting_onion;soup_delivery')."""
    excluded = {
        "cumulative_sparse_rewards_by_agent",
        "cumulative_shaped_rewards_by_agent",
    }
    events = [
        event_name
        for event_name, timestamps_by_agent in game_stats.items()
        if event_name not in excluded and timestep in timestamps_by_agent[agent_index]
    ]
    return ";".join(sorted(events))


def run_experiment(
    layout: str, episodes: int, horizon: int, seed: int
) -> tuple[TelemetryLogger, list[float], list[int]]:
    """Run all requested games on a single kitchen layout and record the telemetry."""
    mdp = OvercookedGridworld.from_layout_name(layout)
    logger = TelemetryLogger()
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    run_id = deterministic_run_id(layout, episodes, horizon, seed)

    # Loop through each game episode
    for episode_id in range(1, episodes + 1):
        # Deterministic seed: starts at base_seed and increments by 1 for each game
        episode_seed = seed + episode_id - 1
        np.random.seed(episode_seed)
        
        # Set up environment and two random agents
        env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        agents = [RandomAgent(all_actions=True), RandomAgent(all_actions=True)]
        for index, agent in enumerate(agents):
            agent.set_agent_index(index)
            agent.set_mdp(mdp)

        state = env.state
        done = False
        episode_reward = 0.0
        episode_length = 0

        # Step through the game until the timer runs out
        while not done:
            actions_and_info = [agent.action(state) for agent in agents]
            joint_action = tuple(item[0] for item in actions_and_info)
            action_info = [item[1] for item in actions_and_info]
            previous_players = state.players
            next_state, reward, done, info = env.step(
                joint_action, joint_agent_action_info=action_info
            )
            episode_reward += reward
            episode_length += 1
            players = next_state.players
            
            # Log the full 27-field telemetry row
            logger.log(
                TelemetryRow(
                    run_id=run_id,
                    episode_id=episode_id,
                    episode_seed=episode_seed,
                    timestep=next_state.timestep,
                    layout_name=layout,
                    agent_0_id=0,
                    agent_1_id=1,
                    agent_0_name="RandomAgent",
                    agent_1_name="RandomAgent",
                    agent_0_action=action_name(joint_action[0]),
                    agent_1_action=action_name(joint_action[1]),
                    reward=reward,
                    agent_0_sparse_reward=info["sparse_r_by_agent"][0],
                    agent_1_sparse_reward=info["sparse_r_by_agent"][1],
                    agent_0_shaped_reward=info["shaped_r_by_agent"][0],
                    agent_1_shaped_reward=info["shaped_r_by_agent"][1],
                    done=done,
                    agent_0_previous_position=json.dumps(
                        previous_players[0].position
                    ),
                    agent_1_previous_position=json.dumps(
                        previous_players[1].position
                    ),
                    agent_0_position=json.dumps(players[0].position),
                    agent_1_position=json.dumps(players[1].position),
                    agent_0_orientation=action_name(players[0].orientation),
                    agent_1_orientation=action_name(players[1].orientation),
                    agent_0_held_object=held_object_name(players[0]),
                    agent_1_held_object=held_object_name(players[1]),
                    agent_0_events=events_at_timestep(
                        env.game_stats, next_state.timestep - 1, 0
                    ),
                    agent_1_events=events_at_timestep(
                        env.game_stats, next_state.timestep - 1, 1
                    ),
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
    manifest_path: Path,
) -> None:
    """Print a clean summary of the completed batch to the terminal."""
    episodes = len(episode_lengths)
    total_timesteps = sum(episode_lengths)
    print("\nMulti-episode random baseline complete")
    print(f"Layout: {layout}")
    print(f"Episodes: {episodes}")
    print(f"Timesteps logged: {total_timesteps}")
    print(f"Average episode length: {total_timesteps / episodes:.1f}")
    print(f"Average reward / score: {sum(episode_rewards) / episodes:.1f}")
    print(f"Output CSV: {output_path.resolve()}")
    print(f"Manifest JSON: {manifest_path.resolve()}")


def execute_batch(
    layouts: list[str],
    episodes: int,
    horizon: int,
    base_seed: int,
    custom_output: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Run experiments across all specified layouts, saving CSV and manifest files."""
    # First, validate all layout names
    validate_layouts(layouts)
    results: list[tuple[Path, Path]] = []

    # Run each layout in turn
    for layout in layouts:
        if custom_output is not None and len(layouts) == 1:
            csv_path = custom_output
        elif custom_output is not None:
            csv_path = custom_output.parent / f"{custom_output.stem}_{layout}.csv"
        else:
            csv_path = (
                PROJECT_ROOT
                / "results"
                / f"multi_episode_random_baseline_{layout}.csv"
            )

        manifest_path = csv_path.with_name(f"{csv_path.stem}.manifest.json")
        run_id = deterministic_run_id(layout, episodes, horizon, base_seed)

        # Run the experiment
        logger, rewards, lengths = run_experiment(
            layout, episodes, horizon, base_seed
        )
        
        # Save and validate CSV
        saved_csv = logger.save_csv(csv_path)
        validated_count = logger.validate_csv(saved_csv)
        if validated_count != len(logger.rows):
            raise RuntimeError(
                f"Saved telemetry row count does not match the experiment for {layout}"
            )

        # Create and save JSON reproducibility manifest
        manifest = create_batch_manifest(
            run_id=run_id,
            layout=layout,
            agent_0_name="RandomAgent",
            agent_1_name="RandomAgent",
            episodes=episodes,
            horizon=horizon,
            base_seed=base_seed,
            output_telemetry_path=saved_csv,
            project_root=PROJECT_ROOT,
        )
        saved_manifest = save_batch_manifest(manifest, manifest_path)

        print_summary(layout, rewards, lengths, saved_csv, saved_manifest)
        results.append((saved_csv, saved_manifest))

    return results


def main() -> None:
    args = parse_args()
    execute_batch(
        layouts=args.layouts,
        episodes=args.episodes,
        horizon=args.horizon,
        base_seed=args.base_seed,
        custom_output=args.output,
    )


if __name__ == "__main__":
    main()
