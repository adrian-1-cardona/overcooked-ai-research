"""Evaluate human-proxy compatibility and render coordination diagnostics.

The frozen behavior-cloning (BC) policy bundled with the PPO_BC checkpoint is
used as a reproducible proxy for human behavior.  We compare its learned
teammate with a self-play PPO policy, while retaining PPO self-play as the
in-distribution baseline.  Results include episode metrics, timestep telemetry,
a comparison graph, and exact trajectory videos.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CURRENT_FILE = Path(__file__).resolve()
LOCAL_PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))

from experiments.run_pretrained_policy_comparison import (
    PROJECT_ROOT,
    bundled_checkpoint,
    inspect_observation_contract,
    load_trainer,
)
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction


CONDITIONS = ("BC+PPO_BC", "BC+PPO_SP", "PPO_SP+PPO_SP")
MOVEMENT_ACTIONS = {
    Direction.NORTH,
    Direction.SOUTH,
    Direction.EAST,
    Direction.WEST,
}


@dataclass(frozen=True)
class EpisodeMetrics:
    condition: str
    seat_order: str
    episode: int
    seed: int
    sparse_return: float
    deliveries: int
    episode_length: int
    joint_stationary_rate: float
    blocked_move_rate: float
    longest_joint_stationary_streak: int
    max_delivery_drought: int
    repeated_joint_position_rate: float
    unique_joint_positions: int


@dataclass(frozen=True)
class EpisodeRecord:
    metrics: EpisodeMetrics
    states: Sequence[Any]
    actions: Sequence[Sequence[Any]]
    rewards: Sequence[float]
    infos: Sequence[dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PPO_BC and PPO_SP compatibility with a frozen BC human proxy"
    )
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--ppo-bc-checkpoint", type=Path)
    parser.add_argument("--ppo-sp-checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "human_model_coordination",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip representative MP4 exports"
    )
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    return args


def _agent(trainer: Any, policy_id: str, index: int) -> Any:
    from human_aware_rl.rllib.rllib import get_agent_from_trainer

    return get_agent_from_trainer(trainer, policy_id=policy_id, agent_index=index)


def make_pair(
    first_trainer: Any,
    first_policy: str,
    second_trainer: Any,
    second_policy: str,
) -> AgentPair:
    """Build a pair even when its policies come from different trainers."""
    return AgentPair(
        _agent(first_trainer, first_policy, 0),
        _agent(second_trainer, second_policy, 1),
    )


def _positions(state: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(state.players[0].position), tuple(state.players[1].position)


def _longest_true_streak(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _max_zero_reward_streak(rewards: Sequence[float]) -> int:
    return _longest_true_streak([float(reward) <= 0 for reward in rewards])


def compute_metrics(
    condition: str,
    seat_order: str,
    episode: int,
    seed: int,
    states: Sequence[Any],
    actions: Sequence[Sequence[Any]],
    rewards: Sequence[float],
) -> EpisodeMetrics:
    """Compute interpretable proxies for stalled or rigid coordination."""
    step_count = min(len(actions), max(0, len(states) - 1))
    if step_count == 0:
        raise ValueError("An episode must contain at least one transition")

    positions = [_positions(state) for state in states[: step_count + 1]]
    stationary: list[bool] = []
    movement_attempts = blocked_attempts = 0
    for timestep in range(step_count):
        before, after = positions[timestep], positions[timestep + 1]
        stationary.append(before == after)
        for player_index in (0, 1):
            action = actions[timestep][player_index]
            if action in MOVEMENT_ACTIONS:
                movement_attempts += 1
                if before[player_index] == after[player_index]:
                    blocked_attempts += 1

    visited = positions[:-1]
    most_common_visits = max(Counter(visited).values())
    relevant_rewards = [float(value) for value in rewards[:step_count]]
    sparse_return = float(sum(relevant_rewards))
    return EpisodeMetrics(
        condition=condition,
        seat_order=seat_order,
        episode=episode,
        seed=seed,
        sparse_return=sparse_return,
        deliveries=sum(value > 0 for value in relevant_rewards),
        episode_length=step_count,
        joint_stationary_rate=float(np.mean(stationary)),
        blocked_move_rate=(blocked_attempts / movement_attempts if movement_attempts else 0.0),
        longest_joint_stationary_streak=_longest_true_streak(stationary),
        max_delivery_drought=_max_zero_reward_streak(relevant_rewards),
        repeated_joint_position_rate=most_common_visits / step_count,
        unique_joint_positions=len(set(visited)),
    )


def evaluate_pair(
    evaluator: AgentEvaluator,
    pair: AgentPair,
    condition: str,
    seat_order: str,
    episodes: int,
    seed: int,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for index in range(episodes):
        episode_seed = seed + index
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        # Request the successor state as a final sentinel row so transition
        # diagnostics include the episode's last real action. The sentinel's
        # null action/reward/info are deliberately excluded below.
        trajectories = evaluator.env.get_rollouts(
            pair, num_games=1, final_state=True, info=False
        )
        states = trajectories["ep_states"][0]
        actions = trajectories["ep_actions"][0][:-1]
        rewards = trajectories["ep_rewards"][0][:-1]
        infos = trajectories["ep_infos"][0][:-1]
        records.append(
            EpisodeRecord(
                metrics=compute_metrics(
                    condition,
                    seat_order,
                    index + 1,
                    episode_seed,
                    states,
                    actions,
                    rewards,
                ),
                states=states,
                actions=actions,
                rewards=rewards,
                infos=infos,
            )
        )
    return records


def write_metrics(records: Sequence[EpisodeRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(records[0].metrics).keys())
        writer.writeheader()
        writer.writerows(asdict(record.metrics) for record in records)


def action_name(action: Any) -> str:
    names = {
        Direction.NORTH: "north",
        Direction.SOUTH: "south",
        Direction.EAST: "east",
        Direction.WEST: "west",
        Action.STAY: "stay",
        Action.INTERACT: "interact",
    }
    return names.get(action, str(action))


def write_telemetry(records: Sequence[EpisodeRecord], output: Path, layout: str) -> None:
    fields = [
        "condition", "seat_order", "episode", "seed", "layout_name", "timestep",
        "agent_0_action", "agent_1_action", "sparse_reward",
        "agent_0_shaped_reward", "agent_1_shaped_reward", "cumulative_sparse_reward",
        "agent_0_position", "agent_1_position",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            cumulative = 0.0
            for timestep, (actions, reward) in enumerate(zip(record.actions, record.rewards)):
                cumulative += float(reward)
                info = record.infos[timestep] if timestep < len(record.infos) else {}
                shaped = info.get("shaped_r_by_agent", (0.0, 0.0))
                positions = _positions(record.states[timestep])
                writer.writerow(
                    {
                        "condition": record.metrics.condition,
                        "seat_order": record.metrics.seat_order,
                        "episode": record.metrics.episode,
                        "seed": record.metrics.seed,
                        "layout_name": layout,
                        "timestep": timestep,
                        "agent_0_action": action_name(actions[0]),
                        "agent_1_action": action_name(actions[1]),
                        "sparse_reward": float(reward),
                        "agent_0_shaped_reward": float(shaped[0]),
                        "agent_1_shaped_reward": float(shaped[1]),
                        "cumulative_sparse_reward": cumulative,
                        "agent_0_position": positions[0],
                        "agent_1_position": positions[1],
                    }
                )


def condition_summary(records: Sequence[EpisodeRecord]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    fields = (
        "sparse_return",
        "joint_stationary_rate",
        "blocked_move_rate",
        "max_delivery_drought",
        "repeated_joint_position_rate",
    )
    for condition in CONDITIONS:
        metrics = [record.metrics for record in records if record.metrics.condition == condition]
        if not metrics:
            continue
        summary[condition] = {
            field: float(np.mean([getattr(metric, field) for metric in metrics]))
            for field in fields
        }
    return summary


def plot_summary(records: Sequence[EpisodeRecord], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = condition_summary(records)
    labels = list(summary)
    panels = (
        ("sparse_return", "Mean sparse return", "higher is better"),
        ("joint_stationary_rate", "Joint stationary rate", "lower is better"),
        ("blocked_move_rate", "Blocked movement rate", "lower is better"),
        ("max_delivery_drought", "Longest delivery drought", "steps; lower is better"),
    )
    colors = ["#2a9d8f", "#e76f51", "#457b9d"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (field, title, subtitle) in zip(axes.flat, panels):
        values = [summary[label][field] for label in labels]
        axis.bar(labels, values, color=colors[: len(labels)])
        axis.set_title(f"{title}\n({subtitle})")
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Human-proxy compatibility diagnostics", fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_video(record: EpisodeRecord, evaluator: AgentEvaluator, output: Path, fps: int) -> None:
    import cv2
    import pygame
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

    pygame.init()
    visualizer = StateVisualizer()
    rewards = [float(value) for value in record.rewards]
    cumulative = np.cumsum(rewards)
    first_hud = StateVisualizer.default_hud_data(
        record.states[0], score=0, time_left=record.metrics.episode_length
    )
    first = visualizer.render_state(
        record.states[0], evaluator.env.mdp.terrain_mtx, hud_data=first_hud
    )
    width, height = first.get_size()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for timestep, state in enumerate(record.states):
            score = 0.0 if timestep == 0 else cumulative[min(timestep - 1, len(cumulative) - 1)]
            hud = StateVisualizer.default_hud_data(
                state,
                score=score,
                time_left=max(0, record.metrics.episode_length - timestep),
            )
            surface = visualizer.render_state(
                state, evaluator.env.mdp.terrain_mtx, hud_data=hud
            )
            rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        pygame.quit()


def write_report(
    records: Sequence[EpisodeRecord], output: Path, layout: str, contract: Any
) -> None:
    summary = condition_summary(records)
    lines = [
        "# Human-proxy coordination report",
        "",
        f"Layout: `{layout}`. BC observation shape: `{contract.bc_shape}`; "
        f"PPO spatial observation shape: `{contract.sp_shape}`.",
        "",
        "The frozen BC policy is a reproducible human-behavior proxy, not a substitute "
        "for a study with new human participants. Stationary, blocked-movement, repeated-"
        "position, and delivery-drought statistics are diagnostic evidence of coordination "
        "lock; they do not by themselves prove a psychological cause.",
        "",
        "| Condition | Episodes | Mean return | Joint stationary | Blocked moves | Delivery drought |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        if condition not in summary:
            continue
        values = summary[condition]
        count = sum(record.metrics.condition == condition for record in records)
        lines.append(
            f"| {condition} | {count} | {values['sparse_return']:.1f} | "
            f"{values['joint_stationary_rate']:.1%} | {values['blocked_move_rate']:.1%} | "
            f"{values['max_delivery_drought']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Use `comparison.png` for the aggregate view, `episode_metrics.csv` for "
            "numerical analysis, `telemetry.csv` for timestep-level inspection, and the "
            "MP4 files to visually inspect the exact evaluated trajectories.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = AgentEvaluator.from_layout_name(
        {"layout_name": args.layout, "old_dynamics": True},
        {"horizon": args.horizon},
    )
    contract = inspect_observation_contract(evaluator)
    bc_path = args.ppo_bc_checkpoint or bundled_checkpoint(args.layout, "BC")
    sp_path = args.ppo_sp_checkpoint or bundled_checkpoint(args.layout, "SP")

    records: list[EpisodeRecord] = []
    with ExitStack() as stack:
        bc_trainer = load_trainer(bc_path)
        stack.callback(bc_trainer.stop)
        sp_trainer = load_trainer(sp_path)
        stack.callback(sp_trainer.stop)

        specifications = (
            ("BC+PPO_BC", "bc_0+ppo_bc_1", bc_trainer, "bc", bc_trainer, "ppo"),
            ("BC+PPO_BC", "ppo_bc_0+bc_1", bc_trainer, "ppo", bc_trainer, "bc"),
            ("BC+PPO_SP", "bc_0+ppo_sp_1", bc_trainer, "bc", sp_trainer, "ppo"),
            ("BC+PPO_SP", "ppo_sp_0+bc_1", sp_trainer, "ppo", bc_trainer, "bc"),
            ("PPO_SP+PPO_SP", "ppo_sp_0+ppo_sp_1", sp_trainer, "ppo", sp_trainer, "ppo"),
        )
        for offset, specification in enumerate(specifications):
            condition, seats, trainer_0, policy_0, trainer_1, policy_1 = specification
            count = args.episodes * 2 if condition == "PPO_SP+PPO_SP" else args.episodes
            records.extend(
                evaluate_pair(
                    evaluator,
                    make_pair(trainer_0, policy_0, trainer_1, policy_1),
                    condition,
                    seats,
                    count,
                    args.seed + offset * args.episodes,
                )
            )

    write_metrics(records, output_dir / "episode_metrics.csv")
    write_telemetry(records, output_dir / "telemetry.csv", args.layout)
    plot_summary(records, output_dir / "comparison.png")
    write_report(records, output_dir / "README.md", args.layout, contract)

    if not args.no_video:
        for condition in CONDITIONS:
            candidates = [record for record in records if record.metrics.condition == condition]
            representative = min(candidates, key=lambda record: record.metrics.sparse_return)
            filename = condition.lower().replace("+", "_with_") + ".mp4"
            render_video(representative, evaluator, output_dir / filename, args.fps)

    print("Human-proxy coordination analysis complete")
    for condition, values in condition_summary(records).items():
        print(
            f"{condition}: return={values['sparse_return']:.1f}, "
            f"stationary={values['joint_stationary_rate']:.1%}, "
            f"blocked={values['blocked_move_rate']:.1%}, "
            f"delivery_drought={values['max_delivery_drought']:.1f}"
        )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
