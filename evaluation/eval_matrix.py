"""
Automated n x n x m x k Headless Evaluation Engine for Overcooked-AI.

Runs all n x n agent pairings across m levels for k episodes each.
Executes purely in memory (headless, no rendering window) for maximum speed.
Supports multi-process level sharding across CPU cores.
Saves results to results/matrix.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project directories are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from gui.agent_manager import create_agent, is_ppo_supported

DEFAULT_AGENTS = ["ppo_sp", "greedy", "random"]
DEFAULT_LEVELS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]
DEFAULT_HORIZON = 400
DEFAULT_K = 5
DEFAULT_OUTPUT = REPO_ROOT / "results" / "matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated n x n x m x k Overcooked-AI Evaluation Engine"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        help="List of n agents to test in cross-play (e.g., ppo_sp greedy random).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=DEFAULT_LEVELS,
        help="List of m layout names (e.g., cramped_room forced_coordination).",
    )
    parser.add_argument(
        "-k",
        "--episodes",
        type=int,
        default=DEFAULT_K,
        help="Number of trials/episodes per pairing on each level (default: 5).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Timestep horizon per episode (default: 400).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel worker processes across layout levels (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the output CSV (default: results/matrix.csv).",
    )
    return parser.parse_args()


def run_single_episode(
    agent_0_type: str,
    agent_1_type: str,
    layout_name: str,
    horizon: int,
    seed: int,
) -> tuple[float, int, int]:
    """Run one episode headlessly in memory.

    Returns:
        (cumulative_sparse_reward, dishes_served, collision_count)
    """
    np.random.seed(seed)

    try:
        evaluator = AgentEvaluator.from_layout_name(
            {"layout_name": layout_name, "old_dynamics": True},
            {"horizon": horizon},
        )
    except Exception:
        evaluator = AgentEvaluator.from_layout_name(
            {"layout_name": layout_name, "old_dynamics": False},
            {"horizon": horizon},
        )
    env = evaluator.env
    mdp = env.mdp

    agent_0, _ = create_agent(agent_0_type, agent_index=0, mdp=mdp, layout_name=layout_name)
    agent_1, _ = create_agent(agent_1_type, agent_index=1, mdp=mdp, layout_name=layout_name)

    env.reset()
    # Note: Do not call agent.reset() after create_agent() because Agent.reset() wipes agent_index to None!

    total_reward = 0.0
    dishes_served = 0
    collisions = 0

    while not env.is_done():
        state = env.state
        a0, _ = agent_0.action(state)
        a1, _ = agent_1.action(state)
        joint_action = (a0, a1)

        # Track collisions (both moving into same tile or bumping head-on)
        p0_pos = state.players[0].position
        p1_pos = state.players[1].position
        next_state, reward, done, info = env.step(joint_action)

        if reward > 0:
            total_reward += reward
            # Standard Overcooked soup delivery reward is 20
            dishes_served += int(round(reward / 20.0))

        # Check for collision / mutual obstruction
        new_p0 = next_state.players[0].position
        new_p1 = next_state.players[1].position
        if a0 != Action.STAY and new_p0 == p0_pos:
            collisions += 1
        if a1 != Action.STAY and new_p1 == p1_pos:
            collisions += 1

    return total_reward, dishes_served, collisions


def _evaluate_level_worker(
    level: str,
    agents: list[str],
    k: int,
    horizon: int,
    progress_queue: Any,
) -> None:
    """Worker process evaluating all n x n agent pairings on a single layout."""
    for a0, a1 in itertools.product(agents, repeat=2):
        for trial in range(k):
            seed = 1000 + trial * 37
            score, dishes, collisions = run_single_episode(
                agent_0_type=a0,
                agent_1_type=a1,
                layout_name=level,
                horizon=horizon,
                seed=seed,
            )
            row = {
                "agent_0": a0,
                "agent_1": a1,
                "level": level,
                "trial": trial + 1,
                "score": score,
                "dishes": dishes,
                "collisions": collisions,
            }
            if progress_queue is not None:
                progress_queue.put(("row", row))
    if progress_queue is not None:
        progress_queue.put(("done", level))


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agents = args.agents
    levels = args.levels
    k = args.episodes
    horizon = args.horizon

    n = len(agents)
    m = len(levels)
    total_matchups = n * n * m
    total_episodes = total_matchups * k

    print("=" * 60)
    print("Overcooked-AI Matrix Benchmark Engine")
    print(f"Agents (n={n}): {', '.join(agents)}")
    print(f"Levels (m={m}): {', '.join(levels)}")
    print(f"Trials per pair (k): {k}")
    print(f"Horizon per episode: {horizon}")
    print(f"Total Matchups: {total_matchups} ({n}x{n} pairings x {m} levels)")
    print(f"Total Episodes: {total_episodes}")
    print(f"Output Path: {output_path}")
    print("=" * 60, flush=True)

    rows: list[dict[str, Any]] = []
    start_time = time.time()
    completed_episodes = 0

    num_workers = min(args.workers, len(levels))
    if num_workers > 1:
        print(f"Executing with {num_workers} parallel layout worker processes...\n", flush=True)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes: list[mp.Process] = []

        for lvl in levels:
            p = ctx.Process(
                target=_evaluate_level_worker,
                args=(lvl, agents, k, horizon, queue),
            )
            p.daemon = True
            p.start()
            processes.append(p)

        finished_workers = 0
        while finished_workers < len(processes):
            msg_type, payload = queue.get()
            if msg_type == "row":
                row = payload
                rows.append(row)
                completed_episodes += 1
                progress = f"[{completed_episodes}/{total_episodes}]"
                print(
                    f"{progress} {row['agent_0']} x {row['agent_1']} ({row['level']}) Trial {row['trial']}/{k}: Score = {row['score']:.1f}, Dishes = {row['dishes']}",
                    flush=True,
                )
            elif msg_type == "done":
                finished_workers += 1

        for p in processes:
            p.join(timeout=30)
    else:
        for level in levels:
            print(f"\n--- [Level: {level}] ---", flush=True)
            for a0, a1 in itertools.product(agents, repeat=2):
                scores: list[float] = []
                dishes_list: list[int] = []
                collisions_list: list[int] = []

                for trial in range(k):
                    seed = 1000 + trial * 37
                    score, dishes, collisions = run_single_episode(
                        agent_0_type=a0,
                        agent_1_type=a1,
                        layout_name=level,
                        horizon=horizon,
                        seed=seed,
                    )
                    completed_episodes += 1
                    scores.append(score)
                    dishes_list.append(dishes)
                    collisions_list.append(collisions)

                    rows.append({
                        "agent_0": a0,
                        "agent_1": a1,
                        "level": level,
                        "trial": trial + 1,
                        "score": score,
                        "dishes": dishes,
                        "collisions": collisions,
                    })

                    progress = f"[{completed_episodes}/{total_episodes}]"
                    print(
                        f"{progress} {a0} x {a1} ({level}) Trial {trial+1}/{k}: Score = {score:.1f}, Dishes = {dishes}",
                        flush=True,
                    )

                mean_score = np.mean(scores)
                std_score = np.std(scores)
                mean_dishes = np.mean(dishes_list)
                is_self_play = (a0 == a1)
                play_type = "Self-Play" if is_self_play else "Cross-Play"

                print(
                    f"    Summary {a0:>8} x {a1:<8} ({play_type:>10}): "
                    f"Mean Score = {mean_score:5.1f} +/- {std_score:4.1f} | "
                    f"Avg Dishes = {mean_dishes:3.1f}",
                    flush=True,
                )

    # Sort rows to ensure deterministic ordering: level, agent_0, agent_1, trial
    level_order = {lvl: i for i, lvl in enumerate(levels)}
    agent_order = {ag: i for i, ag in enumerate(agents)}
    rows.sort(
        key=lambda r: (
            level_order.get(r["level"], 999),
            agent_order.get(r["agent_0"], 999),
            agent_order.get(r["agent_1"], 999),
            r["trial"],
        )
    )

    # Print summary table across layouts
    print("\n" + "=" * 60, flush=True)
    print("Cross-Play Macro Summary (Mean Score across trials):", flush=True)
    for lvl in levels:
        print(f"\n--- Level: {lvl} ---", flush=True)
        for a0 in agents:
            pair_strs = []
            for a1 in agents:
                pair_scores = [
                    r["score"]
                    for r in rows
                    if r["level"] == lvl and r["agent_0"] == a0 and r["agent_1"] == a1
                ]
                mean_s = np.mean(pair_scores) if pair_scores else 0.0
                pair_strs.append(f"{a1}:{mean_s:.1f}")
            print(f"  {a0:<14} -> " + ", ".join(pair_strs), flush=True)

    # Save to CSV
    fieldnames = ["agent_0", "agent_1", "level", "trial", "score", "dishes", "collisions"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    elapsed = time.time() - start_time
    print("\n" + "=" * 60, flush=True)
    print(f"Benchmark Complete in {elapsed:.1f}s ({total_episodes / max(elapsed, 0.001):.1f} eps/sec)", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
