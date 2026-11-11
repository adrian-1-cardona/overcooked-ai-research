"""Paired-seed soup-delivery comparison on the trained cramped_room layout."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
import statistics

# Establish the same dependency paths and TensorFlow mode as the training runner.
from training import train_agent
import numpy as np
from agents.greedy_symbol_search import GreedySymbolSearchAgent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from training.checkpoint_policy import load_checkpoint_policy, agent_from_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/frame_stacking/evaluation"
CHECKPOINTS = {
    "stack_greedy": ROOT / "results/frame_stacking/ppo_greedy/checkpoint_000650/checkpoint-650",
    "stack_sp": ROOT / "results/frame_stacking/ppo_sp/checkpoint_000650/checkpoint-650",
    "old_greedy": ROOT / "models/ppo_greedy/cramped_room/checkpoint_000650/checkpoint-650",
    "old_mix": ROOT / "models/ppo_v_greedy/cramped_room/checkpoint_000650/checkpoint-650",
}
PAIRINGS = [
    ("greedy", "greedy"),
    ("old_greedy", "greedy"),
    ("stack_greedy", "greedy"),
    ("stack_sp", "greedy"),
    ("stack_sp", "stack_sp"),
    ("greedy", "old_mix"),
    ("old_greedy", "old_mix"),
    ("stack_greedy", "old_mix"),
    ("stack_sp", "old_mix"),
]


def episode(names, policies, seed, horizon=400):
    random.seed(seed)
    np.random.seed(seed)
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name("cramped_room"), horizon=horizon)
    agents = []
    for seat, name in enumerate(names):
        if name == "greedy":
            agent = GreedySymbolSearchAgent()
            agent.set_mdp(env.mdp)
            agent.set_agent_index(seat)
        else:
            agent = agent_from_policy(policies[name], seat, env)
        agents.append(agent)
    score = wall_blocks = partner_blocks = joint_idle_steps = idle_streak = 0
    valid_tiles = set(env.mdp.get_valid_player_positions())
    for _ in range(horizon):
        state = env.state
        actions = tuple(agent.action(state)[0] for agent in agents)
        new_state, reward, done, _ = env.step(actions)
        score += reward
        unchanged = True
        for seat, action in enumerate(actions):
            pos = state.players[seat].position
            new_pos = new_state.players[seat].position
            unchanged &= pos == new_pos
            if action in Direction.ALL_DIRECTIONS and pos == new_pos:
                target = (pos[0] + action[0], pos[1] + action[1])
                if target in valid_tiles:
                    partner_blocks += 1
                else:
                    wall_blocks += 1
        idle_streak = idle_streak + 1 if unchanged and reward == 0 else 0
        joint_idle_steps += int(idle_streak >= 10)
        if done:
            break
    return {"score": float(score), "soups": score / 20,
            "partner_blocked_moves": partner_blocks, "wall_blocked_moves": wall_blocks,
            "joint_stationary_steps_after_10": joint_idle_steps}


def evaluate(checkpoints=CHECKPOINTS, pairings=PAIRINGS, seeds=range(20), output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    policies = {name: load_checkpoint_policy(path) for name, path in checkpoints.items()}
    rows = []
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = None
        for left, right in pairings:
            for swapped in (False, True):
                names = (right, left) if swapped else (left, right)
                for seed in seeds:
                    row = {"pairing": f"{left}+{right}", "swapped": swapped, "seed": seed,
                           **episode(names, policies, seed)}
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=row.keys())
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                print(f"Finished {left}+{right}, swapped={swapped}", flush=True)
    summary = {}
    for left, right in pairings:
        name = f"{left}+{right}"
        group = [r for r in rows if r["pairing"] == name]
        summary[name] = {"episodes": len(group), **{
            key: statistics.mean(row[key] for row in group)
            for key in ("score", "soups", "partner_blocked_moves", "wall_blocked_moves", "joint_stationary_steps_after_10")
        }, "score_stdev": statistics.stdev(row["score"] for row in group)}
    comparisons = {}
    rng = np.random.default_rng(2026)
    for new, old in (("stack_greedy+greedy", "old_greedy+greedy"),
                     ("stack_greedy+old_mix", "greedy+old_mix"),
                     ("stack_sp+old_mix", "greedy+old_mix")):
        new_rows = [r for r in rows if r["pairing"] == new]
        old_rows = [r for r in rows if r["pairing"] == old]
        if not new_rows or not old_rows:
            continue
        # Resample seeds as clusters: both seat results share a seed.
        differences = np.array([
            statistics.mean(r['score'] for r in new_rows if r['seed'] == seed) -
            statistics.mean(r['score'] for r in old_rows if r['seed'] == seed)
            for seed in seeds
        ])
        boot = rng.choice(differences, size=(10000, len(differences))).mean(axis=1)
        comparisons[f"{new} minus {old}"] = {
            "mean_score_difference": float(differences.mean()),
            "paired_seed_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
        }
    report = {"layout": "cramped_room", "horizon": 400, "seeds": list(seeds),
              "checkpoints": {k: str(v) for k, v in checkpoints.items()},
              "summary": summary, "comparisons": comparisons,
              "limits": "One training seed per model; existing checkpoints are historical baselines, not a controlled multi-seed ablation. Stationary steps can include useful interactions or cooking waits."}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    for policy in policies.values():
        policy.get_session().close()
    return report


if __name__ == "__main__":
    evaluate()
