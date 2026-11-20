"""Evaluate PPO paired with frame-stacking agents across canonical pairings."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "overcooked-agent-eval", REPO_ROOT / "external/overcooked_ai/src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gui.agent_manager import create_agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


def run_episode(agent0_type: str, agent1_type: str, layout: str, seed: int, horizon: int = 400):
    random.seed(seed)
    np.random.seed(seed)
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

    agent0, _ = create_agent(agent0_type, 0, mdp, layout)
    agent1, _ = create_agent(agent1_type, 1, mdp, layout)

    agents = [agent0, agent1]
    score = partner_blocks = wall_blocks = joint_idle_steps = idle_streak = 0
    valid_tiles = set(mdp.get_valid_player_positions())

    for _ in range(horizon):
        state = env.state
        a0 = agents[0].action(state)[0]
        a1 = agents[1].action(state)[0]
        actions = (a0, a1)

        new_state, reward, done, _ = env.step(actions)
        score += reward

        unchanged = True
        for seat, act in enumerate(actions):
            pos = state.players[seat].position
            new_pos = new_state.players[seat].position
            unchanged &= (pos == new_pos)
            if act in Direction.ALL_DIRECTIONS and pos == new_pos:
                target = (pos[0] + act[0], pos[1] + act[1])
                if target in valid_tiles:
                    partner_blocks += 1
                else:
                    wall_blocks += 1

        idle_streak = idle_streak + 1 if unchanged and reward == 0 else 0
        joint_idle_steps += int(idle_streak >= 10)

        if done:
            break

    soups = score / 20.0
    return {
        "score": float(score),
        "soups": soups,
        "partner_blocks": partner_blocks,
        "wall_blocks": wall_blocks,
        "joint_idle_steps": joint_idle_steps,
    }


def evaluate_pairing(a0_type: str, a1_type: str, layout: str = "cramped_room", seeds: list[int] = [42, 43, 44, 45, 46]):
    results = []
    # Test both seat orientations
    for s in seeds:
        # Orient 1: A in seat 0, B in seat 1
        res1 = run_episode(a0_type, a1_type, layout, seed=s)
        # Orient 2: B in seat 0, A in seat 1
        res2 = run_episode(a1_type, a0_type, layout, seed=s)
        results.extend([res1, res2])

    scores = [r["score"] for r in results]
    soups = [r["soups"] for r in results]
    blocks = [r["partner_blocks"] for r in results]
    idles = [r["joint_idle_steps"] for r in results]

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "scores": scores,
        "mean_soups": float(np.mean(soups)),
        "mean_blocks": float(np.mean(blocks)),
        "mean_idles": float(np.mean(idles)),
    }


def main():
    layout = "cramped_room"
    pairings = [
        # 1. PPO (Bundled SP) baselines
        ("ppo", "ppo", "Bundled PPO + Bundled PPO (Self-Play baseline)"),
        ("ppo", "greedy", "Bundled PPO + Stateless Greedy"),
        # 2. PPO (Bundled SP) with Frame-Stacked agents
        ("ppo", "custom:models/frgreedy", "Bundled PPO + Frame-Stacked Greedy (frgreedy)"),
        ("ppo", "custom:models/frppovgreedy", "Bundled PPO + Frame-Stacked PPO (frppovgreedy)"),
        # 3. Trained 26-ch PPO with Frame-Stacked agents
        ("custom:models/ppo_v_greedy", "custom:models/frgreedy", "PPO (26-ch) + Frame-Stacked Greedy (frgreedy)"),
        ("custom:models/ppo_v_greedy", "custom:models/frppovgreedy", "PPO (26-ch) + Frame-Stacked PPO (frppovgreedy)"),
        # 4. Frame-stacked high-performance combos
        ("custom:models/frppovgreedy", "greedy", "Frame-Stacked PPO (frppovgreedy) + Stateless Greedy"),
        ("custom:models/frgreedy", "greedy", "Frame-Stacked Greedy (frgreedy) + Stateless Greedy"),
        ("custom:models/frppovgreedy", "custom:models/frppovgreedy", "Frame-Stacked PPO + Frame-Stacked PPO (Self-Play)"),
        ("custom:models/frppovgreedy", "custom:models/frgreedy", "Frame-Stacked PPO + Frame-Stacked Greedy"),
    ]

    print(f"=== Running Comprehensive Evaluation on '{layout}' ===")
    summary = {}
    for a0, a1, label in pairings:
        print(f"\nEvaluating: {label}...")
        res = evaluate_pairing(a0, a1, layout=layout, seeds=[42, 43, 44, 45, 46])
        summary[label] = res
        print(f"  --> Score: {res['mean_score']:.1f} ± {res['std_score']:.1f} | Soups: {res['mean_soups']:.1f} | Collisions: {res['mean_blocks']:.1f} | Idle: {res['mean_idles']:.1f}")

    out_file = REPO_ROOT / "results" / "eval_ppo_with_fr_cramped_room.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
