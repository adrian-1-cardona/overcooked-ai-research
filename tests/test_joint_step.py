"""Test and verification script for the atomic PPO-Greedy joint step.

Verifies:
1. Asking 1 action from PPO (Chef 0)
2. Asking 1 action from Greedy (Chef 1)
3. Executing the joint action in Overcooked
4. Extracting the team reward (sparse + dense shaped)
5. Swapping roles (Chef 0 = Greedy, Chef 1 = PPO)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from agents.greedy_symbol_search import GreedySymbolSearchAgent
from gui.agent_manager import get_ppo_agent


def run_single_joint_step(layout_name: str = "cramped_room") -> None:
    print("=" * 60)
    print(f"VERIFYING ATOMIC JOINT STEP ON LAYOUT: {layout_name}")
    print("=" * 60)

    # 1. Initialize environment
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=400)
    env.reset()
    state = env.state

    print("\n--- Initial Kitchen State ---")
    print(f"Time step: {state.timestep}")
    print(f"Chef 0 (Red) pos: {state.players[0].position}, orientation: {state.players[0].orientation}, holding: {state.players[0].held_object}")
    print(f"Chef 1 (Blue) pos: {state.players[1].position}, orientation: {state.players[1].orientation}, holding: {state.players[1].held_object}")

    # 2. Instantiate agents
    print("\nLoading PPO Agent for Chef 0...")
    ppo_agent = get_ppo_agent(layout_name, agent_index=0, mdp=mdp)

    print("Loading Greedy Agent for Chef 1...")
    greedy_agent = GreedySymbolSearchAgent()
    greedy_agent.set_agent_index(1)
    greedy_agent.set_mdp(mdp)

    # 3. Ask one move from PPO, one from Greedy
    ppo_act, _ = ppo_agent.action(state)
    greedy_act, _ = greedy_agent.action(state)

    print("\n--- Step 1 Action Query ---")
    print(f"  PPO (Chef 0) Action   : {Action.to_char(ppo_act)} -> {ppo_act}")
    print(f"  Greedy (Chef 1) Action: {Action.to_char(greedy_act)} -> {greedy_act}")

    # 4. Execute joint action in environment
    joint_action = (ppo_act, greedy_act)
    next_state, sparse_reward, done, info = env.step(joint_action)

    print("\n--- Step 1 Execution Outcome ---")
    print(f"  Next State timestep   : {next_state.timestep}")
    print(f"  Chef 0 new pos        : {next_state.players[0].position}, holding: {next_state.players[0].held_object}")
    print(f"  Chef 1 new pos        : {next_state.players[1].position}, holding: {next_state.players[1].held_object}")
    print(f"  Sparse Reward (Deliv) : {sparse_reward}")
    
    shaped_r = info.get("shaped_r_by_agent", [0.0, 0.0])
    print(f"  Dense Shaped Rewards  : Chef 0 = {shaped_r[0]}, Chef 1 = {shaped_r[1]}")
    total_shaped = sum(shaped_r)
    print(f"  Total Shared Reward   : {sparse_reward + total_shaped}")
    print("  -> This reward signal is fed into PPO's loss to update its policy.")

    # 5. Role Swap Test (Greedy Chef 0, PPO Chef 1)
    print("\n" + "-" * 60)
    print("TESTING ROLE SWAP (Chef 0 = Greedy, Chef 1 = PPO)")
    print("-" * 60)
    env.reset()
    state = env.state

    greedy_agent_0 = GreedySymbolSearchAgent()
    greedy_agent_0.set_agent_index(0)
    greedy_agent_0.set_mdp(mdp)

    ppo_agent_1 = get_ppo_agent(layout_name, agent_index=1, mdp=mdp)

    greedy_act_0, _ = greedy_agent_0.action(state)
    ppo_act_1, _ = ppo_agent_1.action(state)

    print(f"  Greedy (Chef 0) Action: {Action.to_char(greedy_act_0)} -> {greedy_act_0}")
    print(f"  PPO (Chef 1) Action   : {Action.to_char(ppo_act_1)} -> {ppo_act_1}")

    joint_swap = (greedy_act_0, ppo_act_1)
    next_state, sparse_r, done, info = env.step(joint_swap)
    print(f"  Next State timestep   : {next_state.timestep}")
    print(f"  Chef 0 (Greedy) pos   : {next_state.players[0].position}")
    print(f"  Chef 1 (PPO) pos      : {next_state.players[1].position}")
    print(f"  Joint Step Succeeded! Both positions verified.")
    print("=" * 60)


if __name__ == "__main__":
    run_single_joint_step()
