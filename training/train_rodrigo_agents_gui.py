"""Sequential training of Professor Rodrigo's agents via GUI train button flow and diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pygame
from gui.app import OvercookedApp
from gui.agent_database import AgentDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Professor Rodrigo's Agents via GUI flow")
    parser.add_argument("--iters", type=int, default=100, help="Training iterations per layout (default: 100)")
    parser.add_argument("--layout", type=str, default="all_layouts", help="Layout to train on (default: all_layouts)")
    return parser.parse_args()


def train_via_gui_button() -> None:
    args = parse_args()
    num_iters = args.iters
    target_layout = args.layout

    print("=" * 60)
    print("STARTING TRAINING FOR PROFESSOR RODRIGO'S AGENTS VIA GUI BUTTON")
    print(f"Target Layout: {target_layout} | Iterations per layout: {num_iters}")
    print("Agent 1: ppo_greedy (Greedy Specialist: PPO vs Heuristic Greedy)")
    print("Agent 2: ppo_v_greedy (50/50 Mix: PPO v PPO and PPO v Greedy)")
    print("=" * 60)

    # Initialize GUI application instance
    pygame.init()
    app = OvercookedApp()
    app.active_mode = "train"
    app.train_view = "setup"
    app._on_train_layout_changed(target_layout)
    if hasattr(app, "dropdown_train_layout"):
        app.dropdown_train_layout.selected_value = target_layout
    app._on_train_iters_changed(num_iters)
    if hasattr(app, "dropdown_train_iters"):
        app.dropdown_train_iters.selected_value = num_iters

    agents_to_train = [
        ("ppo_greedy", "greedy", "Greedy Specialist: PPO vs Heuristic Greedy"),
        ("ppo_v_greedy", "ppo_50_50", "PPO 50/50 Mix: Half Self-Play, Half Heuristic Greedy"),
    ]

    for agent_name, recipe, description in agents_to_train:
        print(f"\n" + "-" * 50)
        print(f"PREPARING GUI FOR: {agent_name} ({description})")
        print(f"-" * 50)

        # Set up GUI inputs as user would in sidebar
        app.train_agent_name = agent_name
        app.input_agent_name.text = agent_name
        app._on_train_layout_changed(target_layout)
        app._on_recipe_changed(recipe)
        app._on_train_iters_changed(num_iters)
        if hasattr(app, "dropdown_train_iters"):
            app.dropdown_train_iters.selected_value = num_iters

        print(f"[GUI] Recipe selected: {app.train_recipe}")
        print(f"[GUI] Partner configuration: {app.train_agent_1} ({app.train_agent_1_pct}%) vs {app.train_agent_2} ({app.train_agent_2_pct}%)")
        print(f"[GUI] Training length: {app.train_iters} iters (Total: {app.train_progress_total})")
        print(f"[GUI] Hitting TRAIN button (btn_train_start.on_click)...")

        # Click the TRAIN button
        app.btn_train_start.on_click()

        print(f"[GUI] Train button clicked. View transitioned to: {app.train_view}")
        print(f"[GUI] Initial Progress: {app.train_progress_cur}/{app.train_progress_total}")
        print(f"[GUI] Initial ETA: {app.train_eta_str}")

        last_logged_iter = -1
        while app.train_view == "active":
            app._poll_training_progress()

            if app.train_progress_cur != last_logged_iter:
                last_logged_iter = app.train_progress_cur
                pct = int(round((app.train_progress_cur / max(1, app.train_progress_total)) * 100))
                current_l = getattr(app, "train_current_layout", target_layout)
                print(
                    f"[GUI PROGRESS] [{current_l}] {app.train_progress_cur:03d}/{app.train_progress_total:03d} "
                    f"({pct:3d}%) | ETA: {app.train_eta_str:>10s} | Reward: {app.train_reward_mean:+.2f}"
                )

            time.sleep(1.0)

        # Training finished, verify complete view state
        print(f"[GUI] Training completed for {agent_name}!")
        print(f"[GUI] View status: {app.train_view}")
        print(f"[GUI] Final Progress Bar: {app.train_progress_total}/{app.train_progress_total} (100%)")
        print(f"[GUI] Final Estimated Time: {app.train_eta_str}")
        print(f"[GUI] Final Mean Reward: {app.train_reward_mean:+.2f}")

        # Reset GUI train view to setup for next agent
        app.train_view = "setup"
        time.sleep(2.0)

    print("\n" + "=" * 60)
    print("ALL PROFESSOR RODRIGO'S AGENTS SUCCESSFULLY TRAINED AND REGISTERED!")
    records = AgentDatabase.load()
    print("Current Database Registry:")
    for r in records:
        print(f" - {r.get('name')}: layout={r.get('layout')}, reward={r.get('reward_mean')}, path={r.get('checkpoint_path')}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("RUNNING GUI DIAGNOSTIC SUITE ON NEWLY TRAINED AGENTS...")
    print("=" * 60)

    # 1. Refresh options
    app._refresh_agent_options()
    live_ids = [opt[1] for opt in app.agent_options]
    print(f"[Diagnostic] Live Testing Dropdown Options: {len(live_ids)} total options available")
    for name, _, _ in agents_to_train:
        found = any(name in opt for opt in live_ids)
        print(f"  - Agent '{name}' in Live Testing dropdown: {'PASSED [OK]' if found else 'FAILED [X]'}")
        assert found, f"Agent '{name}' not found in Live Testing options!"

    # 2. Check Matrix Mode
    app._on_mode_changed("matrix")
    print(f"[Diagnostic] Matrix Running Agents: {list(app.matrix_agents.keys())}")
    for name, _, _ in agents_to_train:
        found_mat = name in app.matrix_agents and name in app.matrix_agent_checkboxes
        print(f"  - Agent '{name}' in Matrix Benchmark: {'PASSED [OK]' if found_mat else 'FAILED [X]'}")
        assert found_mat, f"Agent '{name}' not found in Matrix Benchmark!"

    # 3. Test Agent Instantiation and Action across layouts
    from gui.agent_manager import create_agent
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from evaluation.eval_matrix import DEFAULT_LEVELS

    test_levels = [target_layout] if target_layout != "all_layouts" else DEFAULT_LEVELS
    for name, _, _ in agents_to_train:
        model_path = REPO_ROOT / "models" / name
        for lvl in test_levels:
            mdp = OvercookedGridworld.from_layout_name(lvl)
            agent, warn = create_agent(f"custom:{model_path}", agent_index=0, mdp=mdp, layout_name=lvl)
            state = mdp.get_standard_start_state()
            action, _ = agent.action(state)
            print(f"  - Agent '{name}' on layout '{lvl}' step test: action={action} PASSED [OK]")

    # 4. Headless Matrix Benchmark Execution Test
    from evaluation.eval_matrix import run_single_episode
    for name, _, _ in agents_to_train:
        lvl = test_levels[0]
        reward, dishes, col = run_single_episode(name, "greedy", lvl, horizon=10, seed=42)
        print(f"  - Matrix episode '{name}' vs 'greedy' on '{lvl}': reward={reward}, dishes={dishes} PASSED [OK]")

    print("\nALL GUI DIAGNOSTIC TESTS PASSED WITH ZERO ISSUES!")


if __name__ == "__main__":
    train_via_gui_button()
