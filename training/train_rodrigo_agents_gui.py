"""Sequential training of Professor Rodrigo's agents via GUI train button flow."""

from __future__ import annotations

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


def train_via_gui_button() -> None:
    print("=" * 60)
    print("STARTING TRAINING FOR PROFESSOR RODRIGO'S AGENTS VIA GUI BUTTON")
    print("Agent 1: ppo_greedy (Greedy Specialist: PPO vs Heuristic Greedy)")
    print("Agent 2: ppo_v_greedy (50/50 Mix: PPO v PPO and PPO v Greedy)")
    print("=" * 60)

    # Initialize GUI application instance
    pygame.init()
    app = OvercookedApp()
    app.active_mode = "train"
    app.train_view = "setup"
    app.train_layout = "all_layouts"
    if hasattr(app, "dropdown_train_layout"):
        app.dropdown_train_layout.selected_value = "all_layouts"
    app.train_iters = 100

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
        app._on_train_layout_changed("all_layouts")
        app._on_recipe_changed(recipe)
        app.train_iters = 100

        print(f"[GUI] Recipe selected: {app.train_recipe}")
        print(f"[GUI] Partner configuration: {app.train_agent_1} ({app.train_agent_1_pct}%) vs {app.train_agent_2} ({app.train_agent_2_pct}%)")
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
                current_l = getattr(app, "train_current_layout", "all_layouts")
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


if __name__ == "__main__":
    train_via_gui_button()
