"""Sequential training pipeline for PPO_SelfPlay and PPO_HumanModel agents."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gui.agent_database import AgentDatabase


def run_pipeline() -> None:
    print("=" * 60)
    print("STARTING SEQUENTIAL TWO-AGENT TRAINING BENCHMARK")
    print("Agent 1: PPO_SelfPlay (100% Self-Play Baseline)")
    print("Agent 2: PPO_HumanModel (50/50 PPO + BC Human Model)")
    print("=" * 60)

    # Agent 1: PPO_SelfPlay
    out_dir_sp = REPO_ROOT / "models" / "PPO_SelfPlay"
    ckpt_sp_target = out_dir_sp / "checkpoint_000500"

    if ckpt_sp_target.exists():
        print(f"\n[1/2] Agent 1 (PPO_SelfPlay) already trained. Found checkpoint: {ckpt_sp_target}")
    else:
        print("\n>>> [1/2] TRAINING AGENT 1: PPO_SelfPlay (Self-Play)")
        cmd_sp = [
            sys.executable,
            "-u",
            "-m",
            "training.train_agent",
            "--recipe",
            "ppo_sp",
            "--partner-type",
            "ppo",
            "--partner-prob",
            "0.0",
            "--layout",
            "cramped_room",
            "--iters",
            "500",
            "--num-workers",
            "4",
            "--out-dir",
            str(out_dir_sp),
        ]
        subprocess.run(cmd_sp, cwd=str(REPO_ROOT), check=True)
        print(f"\n[1/2] Agent 1 finished! Checkpoint saved to: {ckpt_sp_target}")

    # Read final reward from metadata and ensure registered
    meta_sp_file = out_dir_sp / "metadata.json"
    reward_sp = 0.0
    if meta_sp_file.is_file():
        with open(meta_sp_file, "r", encoding="utf-8") as f:
            reward_sp = json.load(f).get("reward_mean", 0.0)

    AgentDatabase.add_agent(
        name="PPO_SelfPlay",
        layout="cramped_room",
        checkpoint_path=out_dir_sp,
        agent_1="ppo",
        agent_1_pct=100,
        agent_2="ppo",
        agent_2_pct=0,
        reward_mean=reward_sp,
    )
    print(f"Registered PPO_SelfPlay in database (reward: {reward_sp}).")

    time.sleep(2)

    # Agent 2: PPO_HumanModel
    out_dir_bc = REPO_ROOT / "models" / "PPO_HumanModel"
    ckpt_bc_target = out_dir_bc / "checkpoint_000500"

    if ckpt_bc_target.exists():
        print(f"\n[2/2] Agent 2 (PPO_HumanModel) already trained. Found checkpoint: {ckpt_bc_target}")
    else:
        print("\n>>> [2/2] TRAINING AGENT 2: PPO_HumanModel (PPO + BC 50/50 Mix)")
        cmd_bc = [
            sys.executable,
            "-u",
            "-m",
            "training.train_agent",
            "--recipe",
            "ppo_bc",
            "--partner-type",
            "bc",
            "--partner-prob",
            "0.5",
            "--layout",
            "cramped_room",
            "--iters",
            "500",
            "--num-workers",
            "4",
            "--out-dir",
            str(out_dir_bc),
        ]
        subprocess.run(cmd_bc, cwd=str(REPO_ROOT), check=True)
        print(f"\n[2/2] Agent 2 finished! Checkpoint saved to: {ckpt_bc_target}")

    meta_bc_file = out_dir_bc / "metadata.json"
    reward_bc = 0.0
    if meta_bc_file.is_file():
        with open(meta_bc_file, "r", encoding="utf-8") as f:
            reward_bc = json.load(f).get("reward_mean", 0.0)

    AgentDatabase.add_agent(
        name="PPO_HumanModel",
        layout="cramped_room",
        checkpoint_path=out_dir_bc,
        agent_1="ppo",
        agent_1_pct=50,
        agent_2="bc",
        agent_2_pct=50,
        reward_mean=reward_bc,
    )
    print(f"Registered PPO_HumanModel in database (reward: {reward_bc}).")

    print("\n" + "=" * 60)
    print("ALL TWO AGENTS TRAINED AND REGISTERED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
