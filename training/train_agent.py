"""Training runner for PPO agents with custom partner recipes (Greedy Specialist, 50/50 Mix, Self-Play)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import gym as gymnasium
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.result import DEFAULT_RESULTS_DIR

from human_aware_rl.ppo.ppo_rllib import RllibPPOModel
from human_aware_rl.rllib.rllib import save_trainer
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from training.env_wrapper import OvercookedPartnerMultiAgent
from training.greedy_policy import GreedyRllibPolicy

STATUS_FILE = REPO_ROOT / "results" / "training_status.json"


def update_status(data: dict[str, Any]) -> None:
    """Write current training status to a JSON file for the GUI to read."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(STATUS_FILE)
    except Exception:
        pass


def build_trainer(
    recipe: str = "ppo_greedy",
    layout_name: str = "cramped_room",
    partner_type: str = "greedy",
    partner_prob: float | None = None,
    num_workers: int = 4,
    lr: float = 2.5e-4,
    train_batch_size: int = 4800,
    sgd_minibatch_size: int = 800,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> tuple[PPOTrainer, dict[str, Any]]:
    """Build and configure the PPOTrainer with the requested partner recipe."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    if partner_prob is None:
        if recipe in ("greedy", "ppo_greedy"):
            partner_type = "greedy"
            partner_prob = 1.0
        elif recipe == "ppo_50_50":
            partner_type = "greedy"
            partner_prob = 0.5
        elif recipe in ("ppo_bc", "bc"):
            partner_type = "bc"
            partner_prob = 0.5
        elif recipe in ("ppo", "ppo_sp"):
            partner_type = "ppo"
            partner_prob = 0.0
        else:
            partner_prob = 0.5

    # Register custom PPO model
    ModelCatalog.register_custom_model("MyPPOModel", RllibPPOModel)

    env_config = {
        "layout_name": layout_name,
        "partner_type": partner_type,
        "partner_prob": partner_prob,
        "horizon": 400,
    }

    def env_creator(cfg: dict[str, Any]) -> OvercookedPartnerMultiAgent:
        layout = cfg.get("layout_name", layout_name)
        prob = cfg.get("partner_prob", partner_prob)
        ptype = cfg.get("partner_type", partner_type)
        mdp = OvercookedGridworld.from_layout_name(layout)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=cfg.get("horizon", 400))
        return OvercookedPartnerMultiAgent(
            base_env=base_env,
            reward_shaping_factor=1.0,
            reward_shaping_horizon=5000000,
            use_phi=False,
            partner_type=ptype,
            partner_prob=prob,
        )

    env_name = f"OvercookedPartnerEnv_{recipe}_{partner_type}_{int(partner_prob*100)}_{layout_name}"
    register_env(env_name, env_creator)

    # Probe spaces with a dummy environment
    probe_env = env_creator(env_config)
    ppo_obs_space = probe_env.ppo_observation_space
    greedy_obs_space = probe_env.greedy_observation_space
    action_space = probe_env.shared_action_space

    custom_model_config = {
        "use_lstm": False,
        "NUM_HIDDEN_LAYERS": 3,
        "SIZE_HIDDEN_LAYERS": 64,
        "NUM_FILTERS": 25,
        "NUM_CONV_LAYERS": 3,
        "CELL_SIZE": 256,
        "D2RL": False,
    }

    ppo_policy_config = {
        "model": {
            "custom_model": "MyPPOModel",
            "custom_model_config": custom_model_config,
        }
    }

    policies = {
        "ppo": (None, ppo_obs_space, action_space, ppo_policy_config),
    }

    if partner_prob > 0.0:
        if partner_type == "greedy":
            policies["greedy"] = (GreedyRllibPolicy, greedy_obs_space, action_space, {})
        elif partner_type == "bc":
            import pickle
            from human_aware_rl.imitation import behavior_cloning_tf2 as bc_module
            from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningPolicy
            from experiments.run_pretrained_policy_comparison import bundled_checkpoint, LegacyPredictModel

            original_bc_loader = bc_module.load_bc_model

            def load_bc_model_compat(model_dir: str, verbose: bool = False):
                model_path = Path(model_dir)
                if (model_path / "model.keras").is_file():
                    return original_bc_loader(model_dir, verbose=verbose)
                if not (model_path / "saved_model.pb").is_file():
                    raise FileNotFoundError(f"No BC model found under {model_path}")
                model = bc_module.keras.models.load_model(str(model_path), compile=False)
                session = bc_module.tf.compat.v1.keras.backend.get_session()
                model = LegacyPredictModel(model, session)
                with (model_path / "metadata.pickle").open("rb") as metadata_file:
                    bc_params = pickle.load(metadata_file)
                return model, bc_params

            bc_module.load_bc_model = load_bc_model_compat

            bc_root = bundled_checkpoint(layout_name, "BC") / "bc_params"
            bc_dirs = [p for p in bc_root.iterdir() if p.is_dir()]
            if bc_dirs:
                policies["bc"] = (
                    BehaviorCloningPolicy,
                    probe_env.bc_observation_space,
                    action_space,
                    {
                        "model_dir": str(bc_dirs[0]),
                        "stochastic": True,
                    },
                )

    def policy_mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
        if agent_id.startswith("greedy"):
            return "greedy"
        if agent_id.startswith("bc"):
            return "bc"
        return "ppo"

    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["ppo"],
    }

    rollout_fragment_length = 400
    if num_workers > 0:
        batch_multiple = num_workers * rollout_fragment_length
        if train_batch_size % batch_multiple != 0:
            import math
            train_batch_size = max(train_batch_size, math.ceil(train_batch_size / batch_multiple) * batch_multiple)

    trainer_config = {
        "env": env_name,
        "env_config": env_config,
        "multiagent": multiagent_config,
        "disable_env_checking": True,
        "num_workers": num_workers,
        "num_gpus": 0,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "num_sgd_iter": 8,
        "lr": lr,
        "gamma": 0.99,
        "lambda": 0.98,
        "clip_param": 0.1,
        "vf_loss_coeff": 1e-4,
        "kl_coeff": 0.2,
        "grad_clip": 0.1,
        "seed": seed,
        "framework": "tf",
        "eager_tracing": False,
    }

    save_path = Path(out_dir) if out_dir else REPO_ROOT / "models" / f"{recipe}_{layout_name}"
    trainer = PPOTrainer(config=trainer_config)
    return trainer, {
        "save_path": save_path,
        "recipe": recipe,
        "layout_name": layout_name,
        "partner_prob": partner_prob,
        "config": trainer_config,
    }


BENCHMARK_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]

import subprocess


def train_single_layout(
    recipe: str = "ppo_greedy",
    layout_name: str = "cramped_room",
    partner_type: str = "greedy",
    partner_prob: float | None = None,
    num_iters: int = 600,
    num_workers: int = 4,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    layout_index: int | None = None,
    total_layouts: int | None = None,
    multi_layout_parent: str | Path | None = None,
) -> Path:
    """Execute training run for a single layout."""
    print("=" * 60)
    print(f"STARTING TRAINING PIPELINE: {recipe.upper()}")
    print(f"  Layout: {layout_name}")
    print(f"  Partner Type: {partner_type}")
    print(f"  Partner Prob: {partner_prob}")
    print(f"  Iterations: {num_iters}")
    print(f"  Workers: {num_workers}")
    if layout_index and total_layouts:
        print(f"  Multi-Layout Mode: Step {layout_index}/{total_layouts}")
    print("=" * 60)

    save_path = Path(out_dir) if out_dir else REPO_ROOT / "models" / f"{recipe}_{layout_name}"
    save_path.mkdir(parents=True, exist_ok=True)

    is_multi = layout_index is not None and total_layouts is not None
    overall_total_iters = (total_layouts * num_iters) if is_multi else num_iters
    initial_eta = overall_total_iters * 2

    status_layout = "all_layouts" if is_multi else layout_name
    curr_overall = ((layout_index - 1) * num_iters) if is_multi else 0

    update_status({
        "status": "initializing",
        "recipe": recipe,
        "layout": status_layout,
        "current_layout": layout_name,
        "current_iter": curr_overall,
        "total_iters": overall_total_iters,
        "reward_mean": 0.0,
        "percent": round((curr_overall / max(1, overall_total_iters)) * 100.0, 1),
        "eta_seconds": initial_eta,
    })

    if dry_run:
        print(f"[Dry Run] Verifying trainer initialization for {layout_name}...")
        trainer, meta = build_trainer(
            recipe=recipe,
            layout_name=layout_name,
            partner_type=partner_type,
            partner_prob=partner_prob,
            num_workers=0,
            out_dir=save_path,
        )
        print(f"[Dry Run] Trainer for {layout_name} passed!")
        update_status({
            "status": "ready",
            "recipe": recipe,
            "layout": status_layout,
            "current_layout": layout_name,
            "current_iter": curr_overall,
            "total_iters": overall_total_iters,
            "reward_mean": 0.0,
            "percent": 0.0,
            "eta_seconds": 0,
        })
        return save_path

    trainer, meta = build_trainer(
        recipe=recipe,
        layout_name=layout_name,
        partner_type=partner_type,
        partner_prob=partner_prob,
        num_workers=num_workers,
        out_dir=save_path,
    )

    start_time = time.time()
    iter_durations: list[float] = []

    for iteration in range(1, num_iters + 1):
        iter_start = time.time()
        res = trainer.train()
        reward_mean = res.get("episode_reward_mean", 0.0) or 0.0
        iter_time = time.time() - iter_start
        iter_durations.append(iter_time)

        window = iter_durations[-5:] if len(iter_durations) >= 5 else iter_durations
        avg_iter_time = sum(window) / len(window)

        if is_multi:
            overall_current_iter = (layout_index - 1) * num_iters + iteration
            overall_remaining_iters = overall_total_iters - overall_current_iter
            eta_seconds = max(0, int(overall_remaining_iters * avg_iter_time))
            pct = round((overall_current_iter / overall_total_iters) * 100.0, 1)

            print(
                f"[{layout_name} {iteration:03d}/{num_iters:03d}] "
                f"Overall: {overall_current_iter:03d}/{overall_total_iters:03d} ({pct}%) | "
                f"Reward: {reward_mean:.2f} | ETA: {eta_seconds}s ({avg_iter_time:.1f}s/iter)"
            )

            update_status({
                "status": "training",
                "recipe": recipe,
                "layout": "all_layouts",
                "current_layout": layout_name,
                "layout_index": layout_index,
                "total_layouts": total_layouts,
                "current_iter": overall_current_iter,
                "total_iters": overall_total_iters,
                "layout_iter": iteration,
                "layout_total_iters": num_iters,
                "reward_mean": round(float(reward_mean), 2),
                "percent": pct,
                "eta_seconds": eta_seconds,
            })
        else:
            remaining_iters = num_iters - iteration
            eta_seconds = max(0, int(remaining_iters * avg_iter_time))
            pct = round((iteration / num_iters) * 100.0, 1)

            print(f"[{iteration:03d}/{num_iters:03d}] {pct}% | Reward: {reward_mean:.2f} | ETA: {eta_seconds}s ({avg_iter_time:.1f}s/iter)")

            update_status({
                "status": "training",
                "recipe": recipe,
                "layout": layout_name,
                "current_iter": iteration,
                "total_iters": num_iters,
                "reward_mean": round(float(reward_mean), 2),
                "percent": pct,
                "eta_seconds": eta_seconds,
            })

    # Save final checkpoint for this layout
    checkpoint_dir = trainer.save(str(save_path))
    print(f"\nLayout {layout_name} training complete! Checkpoint saved to: {checkpoint_dir}")

    meta = {
        "recipe": recipe,
        "layout_name": layout_name,
        "partner_type": partner_type,
        "partner_prob": partner_prob,
        "current_iter": num_iters,
        "total_iters": num_iters,
        "reward_mean": round(float(reward_mean), 2),
        "checkpoint_dir": str(checkpoint_dir),
    }
    with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not is_multi:
        update_status({
            "status": "completed",
            "recipe": recipe,
            "layout": layout_name,
            "current_iter": num_iters,
            "total_iters": num_iters,
            "reward_mean": round(float(reward_mean), 2),
            "percent": 100.0,
            "eta_seconds": 0,
            "checkpoint_dir": str(checkpoint_dir),
        })

    if ray.is_initialized():
        ray.shutdown()

    return Path(checkpoint_dir)


def train_multi_layout(
    recipe: str = "ppo_greedy",
    partner_type: str = "greedy",
    partner_prob: float | None = None,
    num_iters: int = 600,
    num_workers: int = 4,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Train an agent across all 5 benchmark layouts sequentially in clean processes."""
    save_path = Path(out_dir) if out_dir else REPO_ROOT / "models" / f"{recipe}_all_layouts"
    save_path.mkdir(parents=True, exist_ok=True)

    layouts = BENCHMARK_LAYOUTS
    total_layouts = len(layouts)
    overall_total_iters = total_layouts * num_iters
    reward_means: dict[str, float] = {}

    initial_eta = overall_total_iters * 2
    update_status({
        "status": "training",
        "recipe": recipe,
        "layout": "all_layouts",
        "current_layout": layouts[0],
        "layout_index": 1,
        "total_layouts": total_layouts,
        "current_iter": 0,
        "total_iters": overall_total_iters,
        "reward_mean": 0.0,
        "percent": 0.0,
        "eta_seconds": initial_eta,
    })

    if dry_run:
        print("[Dry Run] Verifying multi-layout pipeline for all 5 layouts...")
        for l in layouts:
            train_single_layout(
                recipe=recipe,
                layout_name=l,
                partner_type=partner_type,
                partner_prob=partner_prob,
                num_iters=1,
                num_workers=0,
                out_dir=save_path / l,
                dry_run=True,
                layout_index=1,
                total_layouts=total_layouts,
            )
        print("[Dry Run] Multi-layout dry run passed successfully!")
        return save_path

    for l_idx, target_layout in enumerate(layouts, 1):
        print("\n" + "=" * 60)
        print(f"[{l_idx}/{total_layouts}] TRAINING BENCHMARK LAYOUT: {target_layout}")
        print("=" * 60)

        layout_out_dir = save_path / target_layout
        venv_python = REPO_ROOT / "overcooked-agent-eval" / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.is_file() else sys.executable
        cmd = [
            python_bin,
            "-u",
            "-m", "training.train_agent",
            "--recipe", recipe,
            "--partner-type", partner_type,
            *(["--partner-prob", str(partner_prob)] if partner_prob is not None else []),
            "--layout", target_layout,
            "--iters", str(num_iters),
            "--num-workers", str(num_workers),
            "--out-dir", str(layout_out_dir),
            "--layout-index", str(l_idx),
            "--total-layouts", str(total_layouts),
            "--multi-layout-parent", str(save_path),
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

        meta_file = layout_out_dir / "metadata.json"
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                reward_means[target_layout] = json.load(f).get("reward_mean", 0.0)
        else:
            reward_means[target_layout] = 0.0

    overall_avg_reward = round(sum(reward_means.values()) / max(1, len(reward_means)), 2)

    root_meta = {
        "recipe": recipe,
        "layout_name": "all_layouts",
        "layouts": layouts,
        "partner_type": partner_type,
        "partner_prob": partner_prob,
        "current_iter": overall_total_iters,
        "total_iters": overall_total_iters,
        "reward_mean": overall_avg_reward,
        "reward_means": reward_means,
        "checkpoint_dir": str(save_path),
    }
    with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(root_meta, f, indent=2)

    update_status({
        "status": "completed",
        "recipe": recipe,
        "layout": "all_layouts",
        "current_layout": "all_layouts",
        "current_iter": overall_total_iters,
        "total_iters": overall_total_iters,
        "reward_mean": overall_avg_reward,
        "percent": 100.0,
        "eta_seconds": 0,
        "checkpoint_dir": str(save_path),
    })

    print("\n" + "=" * 60)
    print(f"ALL-MAPS TRAINING COMPLETE FOR {recipe.upper()}!")
    print(f"Average Reward Across 5 Layouts: {overall_avg_reward:+.2f}")
    print(f"Per-Layout Rewards: {reward_means}")
    print(f"Checkpoints Saved To: {save_path}")
    print("=" * 60)

    return save_path


def train(
    recipe: str = "ppo_greedy",
    layout_name: str = "cramped_room",
    partner_type: str = "greedy",
    partner_prob: float | None = None,
    num_iters: int = 600,
    num_workers: int = 4,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    layout_index: int | None = None,
    total_layouts: int | None = None,
    multi_layout_parent: str | Path | None = None,
) -> Path:
    """Execute training run supporting single-layout or all-maps training."""
    if layout_name in ("all", "all_layouts", "all_maps"):
        return train_multi_layout(
            recipe=recipe,
            partner_type=partner_type,
            partner_prob=partner_prob,
            num_iters=num_iters,
            num_workers=num_workers,
            out_dir=out_dir,
            dry_run=dry_run,
        )
    return train_single_layout(
        recipe=recipe,
        layout_name=layout_name,
        partner_type=partner_type,
        partner_prob=partner_prob,
        num_iters=num_iters,
        num_workers=num_workers,
        out_dir=out_dir,
        dry_run=dry_run,
        layout_index=layout_index,
        total_layouts=total_layouts,
        multi_layout_parent=multi_layout_parent,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO Overcooked agents with partner recipes.")
    parser.add_argument(
        "--recipe",
        type=str,
        choices=["ppo", "ppo_sp", "ppo_bc", "greedy", "ppo_greedy", "ppo_50_50", "custom"],
        default="ppo_greedy",
        help="Training recipe: ppo (Standard PPO), ppo_sp (Self-Play), ppo_bc (BC human model), greedy (Greedy Specialist), ppo_50_50 (50/50 mix), or custom.",
    )
    parser.add_argument(
        "--partner-type",
        type=str,
        default="greedy",
        choices=["greedy", "bc", "ppo", "random"],
        help="Partner type for rollouts.",
    )
    parser.add_argument(
        "--partner-prob",
        type=float,
        default=None,
        help="Probability of sampling partner agent (e.g. 0.5 for 50/50).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="cramped_room",
        help="Layout name (default: cramped_room, or 'all_layouts' for all benchmark maps)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=600,
        help="Number of training iterations (default: 600)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel rollout workers (default: 4)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for saved model checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize and verify trainer without running full training iterations.",
    )
    parser.add_argument(
        "--layout-index",
        type=int,
        default=None,
        help="Layout index when running in multi-layout mode.",
    )
    parser.add_argument(
        "--total-layouts",
        type=int,
        default=None,
        help="Total number of layouts in multi-layout mode.",
    )
    parser.add_argument(
        "--multi-layout-parent",
        type=str,
        default=None,
        help="Parent directory for multi-layout run.",
    )
    args = parser.parse_args()

    train(
        recipe=args.recipe,
        layout_name=args.layout,
        partner_type=args.partner_type,
        partner_prob=args.partner_prob,
        num_iters=args.iters,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        layout_index=args.layout_index,
        total_layouts=args.total_layouts,
        multi_layout_parent=args.multi_layout_parent,
    )


if __name__ == "__main__":
    main()
