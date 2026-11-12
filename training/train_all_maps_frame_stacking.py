"""Multi-Map Training Pipeline for Frame-Stacked Agents (frgreedy & frppovgreedy).

Trains DeepMind k=4 frame-stacked PPO agents across all 5 benchmark layouts:
1. cramped_room
2. asymmetric_advantages
3. coordination_ring
4. forced_coordination
5. counter_circuit_o_1order

Agents Trained:
- frgreedy: PPO trained with Greedy heuristic partner (100% Greedy)
- frppovgreedy: PPO trained with Self-Play (100% PPO Self-Play)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training.train_agent import train_multi_layout, train_single_layout
from training.checkpoint_policy import load_checkpoint_policy
import shutil


def main() -> None:
    start_total = time.time()
    print("=" * 70)
    print("ALL-MAPS FRAME-STACKED AGENT TRAINING PIPELINE")
    print("Layouts: cramped_room, asymmetric_advantages, coordination_ring,")
    print("         forced_coordination, counter_circuit_o_1order")
    print("Iterations per layout: 650")
    print("Workers: 6")
    print("=" * 70)

    # 1. Train frgreedy across all 5 layouts
    print("\n" + "#" * 70)
    print("# [PHASE 1/2] TRAINING frgreedy (PPO + Greedy Partner) ACROSS ALL MAPS")
    print("#" * 70 + "\n")
    t0 = time.time()
    train_multi_layout(
        recipe="ppo_greedy",
        partner_type="greedy",
        partner_prob=1.0,
        num_iters=650,
        num_workers=6,
        out_dir=REPO_ROOT / "models" / "frgreedy",
    )
    print(f"\n[PHASE 1 COMPLETE] frgreedy finished in {(time.time() - t0) / 60:.1f} minutes.\n")

    # 2. Train frppovgreedy across all 5 layouts
    print("\n" + "#" * 70)
    print("# [PHASE 2/2] TRAINING frppovgreedy (PPO Self-Play) ACROSS ALL MAPS")
    print("#" * 70 + "\n")
    t1 = time.time()
    train_multi_layout(
        recipe="ppo_sp",
        partner_type="ppo",
        partner_prob=0.0,
        num_iters=650,
        num_workers=6,
        out_dir=REPO_ROOT / "models" / "frppovgreedy",
    )
    print(f"\n[PHASE 2 COMPLETE] frppovgreedy finished in {(time.time() - t1) / 60:.1f} minutes.\n")

    # 3. Verify all layouts in frppovgreedy have native 104-channel frame stacking
    layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring", "forced_coordination", "counter_circuit_o_1order"]
    for l_idx, l_name in enumerate(layouts, 1):
        l_dir = REPO_ROOT / "models" / "frppovgreedy" / l_name
        ckpts = sorted(l_dir.glob("checkpoint_*"))
        if ckpts:
            latest = ckpts[-1]
            c_files = [f for f in latest.glob("checkpoint-*") if not f.name.endswith(".tune_metadata")]
            if c_files:
                try:
                    pol = load_checkpoint_policy(c_files[0])
                    ch = pol.model.base_model.input_shape[-1]
                    if ch != 104:
                        print(f"\n[Channel Check] {l_name} has {ch} channels (expected 104). Retraining with 104 channels...")
                        shutil.rmtree(l_dir, ignore_errors=True)
                        train_single_layout(
                            recipe="ppo_sp",
                            layout_name=l_name,
                            partner_type="ppo",
                            partner_prob=0.0,
                            num_iters=650,
                            num_workers=6,
                            out_dir=l_dir,
                            layout_index=l_idx,
                            total_layouts=len(layouts),
                            multi_layout_parent=REPO_ROOT / "models" / "frppovgreedy",
                        )
                except Exception as e:
                    print(f"[Warning] Failed channel check on {l_name}: {e}")

    total_time = (time.time() - start_total) / 60
    print("=" * 70)
    print(f"ALL-MAPS TRAINING PIPELINE COMPLETE IN {total_time:.1f} MINUTES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
