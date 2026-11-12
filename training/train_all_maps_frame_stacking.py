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

from training.train_agent import train_multi_layout


def main() -> None:
    start_total = time.time()
    print("=" * 70)
    print("ALL-MAPS FRAME-STACKED AGENT TRAINING PIPELINE")
    print("Layouts: cramped_room, asymmetric_advantages, coordination_ring,")
    print("         forced_coordination, counter_circuit_o_1order")
    print("Iterations per layout: 650")
    print("Workers: 4")
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
        num_workers=4,
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
        num_workers=4,
        out_dir=REPO_ROOT / "models" / "frppovgreedy",
    )
    print(f"\n[PHASE 2 COMPLETE] frppovgreedy finished in {(time.time() - t1) / 60:.1f} minutes.\n")

    total_time = (time.time() - start_total) / 60
    print("=" * 70)
    print(f"ALL-MAPS TRAINING PIPELINE COMPLETE IN {total_time:.1f} MINUTES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
