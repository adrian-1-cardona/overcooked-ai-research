"""Multi-Map Benchmark Training Pipeline for Overcooked-AI.

Trains benchmark-aligned agents matching Carroll et al. (NeurIPS 2019) Recurrent LSTM
architecture with DeepMind k=4 Frame Stacking:
1. ppo_fr_self (PPO Self-Play baseline with LSTM recurrence + Frame Stacking)
2. ppo_fr_bc (PPO trained alongside Behavior Cloning Human Model with LSTM recurrence + Frame Stacking)

Layouts:
- cramped_room
- asymmetric_advantages
- coordination_ring
- forced_coordination
- counter_circuit_o_1order
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

from training.train_agent import train_multi_layout, BENCHMARK_LAYOUTS


def main() -> None:
    start_total = time.time()
    print("=" * 70)
    print("BENCHMARK TRAINING PIPELINE: CARROLL ET AL. (2019) LSTM + FRAME STACKING")
    print(f"Layouts: {', '.join(BENCHMARK_LAYOUTS)}")
    print("Architecture: RllibLSTMPPOModel (CELL_SIZE=256) + Frame Stacking (104 ch)")
    print("Iterations per layout: 650")
    print("Execution: Strict 1-by-1 Layout Sequential ('1 1 1 1 1' dedicated models, 8 rollout workers)")
    print("=" * 70)

    # 1. Train ppo_fr_self across all 5 layouts
    print("\n" + "#" * 70)
    print("# [PHASE 1/2] TRAINING ppo_fr_self (PPO Self-Play + LSTM) ACROSS ALL MAPS")
    print("#" * 70 + "\n")
    t0 = time.time()
    train_multi_layout(
        recipe="ppo_sp",
        partner_type="ppo",
        partner_prob=0.0,
        num_iters=650,
        num_workers=8,
        max_concurrent=1,
        use_lstm=True,
        out_dir=REPO_ROOT / "models" / "ppo_fr_self",
    )
    print(f"\n[PHASE 1 COMPLETE] ppo_fr_self finished in {(time.time() - t0) / 60:.1f} minutes.\n")

    # 2. Train ppo_fr_bc across all 5 layouts
    print("\n" + "#" * 70)
    print("# [PHASE 2/2] TRAINING ppo_fr_bc (PPO + BC Human Model + LSTM) ACROSS ALL MAPS")
    print("#" * 70 + "\n")
    t1 = time.time()
    train_multi_layout(
        recipe="ppo_bc",
        partner_type="bc",
        partner_prob=0.5,
        num_iters=650,
        num_workers=8,
        max_concurrent=1,
        use_lstm=True,
        out_dir=REPO_ROOT / "models" / "ppo_fr_bc",
    )
    print(f"\n[PHASE 2 COMPLETE] ppo_fr_bc finished in {(time.time() - t1) / 60:.1f} minutes.\n")

    total_time = (time.time() - start_total) / 60
    print("=" * 70)
    print(f"BENCHMARK TRAINING PIPELINE COMPLETE IN {total_time:.1f} MINUTES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
