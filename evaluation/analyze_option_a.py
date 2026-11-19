"""
Analysis script for Option A evaluation matrix.
Ingests results/option_a_eval_matrix.csv and generates comparative tables
matching Carroll et al. (NeurIPS 2019) Figure 4, isolating:
1. Baseline calibration: PPO_pretrain vs PPO_new (Exp 1 sanity check)
2. Frame stacking effect: PPO_new (k=1) vs PPO_fr (k=4) (Exp 2 ablation)
3. Cross-play generalization with Human Proxy (BC) and Greedy
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

CSV_PATH = Path("results/option_a_eval_matrix.csv")

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]

# Published Carroll et al. (NeurIPS 2019) Figure 4 approximate benchmark scores
CARROLL_FIG4 = {
    "cramped_room": {"PPO_SP+PPO_SP": 240, "PPO_SP+BC": 160, "PPO_BC+BC": 220},
    "asymmetric_advantages": {"PPO_SP+PPO_SP": 420, "PPO_SP+BC": 200, "PPO_BC+BC": 300},
    "coordination_ring": {"PPO_SP+PPO_SP": 230, "PPO_SP+BC": 10, "PPO_BC+BC": 180},
    "forced_coordination": {"PPO_SP+PPO_SP": 180, "PPO_SP+BC": 20, "PPO_BC+BC": 140},
    "counter_circuit_o_1order": {"PPO_SP+PPO_SP": 160, "PPO_SP+BC": 20, "PPO_BC+BC": 120},
}


def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze(data):
    # layout -> (a0, a1) -> list of scores
    by_layout_pair = defaultdict(lambda: defaultdict(list))
    for row in data:
        lvl = row["level"]
        a0 = row["agent_0"]
        a1 = row["agent_1"]
        score = float(row["score"])
        by_layout_pair[lvl][(a0, a1)].append(score)

    def get_sym_scores(lvl, ag_a, ag_b):
        """Get scores pooled across both seat configurations (A, B) and (B, A)."""
        if ag_a == ag_b:
            return by_layout_pair[lvl].get((ag_a, ag_a), [])
        scores_ab = by_layout_pair[lvl].get((ag_a, ag_b), [])
        scores_ba = by_layout_pair[lvl].get((ag_b, ag_a), [])
        return scores_ab + scores_ba

    def get_mean_std(scores):
        if not scores:
            return 0.0, 0.0
        return np.mean(scores), np.std(scores)

    print("=" * 80)
    print("EXP 1 & EXP 2 EVALUATION: CARROLL ET AL. (2019) BENCHMARK COMPARISON")
    print("=" * 80)

    # 1. Self-Play Table
    print("\n### 1. SELF-PLAY PERFORMANCE (Agent + Agent)")
    print("-" * 80)
    header = f"{'Layout':<26} | {'PPO Pretrain':<14} | {'PPO New (k=1)':<14} | {'PPO FR (k=4)':<14} | {'FrameStack Delta':<16}"
    print(header)
    print("-" * 80)

    macro_pre = []
    macro_new = []
    macro_fr = []

    for lvl in LAYOUTS:
        s_pre = by_layout_pair[lvl].get(("ppo_sp", "ppo_sp"), [])
        s_new = by_layout_pair[lvl].get(("ppo_new_self", "ppo_new_self"), [])
        s_fr = by_layout_pair[lvl].get(("ppo_fr_self", "ppo_fr_self"), [])

        m_pre, _ = get_mean_std(s_pre)
        m_new, _ = get_mean_std(s_new)
        m_fr, _ = get_mean_std(s_fr)

        delta = m_fr - m_new
        delta_str = f"{delta:+6.1f} ({delta/max(m_new, 1)*100:+5.1f}%)" if m_new > 0 else f"{delta:+6.1f}"

        macro_pre.append(m_pre)
        macro_new.append(m_new)
        macro_fr.append(m_fr)

        print(f"{lvl:<26} | {m_pre:6.1f}         | {m_new:6.1f}         | {m_fr:6.1f}         | {delta_str:<16}")

    print("-" * 80)
    m_avg_pre = np.mean(macro_pre)
    m_avg_new = np.mean(macro_new)
    m_avg_fr = np.mean(macro_fr)
    avg_delta = m_avg_fr - m_avg_new
    avg_delta_str = f"{avg_delta:+6.1f} ({avg_delta/max(m_avg_new, 1)*100:+5.1f}%)"
    print(f"{'OVERALL AVERAGE':<26} | {m_avg_pre:6.1f}         | {m_avg_new:6.1f}         | {m_avg_fr:6.1f}         | {avg_delta_str:<16}")
    print("-" * 80)

    # 2. Cross-Play with Human Proxy (BC) Table
    print("\n### 2. HUMAN PROXY CROSS-PLAY (Agent + H_Proxy / BC, Symmetrized across Seats)")
    print("-" * 80)
    header = f"{'Layout':<26} | {'PPO Pre + BC':<14} | {'PPO New + BC':<14} | {'PPO FR + BC':<14} | {'PPO FR_BC + BC':<14}"
    print(header)
    print("-" * 80)

    macro_bc_pre = []
    macro_bc_new = []
    macro_bc_fr = []
    macro_bc_frbc = []

    for lvl in LAYOUTS:
        s_bc_pre = get_sym_scores(lvl, "ppo_sp", "bc")
        s_bc_new = get_sym_scores(lvl, "ppo_new_self", "bc")
        s_bc_fr = get_sym_scores(lvl, "ppo_fr_self", "bc")
        s_bc_frbc = get_sym_scores(lvl, "ppo_fr_bc", "bc")

        m_bc_pre, _ = get_mean_std(s_bc_pre)
        m_bc_new, _ = get_mean_std(s_bc_new)
        m_bc_fr, _ = get_mean_std(s_bc_fr)
        m_bc_frbc, _ = get_mean_std(s_bc_frbc)

        macro_bc_pre.append(m_bc_pre)
        macro_bc_new.append(m_bc_new)
        macro_bc_fr.append(m_bc_fr)
        macro_bc_frbc.append(m_bc_frbc)

        print(f"{lvl:<26} | {m_bc_pre:6.1f}         | {m_bc_new:6.1f}         | {m_bc_fr:6.1f}         | {m_bc_frbc:6.1f}")

    print("-" * 80)
    print(f"{'OVERALL AVERAGE':<26} | {np.mean(macro_bc_pre):6.1f}         | {np.mean(macro_bc_new):6.1f}         | {np.mean(macro_bc_fr):6.1f}         | {np.mean(macro_bc_frbc):6.1f}")
    print("-" * 80)

    # 3. Cross-Play with Heuristic Greedy Table
    print("\n### 3. GREEDY CROSS-PLAY (Agent + Greedy Heuristic, Symmetrized across Seats)")
    print("-" * 80)
    header = f"{'Layout':<26} | {'PPO Pre + Grdy':<14} | {'PPO New + Grdy':<14} | {'PPO FR + Grdy':<14} | {'PPO FR_BC + Grdy':<16}"
    print(header)
    print("-" * 80)

    macro_gr_pre = []
    macro_gr_new = []
    macro_gr_fr = []
    macro_gr_frbc = []

    for lvl in LAYOUTS:
        s_gr_pre = get_sym_scores(lvl, "ppo_sp", "greedy")
        s_gr_new = get_sym_scores(lvl, "ppo_new_self", "greedy")
        s_gr_fr = get_sym_scores(lvl, "ppo_fr_self", "greedy")
        s_gr_frbc = get_sym_scores(lvl, "ppo_fr_bc", "greedy")

        m_gr_pre, _ = get_mean_std(s_gr_pre)
        m_gr_new, _ = get_mean_std(s_gr_new)
        m_gr_fr, _ = get_mean_std(s_gr_fr)
        m_gr_frbc, _ = get_mean_std(s_gr_frbc)

        macro_gr_pre.append(m_gr_pre)
        macro_gr_new.append(m_gr_new)
        macro_gr_fr.append(m_gr_fr)
        macro_gr_frbc.append(m_gr_frbc)

        print(f"{lvl:<26} | {m_gr_pre:6.1f}         | {m_gr_new:6.1f}         | {m_gr_fr:6.1f}         | {m_gr_frbc:6.1f}")

    print("-" * 80)
    print(f"{'OVERALL AVERAGE':<26} | {np.mean(macro_gr_pre):6.1f}         | {np.mean(macro_gr_new):6.1f}         | {np.mean(macro_gr_fr):6.1f}         | {np.mean(macro_gr_frbc):6.1f}")
    print("-" * 80)


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        sys.exit(1)
    d = load_data(CSV_PATH)
    analyze(d)
