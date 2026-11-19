"""
Comprehensive Analysis Script for Option B Evaluation Matrix (Universal 9x5 Padded Agent).
Ingests results/option_b_eval_matrix.csv and generates comparative tables:
1. Universal Padded Agent (ppo_padded_all) vs Single-Layout Specialists (ppo_fr_self, ppo_new_self, ppo_sp)
2. Self-Play Performance across all 5 layouts
3. Human-Proxy Coordination (PPO + BC)
4. Greedy-Partner Coordination (PPO + Greedy)
5. Cross-play generalization matrix between the Universal Agent and Specialists
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

CSV_PATH = Path("results/option_b_eval_matrix.csv")

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]

def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def analyze(data):
    by_layout_pair = defaultdict(lambda: defaultdict(list))
    for row in data:
        lvl = row["level"]
        a0 = row["agent_0"]
        a1 = row["agent_1"]
        score = float(row["score"])
        by_layout_pair[lvl][(a0, a1)].append(score)

    def get_sym_scores(lvl, ag_a, ag_b):
        """Pooled across both seat configurations (A, B) and (B, A)."""
        if ag_a == ag_b:
            return by_layout_pair[lvl].get((ag_a, ag_a), [])
        scores_ab = by_layout_pair[lvl].get((ag_a, ag_b), [])
        scores_ba = by_layout_pair[lvl].get((ag_b, ag_a), [])
        return scores_ab + scores_ba

    def get_mean_std(scores):
        if not scores:
            return 0.0, 0.0
        return float(np.mean(scores)), float(np.std(scores))

    print("=" * 90)
    print("EXP 3 COMPREHENSIVE EVALUATION: UNIVERSAL 9x5 PADDED AGENT VS SPECIALISTS")
    print("=" * 90)

    # -------------------------------------------------------------
    # 1. Self-Play Comparison Table
    # -------------------------------------------------------------
    print("\n### 1. SELF-PLAY SCORE COMPARISON (Agent + Agent)")
    print("-" * 90)
    print(f"{'Layout':<25} | {'PPO Pretrain':<14} | {'PPO New (k=1)':<14} | {'PPO FR (k=4)':<14} | {'Universal (9x5)':<15}")
    print("-" * 90)
    
    means_pre, means_new, means_fr, means_pad = [], [], [], []
    for lvl in LAYOUTS:
        m_pre, _ = get_mean_std(by_layout_pair[lvl].get(("ppo_sp", "ppo_sp"), []))
        m_new, _ = get_mean_std(by_layout_pair[lvl].get(("ppo_new_self", "ppo_new_self"), []))
        m_fr, _  = get_mean_std(by_layout_pair[lvl].get(("ppo_fr_self", "ppo_fr_self"), []))
        m_pad, _ = get_mean_std(by_layout_pair[lvl].get(("ppo_padded_all", "ppo_padded_all"), []))
        
        means_pre.append(m_pre)
        means_new.append(m_new)
        means_fr.append(m_fr)
        means_pad.append(m_pad)
        print(f"{lvl:<25} | {m_pre:6.1f}         | {m_new:6.1f}         | {m_fr:6.1f}         | {m_pad:6.1f}")
    
    print("-" * 90)
    print(f"{'Macro Average':<25} | {np.mean(means_pre):6.1f}         | {np.mean(means_new):6.1f}         | {np.mean(means_fr):6.1f}         | {np.mean(means_pad):6.1f}")
    print("-" * 90)

    # -------------------------------------------------------------
    # 2. Human Proxy Coordination (Agent + BC)
    # -------------------------------------------------------------
    print("\n### 2. HUMAN PROXY COORDINATION (Agent + BC, Pooled Seats)")
    print("-" * 90)
    print(f"{'Layout':<25} | {'PPO Pretrain+BC':<15} | {'PPO New+BC':<15} | {'PPO FR+BC':<15} | {'Universal+BC':<15}")
    print("-" * 90)
    
    bc_pre, bc_new, bc_fr, bc_pad = [], [], [], []
    for lvl in LAYOUTS:
        m_pre, _ = get_mean_std(get_sym_scores(lvl, "ppo_sp", "bc"))
        m_new, _ = get_mean_std(get_sym_scores(lvl, "ppo_new_self", "bc"))
        m_fr, _  = get_mean_std(get_sym_scores(lvl, "ppo_fr_self", "bc"))
        m_pad, _ = get_mean_std(get_sym_scores(lvl, "ppo_padded_all", "bc"))
        
        bc_pre.append(m_pre)
        bc_new.append(m_new)
        bc_fr.append(m_fr)
        bc_pad.append(m_pad)
        print(f"{lvl:<25} | {m_pre:6.1f}          | {m_new:6.1f}          | {m_fr:6.1f}          | {m_pad:6.1f}")
    
    print("-" * 90)
    print(f"{'Macro Average':<25} | {np.mean(bc_pre):6.1f}          | {np.mean(bc_new):6.1f}          | {np.mean(bc_fr):6.1f}          | {np.mean(bc_pad):6.1f}")
    print("-" * 90)

    # -------------------------------------------------------------
    # 3. Heuristic Partner Coordination (Agent + Greedy)
    # -------------------------------------------------------------
    print("\n### 3. GREEDY PARTNER COORDINATION (Agent + Greedy, Pooled Seats)")
    print("-" * 90)
    print(f"{'Layout':<25} | {'PPO Pretrain+Gr':<15} | {'PPO New+Gr':<15} | {'PPO FR+Gr':<15} | {'Universal+Gr':<15}")
    print("-" * 90)
    
    gr_pre, gr_new, gr_fr, gr_pad = [], [], [], []
    for lvl in LAYOUTS:
        m_pre, _ = get_mean_std(get_sym_scores(lvl, "ppo_sp", "greedy"))
        m_new, _ = get_mean_std(get_sym_scores(lvl, "ppo_new_self", "greedy"))
        m_fr, _  = get_mean_std(get_sym_scores(lvl, "ppo_fr_self", "greedy"))
        m_pad, _ = get_mean_std(get_sym_scores(lvl, "ppo_padded_all", "greedy"))
        
        gr_pre.append(m_pre)
        gr_new.append(m_new)
        gr_fr.append(m_fr)
        gr_pad.append(m_pad)
        print(f"{lvl:<25} | {m_pre:6.1f}          | {m_new:6.1f}          | {m_fr:6.1f}          | {m_pad:6.1f}")
    
    print("-" * 90)
    print(f"{'Macro Average':<25} | {np.mean(gr_pre):6.1f}          | {np.mean(gr_new):6.1f}          | {np.mean(gr_fr):6.1f}          | {np.mean(gr_pad):6.1f}")
    print("-" * 90)

    # -------------------------------------------------------------
    # 4. Universal Agent Cross-Play with Specialists
    # -------------------------------------------------------------
    print("\n### 4. UNIVERSAL AGENT CROSS-PLAY WITH SPECIALISTS (Pooled Seats)")
    print("-" * 90)
    print(f"{'Layout':<25} | {'Univ + PPO_Pre':<15} | {'Univ + PPO_New':<15} | {'Univ + PPO_FR':<15}")
    print("-" * 90)
    
    xp_pre, xp_new, xp_fr = [], [], []
    for lvl in LAYOUTS:
        m_pre, _ = get_mean_std(get_sym_scores(lvl, "ppo_padded_all", "ppo_sp"))
        m_new, _ = get_mean_std(get_sym_scores(lvl, "ppo_padded_all", "ppo_new_self"))
        m_fr, _  = get_mean_std(get_sym_scores(lvl, "ppo_padded_all", "ppo_fr_self"))
        
        xp_pre.append(m_pre)
        xp_new.append(m_new)
        xp_fr.append(m_fr)
        print(f"{lvl:<25} | {m_pre:6.1f}          | {m_new:6.1f}          | {m_fr:6.1f}")
        
    print("-" * 90)
    print(f"{'Macro Average':<25} | {np.mean(xp_pre):6.1f}          | {np.mean(xp_new):6.1f}          | {np.mean(xp_fr):6.1f}")
    print("-" * 90)

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        sys.exit(1)
    data = load_data(CSV_PATH)
    analyze(data)
