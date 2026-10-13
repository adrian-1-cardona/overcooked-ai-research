"""
Interactive Streamlit Dashboard for Overcooked-AI Coordination Matrices.

Visualizes:
1. Master Consolidated Average Heatmap across all levels.
2. Individual Per-Level Heatmaps (cramped_room, asymmetric_advantages, etc.).
3. Metric Cards: Top Teamwork Pair, Worst Failure Pair, Total Games.
4. Scientific disclaimers and notes on role asymmetry.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Overcooked Matrix Dashboard",
    layout="wide",
)

st.title("Overcooked-AI Coordination Matrix Dashboard")
st.markdown(
    "Cross-play evaluation matrix measuring $n \\times n$ teamwork across levels. "
    "Diagonal cells represent **Self-Play (SP)**; off-diagonal cells represent **Cross-Play (XP)**."
)

DEFAULT_CSV = Path("results/matrix.csv")

# Sidebar
st.sidebar.header("Data Configuration")
csv_path = st.sidebar.text_input("CSV File Path", value=str(DEFAULT_CSV))

if not Path(csv_path).exists():
    st.error(f"No results file found at `{csv_path}`. Run `python evaluation/eval_matrix.py` first!")
    st.stop()

df = pd.read_csv(csv_path)

# Normalize column names in case of legacy csv
if "agent" in df.columns and "agent_0" not in df.columns:
    df = df.rename(columns={"agent": "agent_0", "partner": "agent_1"})

agents = sorted(list(set(df["agent_0"].unique()) | set(df["agent_1"].unique())))
levels = sorted(df["level"].unique())
total_episodes = len(df)

# Top metric cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Evaluated Agents (n)", len(agents))
c2.metric("Levels Tested (m)", len(levels))
c3.metric("Total Episodes Logged", total_episodes)
avg_score = df["score"].mean()
c4.metric("Overall Average Score", f"{avg_score:.1f}")

st.markdown("---")

# Matrix plotting function
def render_heatmap(pivot_table: pd.DataFrame, title: str, subtitle: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    # Reindex to ensure square matrix with all agents
    pivot_table = pivot_table.reindex(index=agents, columns=agents)

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=True,
        linewidths=1.0,
        linecolor="#2c3e50",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Chef 1 (Partner) ->", fontsize=11, weight="bold")
    ax.set_ylabel("Chef 0 (Lead) ->", fontsize=11, weight="bold")

    st.pyplot(fig)
    if subtitle:
        st.caption(subtitle)
    plt.close(fig)


# Tabs for Consolidated vs Per-Level
tab_names = ["Consolidated (All Levels)"] + [f"Level: {lvl}" for lvl in levels]
tabs = st.tabs(tab_names)

# Tab 1: Consolidated
with tabs[0]:
    st.subheader("Consolidated Cross-Play Matrix (Macro Average)")
    st.markdown("Averages each agent pair's performance across all tested layouts:")
    
    macro_pivot = df.pivot_table(
        index="agent_0",
        columns="agent_1",
        values="score",
        aggfunc="mean",
    )
    render_heatmap(
        macro_pivot,
        title="Consolidated Teamwork Score (Mean Over All Levels)",
        subtitle="Values represent average score across all tested layouts.",
    )

# Individual Level Tabs
for idx, lvl in enumerate(levels, start=1):
    with tabs[idx]:
        st.subheader(f"Level: {lvl} Matrix")
        level_df = df[df["level"] == lvl]
        level_pivot = level_df.pivot_table(
            index="agent_0",
            columns="agent_1",
            values="score",
            aggfunc="mean",
        )
        render_heatmap(
            level_pivot,
            title=f"Coordination Matrix - {lvl}",
            subtitle=f"Average score on {lvl} across trials.",
        )

# Scientific Disclaimers & Notes
with st.expander("Scientific Notes & Interpretation Guide", expanded=False):
    st.markdown(
        """
        - **Diagonal (Self-Play):** Matchups where an agent plays with an identical clone of itself.
        - **Off-Diagonal (Cross-Play):** Matchups between two different policies.
        - **Asymmetry Notice:** Rows represent **Chef 0 (Red Hat)** and columns represent **Chef 1 (Blue Hat)**. 
          Because players spawn in different positions with different starting access, scores are **not symmetric** (i.e. `Agent A x Agent B != Agent B x Agent A`).
        - **The Self-Play Trap:** A large gap between diagonal scores (high) and off-diagonal scores (low) indicates brittle tacit conventions.
        """
    )
