"""
Interactive Streamlit Dashboard for Overcooked-AI Coordination Matrices.

Visualizes:
1. Dynamic Non-Square Cross-Play Matrices (Custom Row / Column Agent Selection).
2. Direct Side-by-Side Comparison (Frame Stack 104ch vs Baseline 26ch).
3. Preset Filters: Full Matrix, Frame Stack Focus, Heuristic Partner Tests.
4. Per-Level Heatmaps and Consolidated Macro Averages.
5. Clean Publication-Grade Agent Nomenclature (PPO[self], PPO[greedy], Greedy).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Overcooked Matrix Dashboard",
    page_icon="🍳",
    layout="wide",
)

DEFAULT_CSV = Path("results/matrix.csv")

# Standardized Publication Nomenclature
DISPLAY_NAME_MAP = {
    "frppovgreedy": "fr[self,greedy]",
    "frgreedy": "fr[greedy]",
    "ppo_v_greedy": "ppo[self,greedy]",
    "ppo_greedy": "ppo[greedy]",
    "ppo_sp": "ppo[self]",
    "greedy": "greedy",
    "random": "random",
}

REVERSE_DISPLAY_MAP = {v: k for k, v in DISPLAY_NAME_MAP.items()}


def get_display_name(raw_name: str) -> str:
    return DISPLAY_NAME_MAP.get(raw_name, raw_name)


st.title("🍳 Overcooked-AI Coordination Matrix Dashboard")
st.markdown(
    "Customizable cross-play evaluation matrix measuring teamwork and partner adaptation. "
    "Select any custom subset of **Chef 0 (Rows)**, **Chef 1 (Columns)**, and **Layouts**."
)

# -----------------------------------------------------------------------------
# Sidebar: Data & Filters
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data Source")
csv_path = st.sidebar.text_input("CSV File Path", value=str(DEFAULT_CSV))

if not Path(csv_path).exists():
    st.error(f"No results file found at `{csv_path}`. Run evaluation first!")
    st.stop()

raw_df = pd.read_csv(csv_path)

# Normalize column names for legacy CSVs
if "agent" in raw_df.columns and "agent_0" not in raw_df.columns:
    raw_df = raw_df.rename(columns={"agent": "agent_0", "partner": "agent_1"})

# Add readable display name columns
raw_df["agent_0_display"] = raw_df["agent_0"].map(get_display_name)
raw_df["agent_1_display"] = raw_df["agent_1"].map(get_display_name)

all_raw_agents = sorted(list(set(raw_df["agent_0"].unique()) | set(raw_df["agent_1"].unique())))
all_display_agents = [get_display_name(a) for a in all_raw_agents]
all_levels = sorted(raw_df["level"].unique())

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filter Presets")

preset = st.sidebar.selectbox(
    "Quick Preset",
    options=[
        "Custom Selection",
        "Full Matrix (All 6 Agents)",
        "FrameStack Focus (fr vs ppo & greedy)",
        "Curriculum Comparison (fr[self,greedy] vs ppo[self,greedy])",
        "Partner Test with greedy",
    ],
    index=0,
)

# Presets mapping
if preset == "Full Matrix (All 6 Agents)":
    default_rows = all_display_agents
    default_cols = all_display_agents
elif preset == "FrameStack Focus (fr vs ppo & greedy)":
    default_rows = [get_display_name("frppovgreedy"), get_display_name("frgreedy")]
    default_cols = [get_display_name(a) for a in ["greedy", "ppo_sp", "ppo_v_greedy", "frppovgreedy"]]
elif preset == "Curriculum Comparison (fr[self,greedy] vs ppo[self,greedy])":
    default_rows = [get_display_name("frppovgreedy"), get_display_name("ppo_v_greedy")]
    default_cols = [get_display_name(a) for a in ["greedy", "ppo_sp", "frppovgreedy", "ppo_v_greedy"]]
elif preset == "Partner Test with greedy":
    default_rows = [get_display_name(a) for a in ["frppovgreedy", "ppo_v_greedy", "ppo_sp", "greedy"]]
    default_cols = [get_display_name("greedy"), get_display_name("ppo_sp")]
else:
    default_rows = all_display_agents
    default_cols = all_display_agents

# Filter controls
st.sidebar.markdown("---")
st.sidebar.header("🔍 Custom Matrix Dimensions")

selected_rows = st.sidebar.multiselect(
    "Chef 0 (Red Hat - Rows)",
    options=all_display_agents,
    default=[a for a in default_rows if a in all_display_agents],
)

selected_cols = st.sidebar.multiselect(
    "Chef 1 (Blue Hat - Columns)",
    options=all_display_agents,
    default=[a for a in default_cols if a in all_display_agents],
)

selected_levels = st.sidebar.multiselect(
    "Kitchen Layouts",
    options=all_levels,
    default=all_levels,
)

if not selected_rows:
    st.warning("Please select at least 1 agent for Chef 0 (Rows) in the sidebar.")
    st.stop()

if not selected_cols:
    st.warning("Please select at least 1 agent for Chef 1 (Columns) in the sidebar.")
    st.stop()

if not selected_levels:
    st.warning("Please select at least 1 Layout in the sidebar.")
    st.stop()

# Filter dataframe
df = raw_df[
    (raw_df["agent_0_display"].isin(selected_rows))
    & (raw_df["agent_1_display"].isin(selected_cols))
    & (raw_df["level"].isin(selected_levels))
].copy()

# -----------------------------------------------------------------------------
# Top Metric Overview Cards
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows (Chef 0)", len(selected_rows))
m2.metric("Columns (Chef 1)", len(selected_cols))
m3.metric("Selected Layouts", len(selected_levels))
avg_filtered_score = df["score"].mean() if len(df) > 0 else 0.0
m4.metric("Mean Filtered Score", f"{avg_filtered_score:.1f}")

st.markdown("---")


# -----------------------------------------------------------------------------
# Matrix Rendering Function
# -----------------------------------------------------------------------------
def render_matrix(
    pivot_table: pd.DataFrame,
    title: str,
    subtitle: str = "",
    cbar_max: float = 250.0,
) -> None:
    # Ensure ordered according to sidebar selection
    pivot = pivot_table.reindex(index=selected_rows, columns=selected_cols).fillna(0.0)

    # Dynamic figure sizing: does not need to be square!
    n_rows = max(1, len(selected_rows))
    n_cols = max(1, len(selected_cols))
    fig_w = max(4.5, n_cols * 1.5 + 2.0)
    fig_h = max(3.5, n_rows * 0.9 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=True,
        vmin=0.0,
        vmax=max(cbar_max, float(pivot.values.max() if len(pivot.values) else 10.0)),
        linewidths=1.2,
        linecolor="#1e293b",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "Soup Delivery Score"},
    )

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Chef 1 (Blue Hat - Partner) →", fontsize=11, weight="bold", color="#1d4ed8")
    ax.set_ylabel("Chef 0 (Red Hat - Lead) →", fontsize=11, weight="bold", color="#b91c1c")

    plt.xticks(rotation=20, ha="right", fontsize=10, weight="semibold")
    plt.yticks(rotation=0, fontsize=10, weight="semibold")
    plt.tight_layout()

    st.pyplot(fig)
    if subtitle:
        st.caption(subtitle)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab_labels = (
    ["📊 Consolidated (Macro)", "⚡ Side-by-Side: fr vs ppo"]
    + [f"🗺️ {lvl}" for lvl in selected_levels]
    + ["📋 Raw Data"]
)
tabs = st.tabs(tab_labels)

# 1. Consolidated Tab
with tabs[0]:
    st.subheader("Consolidated Matrix (Macro Average Across Selected Levels)")
    st.markdown(
        "Shows average score across the selected layouts. "
        "Rows represent **Chef 0 (Red)**; Columns represent **Chef 1 (Blue)**."
    )

    macro_pivot = df.pivot_table(
        index="agent_0_display",
        columns="agent_1_display",
        values="score",
        aggfunc="mean",
    )
    render_matrix(
        macro_pivot,
        title="Consolidated Teamwork Score (Macro Mean)",
        subtitle=f"Averaged across {len(selected_levels)} layout(s): {', '.join(selected_levels)}",
    )

# 2. Side-by-Side Comparison Tab: fr vs ppo Baseline
with tabs[1]:
    st.subheader("Direct Comparison: fr vs ppo Baseline")
    st.markdown(
        "Directly isolates **`fr[self,greedy]`** against the baseline **`ppo[self,greedy]`** "
        "paired with foreign partners across the selected levels."
    )

    fs_key = get_display_name("frppovgreedy")
    base_key = get_display_name("ppo_v_greedy")

    comp_df = raw_df[raw_df["level"].isin(selected_levels)].copy()

    # Compare when playing with Greedy and PPO[self]
    partner_keys = [get_display_name(p) for p in ["greedy", "ppo_sp"]]

    col_fs, col_base, col_delta = st.columns(3)

    fs_scores = comp_df[
        (comp_df["agent_0_display"] == fs_key)
        & (comp_df["agent_1_display"].isin(partner_keys))
    ].groupby("agent_1_display")["score"].mean()

    base_scores = comp_df[
        (comp_df["agent_0_display"] == base_key)
        & (comp_df["agent_1_display"].isin(partner_keys))
    ].groupby("agent_1_display")["score"].mean()

    with col_fs:
        st.markdown(f"#### 🚀 {fs_key}")
        for p in partner_keys:
            val = fs_scores.get(p, 0.0)
            st.metric(f"Paired with {p}", f"{val:.1f} pts")

    with col_base:
        st.markdown(f"#### 📦 {base_key}")
        for p in partner_keys:
            val = base_scores.get(p, 0.0)
            st.metric(f"Paired with {p}", f"{val:.1f} pts")

    with col_delta:
        st.markdown("#### 📈 Frame Stack Advantage (Δ)")
        for p in partner_keys:
            v_fs = fs_scores.get(p, 0.0)
            v_base = base_scores.get(p, 0.0)
            diff = v_fs - v_base
            pct = (diff / max(v_base, 0.001)) * 100
            st.metric(
                f"Advantage vs {p}",
                f"{diff:+.1f} pts",
                delta=f"{pct:+.1f}%",
            )

    st.markdown("---")
    st.markdown("##### Detailed Head-to-Head Table")
    head_pivot = comp_df[
        (comp_df["agent_0_display"].isin([fs_key, base_key]))
        & (comp_df["agent_1_display"].isin(selected_cols))
    ].pivot_table(
        index="agent_0_display",
        columns="agent_1_display",
        values="score",
        aggfunc="mean",
    )
    st.dataframe(head_pivot.round(1), use_container_width=True)

# 3. Individual Level Tabs
for idx, lvl in enumerate(selected_levels, start=2):
    with tabs[idx]:
        st.subheader(f"Level: {lvl} Coordination Matrix")
        level_df = df[df["level"] == lvl]
        level_pivot = level_df.pivot_table(
            index="agent_0_display",
            columns="agent_1_display",
            values="score",
            aggfunc="mean",
        )
        render_matrix(
            level_pivot,
            title=f"Coordination Matrix - {lvl}",
            subtitle=f"Average score across 20 trials on {lvl}.",
        )

# 4. Raw Data Tab
with tabs[-1]:
    st.subheader("📋 Evaluation Trials Data")
    st.dataframe(
        df[["agent_0_display", "agent_1_display", "level", "trial", "score", "dishes", "collisions"]],
        use_container_width=True,
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered CSV",
        data=csv_bytes,
        file_name="filtered_matrix.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Scientific Notes Expander
# -----------------------------------------------------------------------------
with st.expander("📖 Scientific Notes & Reading Guide", expanded=False):
    st.markdown(
        """
        - **Nomenclature Guide:**
          - `fr[self,greedy]`: Multi-layout Frame-Stack policy trained on 50/50 mix of Self-Play and Greedy.
          - `fr[greedy]`: Frame-Stack policy trained 100% against Greedy partner (Phase 1).
          - `ppo[self,greedy]`: Baseline single-frame policy trained on 50/50 mix.
          - `ppo[self]`: Baseline Self-Play policy.
          - `greedy`: Algorithmic heuristic A* planner (not a neural network).
        - **Non-Square Matrices:** Rows and Columns do not need to match. You can inspect isolated 2×3 or 1×6 comparisons.
        - **Role Asymmetry:** Chef 0 (Red) starts on the left primary station; Chef 1 (Blue) starts on the right support station.
        """
    )
