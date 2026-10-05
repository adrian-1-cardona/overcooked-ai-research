"""
Streamlit dashboard for visualising telemetry run CSVs from render.py
(e.g., results/last_render_run.csv or any recorded gameplay CSV).

Features:
- Metric cards (Total Steps, Final Reward, Layout)
- Cumulative Reward curve over timesteps
- Agent action breakdown (Bar charts for Agent 0 & Agent 1)
- Agent position movement telemetry
- Raw data viewer with filtering
"""
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Overcooked Gameplay Telemetry", layout="wide")

st.title("Overcooked Agent Run Telemetry Dashboard")

# Default CSV file location
DEFAULT_CSV = "overcooked-agent-eval/results/last_render_run.csv"
ALT_CSV = "results/last_render_run.csv"

# Sidebar file selector / uploader
st.sidebar.header("Data Source")

selected_path = None
if os.path.exists(DEFAULT_CSV):
    selected_path = DEFAULT_CSV
elif os.path.exists(ALT_CSV):
    selected_path = ALT_CSV

uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded CSV!")
elif selected_path and os.path.exists(selected_path):
    df = pd.read_csv(selected_path)
    st.sidebar.info(f"Loaded default run: `{selected_path}`")
else:
    st.error("No telemetry CSV found. Run a gameplay simulation first or upload a CSV!")
    st.stop()

# --- TOP LEVEL METRICS ---
col1, col2, col3, col4 = st.columns(4)

total_steps = len(df)
layout_name = df["layout_name"].iloc[0] if "layout_name" in df.columns else "Unknown"
final_score = df["cumulative_sparse_reward"].iloc[-1] if "cumulative_sparse_reward" in df.columns else 0.0
episodes = df["episode"].nunique() if "episode" in df.columns else 1

col1.metric("Layout", layout_name)
col2.metric("Total Timesteps", f"{total_steps}")
col3.metric("Final Cumulative Score", f"{final_score:.1f}")
col4.metric("Episodes", f"{episodes}")

st.markdown("---")

# --- CHARTS SECTION ---
tab1, tab2, tab3 = st.tabs(["Reward Curve", "Action Distribution", "Raw Data"])

with tab1:
    st.subheader("Cumulative Sparse Reward over Timesteps")
    if "cumulative_sparse_reward" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df, x="timestep", y="cumulative_sparse_reward", ax=ax, color="#1f77b4", linewidth=2.5)
        ax.set_title("Reward Accumulation", fontsize=12)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cumulative Score")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
    else:
        st.warning("Column 'cumulative_sparse_reward' not found in CSV.")

with tab2:
    st.subheader("Agent Action Breakdown")
    c1, c2 = st.columns(2)
    
    with c1:
        if "agent_0_action" in df.columns:
            st.markdown("### Chef 0 Actions")
            a0_counts = df["agent_0_action"].value_counts()
            fig0, ax0 = plt.subplots(figsize=(5, 3.5))
            sns.barplot(x=a0_counts.index, y=a0_counts.values, palette="Blues_d", ax=ax0)
            ax0.set_ylabel("Count")
            plt.xticks(rotation=30)
            st.pyplot(fig0)

    with c2:
        if "agent_1_action" in df.columns:
            st.markdown("### Chef 1 Actions")
            a1_counts = df["agent_1_action"].value_counts()
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            sns.barplot(x=a1_counts.index, y=a1_counts.values, palette="Oranges_d", ax=ax1)
            ax1.set_ylabel("Count")
            plt.xticks(rotation=30)
            st.pyplot(fig1)

with tab3:
    st.subheader("Telemetry Data Table")
    st.dataframe(df, use_container_width=True)

st.caption("Overcooked-AI Telemetry Dashboard")
