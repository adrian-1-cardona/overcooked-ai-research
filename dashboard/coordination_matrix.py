"""
Streamlit dashboard that visualises the k×k coordination matrix.
It expects a CSV at `results/matrix.csv` with columns:
    agent,partner,level,score
The heat‑map shows the average score for each (agent, partner) pair across
all levels.  Hovering over a cell reveals the exact value.
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧑‍🍳 Overcooked Coordination Matrix Dashboard")

csv_path = "results/matrix.csv"
if not st.sidebar.checkbox("Show raw CSV", value=False):
    pass

if not st.file_uploader("Upload CSV (optional)", type="csv"):
    # Use default path if no upload
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read `{csv_path}`: {e}")
        st.stop()
else:
    uploaded = st.session_state.uploaded_file
    df = pd.read_csv(uploaded)

# Compute mean score over levels
pivot = df.pivot_table(index="agent", columns="partner", values="score", aggfunc="mean")

st.subheader("Average Score (higher = better teamwork)")
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
st.pyplot(fig)

st.caption("Values are averages across the three benchmark levels.")
