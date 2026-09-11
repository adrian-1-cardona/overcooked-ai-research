"""Plot Overcooked-AI single-episode run telemetry using Matplotlib.

Reads per-timestep telemetry rows (from a CSV file or in-memory dicts) and
generates a publication-grade multi-panel performance & coordination dashboard.

Usage:
    # Plot from an existing CSV and display the window:
    python overcooked-agent-eval/experiments/plot_run.py overcooked-agent-eval/results/random_baseline_cramped_room.csv

    # Save to a specific image file without showing a GUI window (headless-friendly):
    python overcooked-agent-eval/experiments/plot_run.py overcooked-agent-eval/results/random_baseline_cramped_room.csv --output results/run_dashboard.png --no-show
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure writable matplotlib cache in sandboxed or restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.edgecolor"] = "#cccccc"
matplotlib.rcParams["axes.linewidth"] = 0.8


def parse_position(pos_val: Any) -> tuple[int, int] | None:
    """Parse a position string like '(2, 1)' or tuple into (x, y)."""
    if pos_val is None:
        return None
    if isinstance(pos_val, (tuple, list)) and len(pos_val) >= 2:
        return int(pos_val[0]), int(pos_val[1])
    try:
        parsed = ast.literal_eval(str(pos_val).strip())
        if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
            return int(parsed[0]), int(parsed[1])
    except Exception:
        pass
    return None


def parse_telemetry_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load per-timestep telemetry from CSV into typed dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"Telemetry CSV '{csv_path.name}' is empty")

    first_row = raw_rows[0]
    if "timestep" not in first_row and "step" not in first_row:
        raise ValueError(
            f"CSV '{csv_path.name}' does not appear to be timestep-level telemetry. "
            f"Found columns: {list(first_row.keys())}"
        )

    parsed_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        step = int(r.get("timestep", r.get("step", 0)))
        sparse_r = float(r.get("sparse_reward", 0.0))
        cum_sparse_r = float(r.get("cumulative_sparse_reward", 0.0))
        a0_shaped = float(r.get("agent_0_shaped_reward", 0.0))
        a1_shaped = float(r.get("agent_1_shaped_reward", 0.0))
        a0_act = str(r.get("agent_0_action", "stay")).lower()
        a1_act = str(r.get("agent_1_action", "stay")).lower()
        a0_pos = parse_position(r.get("agent_0_position"))
        a1_pos = parse_position(r.get("agent_1_position"))

        parsed_rows.append({
            "timestep": step,
            "sparse_reward": sparse_r,
            "cumulative_sparse_reward": cum_sparse_r,
            "agent_0_shaped_reward": a0_shaped,
            "agent_1_shaped_reward": a1_shaped,
            "agent_0_action": a0_act,
            "agent_1_action": a1_act,
            "agent_0_position": a0_pos,
            "agent_1_position": a1_pos,
            "layout_name": r.get("layout_name", "unknown"),
        })

    return parsed_rows


def plot_run_dashboard(
    rows: Sequence[dict[str, Any]],
    layout_name: str = "Unknown Layout",
    agent_0_label: str = "Chef 0",
    agent_1_label: str = "Chef 1",
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Generate and display/save a 4-panel performance & coordination dashboard."""
    if not rows:
        print("[WARNING] No telemetry rows to plot.")
        return None

    # Detect layout from rows if default
    if layout_name in ("Unknown Layout", "unknown"):
        for r in rows:
            if "layout_name" in r and r["layout_name"] not in ("unknown", None, ""):
                layout_name = str(r["layout_name"])
                break

    timesteps = [r["timestep"] for r in rows]
    cum_scores = [r["cumulative_sparse_reward"] for r in rows]
    sparse_rewards = [r["sparse_reward"] for r in rows]

    # Shaped rewards cumulative
    a0_shaped_raw = [r["agent_0_shaped_reward"] for r in rows]
    a1_shaped_raw = [r["agent_1_shaped_reward"] for r in rows]
    a0_shaped_cum = np.cumsum(a0_shaped_raw)
    a1_shaped_cum = np.cumsum(a1_shaped_raw)

    # Delivery events (sparse_reward > 0)
    deliveries = [(t, r["sparse_reward"]) for t, r in zip(timesteps, rows) if r["sparse_reward"] > 0]
    total_score = cum_scores[-1] if cum_scores else 0.0
    num_deliveries = len(deliveries)

    # Actions categorisation
    move_actions = {"north", "south", "east", "west"}
    a0_moves = sum(1 for r in rows if r["agent_0_action"] in move_actions)
    a0_interacts = sum(1 for r in rows if r["agent_0_action"] == "interact")
    a0_stays = sum(1 for r in rows if r["agent_0_action"] in ("stay", "(0, 0)"))

    a1_moves = sum(1 for r in rows if r["agent_1_action"] in move_actions)
    a1_interacts = sum(1 for r in rows if r["agent_1_action"] == "interact")
    a1_stays = sum(1 for r in rows if r["agent_1_action"] in ("stay", "(0, 0)"))

    total_steps = len(rows)

    # Spatial positions (parse reliably whether passed as tuples or strings)
    a0_positions: list[tuple[int, int]] = []
    for r in rows:
        p = parse_position(r.get("agent_0_position"))
        if p is not None:
            a0_positions.append(p)

    a1_positions: list[tuple[int, int]] = []
    for r in rows:
        p = parse_position(r.get("agent_1_position"))
        if p is not None:
            a1_positions.append(p)

    # Determine grid bounds
    all_positions = a0_positions + a1_positions
    if all_positions:
        max_x = max(p[0] for p in all_positions) + 1
        max_y = max(p[1] for p in all_positions) + 1
    else:
        max_x, max_y = 5, 5
    grid_w = max(5, max_x + 1)
    grid_h = max(5, max_y + 1)

    # Spatial heatmaps
    grid_a0 = np.zeros((grid_h, grid_w), dtype=float)
    grid_a1 = np.zeros((grid_h, grid_w), dtype=float)
    for x, y in a0_positions:
        if 0 <= y < grid_h and 0 <= x < grid_w:
            grid_a0[y, x] += 1
    for x, y in a1_positions:
        if 0 <= y < grid_h and 0 <= x < grid_w:
            grid_a1[y, x] += 1

    # Detect collisions (same position attempts or swaps)
    collision_steps = 0
    for r in rows:
        p0 = parse_position(r.get("agent_0_position"))
        p1 = parse_position(r.get("agent_1_position"))
        if p0 is not None and p1 is not None and p0 == p1:
            collision_steps += 1

    # =========================================================================
    # Build Figure (2x2 Dashboard)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#f9f9fb")

    # Title & Metadata Banner
    header_text = (
        f"Overcooked-AI Run Performance | Layout: {layout_name}\n"
        f"Horizon: {total_steps} steps | Total Score: {total_score:.0f} | Soups Delivered: {num_deliveries}"
    )
    fig.suptitle(header_text, fontsize=15, fontweight="bold", y=0.98, color="#1e293b")

    # -------------------------------------------------------------------------
    # Panel 1 (Top-Left): Score & Delivery Timeline
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(timesteps, cum_scores, color="#2563eb", linewidth=2.5, label="Cumulative Score")
    for d_step, d_rew in deliveries:
        ax1.axvline(x=d_step, color="#16a34a", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.scatter([d_step], [cum_scores[d_step - 1] if d_step <= len(cum_scores) else total_score],
                    color="#16a34a", s=60, zorder=5)
        ax1.text(d_step + 3, (cum_scores[d_step - 1] if d_step <= len(cum_scores) else total_score) - 5,
                 f"+{d_rew:.0f}", color="#16a34a", fontsize=9, fontweight="bold")

    ax1.set_title("Team Score & Delivery Events", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Timestep", fontsize=10)
    ax1.set_ylabel("Cumulative Sparse Reward", fontsize=10)
    ax1.set_xlim(0, max(total_steps, 1))
    ax1.set_ylim(bottom=-1, top=max(total_score * 1.15, 25.0))
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 2 (Top-Right): Shaped Rewards (Subtask Contributions)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(timesteps, a0_shaped_cum, color="#0284c7", linewidth=2, label=f"{agent_0_label} (Potting/Dishes)")
    ax2.plot(timesteps, a1_shaped_cum, color="#ea580c", linewidth=2, label=f"{agent_1_label} (Potting/Dishes)")
    ax2.set_title("Intermediate Shaped Reward Progression", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Timestep", fontsize=10)
    ax2.set_ylabel("Cumulative Shaped Reward", fontsize=10)
    ax2.set_xlim(0, max(total_steps, 1))
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Action Distribution Breakdown
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    categories = ["Movements", "Interactions", "Idle / Stay"]
    c0_counts = [a0_moves, a0_interacts, a0_stays]
    c1_counts = [a1_moves, a1_interacts, a1_stays]
    c0_pct = [c / max(total_steps, 1) * 100 for c in c0_counts]
    c1_pct = [c / max(total_steps, 1) * 100 for c in c1_counts]

    y_pos = np.arange(len(categories))
    bar_height = 0.35

    rects1 = ax3.barh(y_pos - bar_height / 2, c0_pct, bar_height, label=agent_0_label, color="#38bdf8", edgecolor="#0284c7")
    rects2 = ax3.barh(y_pos + bar_height / 2, c1_pct, bar_height, label=agent_1_label, color="#fb923c", edgecolor="#ea580c")

    # Add percentages on bars
    for rect, pct, count in zip(rects1, c0_pct, c0_counts):
        ax3.text(rect.get_width() + 1, rect.get_y() + rect.get_height() / 2,
                 f"{pct:.1f}% ({count})", va="center", fontsize=8.5, color="#1e293b")
    for rect, pct, count in zip(rects2, c1_pct, c1_counts):
        ax3.text(rect.get_width() + 1, rect.get_y() + rect.get_height() / 2,
                 f"{pct:.1f}% ({count})", va="center", fontsize=8.5, color="#1e293b")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(categories, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel("Percentage of Total Steps (%)", fontsize=10)
    ax3.set_xlim(0, 115)
    ax3.set_title("Action & Activity Breakdown", fontsize=12, fontweight="bold", pad=8)
    ax3.legend(loc="lower right", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6, axis="x")

    # -------------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Spatial Occupancy Heatmap
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    max_visits = max(float(np.max(grid_a0)), float(np.max(grid_a1)), 1.0)
    rgb_map = np.ones((grid_h, grid_w, 3), dtype=float)
    r_channel = 1.0 - 0.8 * (grid_a0 / max_visits)
    b_channel = 1.0 - 0.8 * (grid_a1 / max_visits)
    g_channel = 1.0 - 0.5 * ((grid_a0 + grid_a1) / (2 * max_visits))
    rgb_map[:, :, 0] = np.clip(b_channel, 0.1, 1.0)
    rgb_map[:, :, 1] = np.clip(g_channel, 0.1, 1.0)
    rgb_map[:, :, 2] = np.clip(r_channel, 0.1, 1.0)

    ax4.imshow(rgb_map, origin="upper", aspect="equal")

    for y in range(grid_h):
        for x in range(grid_w):
            v0 = int(grid_a0[y, x])
            v1 = int(grid_a1[y, x])
            if v0 > 0 or v1 > 0:
                ax4.text(x, y, f"{v0}|{v1}", ha="center", va="center", fontsize=7.5,
                         fontweight="bold", color="#0f172a")

    ax4.set_xticks(np.arange(-0.5, grid_w, 1), minor=True)
    ax4.set_yticks(np.arange(-0.5, grid_h, 1), minor=True)
    ax4.grid(which="minor", color="#94a3b8", linestyle="-", linewidth=1.2)
    ax4.tick_params(which="minor", size=0)
    ax4.set_xticks(range(grid_w))
    ax4.set_yticks(range(grid_h))
    ax4.set_xlabel("Grid X", fontsize=10)
    ax4.set_ylabel("Grid Y", fontsize=10)
    ax4.set_title("Tile Visits (Chef 0 | Chef 1)", fontsize=12, fontweight="bold", pad=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    saved_path: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        saved_path = output_path
        print(f"\n[Graph Saved] Dashboard saved to: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Overcooked-AI run performance and coordination metrics from a telemetry CSV."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the single-episode telemetry CSV file (e.g. results/random_baseline_cramped_room.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the output dashboard PNG image (default: results/<stem>_summary.png)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="Unknown Layout",
        help="Kitchen layout name for header (auto-detected if present in CSV)",
    )
    parser.add_argument(
        "--agent-0",
        type=str,
        default="Chef 0",
        help="Label for Agent 0 (default: Chef 0)",
    )
    parser.add_argument(
        "--agent-1",
        type=str,
        default="Chef 1",
        help="Label for Agent 1 (default: Chef 1)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the Matplotlib interactive window (useful for headless / batch runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_rows = parse_telemetry_csv(args.csv_path)

    default_output = args.output
    if default_output is None:
        default_output = args.csv_path.parent / f"{args.csv_path.stem}_summary.png"

    plot_run_dashboard(
        rows=parsed_rows,
        layout_name=args.layout,
        agent_0_label=args.agent_0,
        agent_1_label=args.agent_1,
        output_path=default_output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
