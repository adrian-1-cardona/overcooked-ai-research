"""
Run a k × k coordination matrix for Overcooked agents.
Agents: random, ppo_sp, greedy (simple rule‑based).
Partners: human_proxy (placeholder), random, greedy.
Levels: cramped_room, asymmetric, forced_coordination.
The script runs a short episode for each (agent, partner, level) tuple,
collects the final return, and writes a CSV at `results/matrix.csv`.
"""
import csv
import itertools
import os
import subprocess
from pathlib import Path

AGENTS   = ["random", "ppo_sp", "greedy"]
PARTNERS = ["human_proxy", "random", "greedy"]
LEVELS   = ["cramped_room", "asymmetric", "forced_coordination"]

def run_one_episode(agent: str, partner: str, level: str) -> float:
    """Run a single episode with the given agents on the given level.
    Returns the final episode reward (float).  This uses the renderer in
    headless video mode and discards the output video – we only need the
    printed score, which the renderer emits as the last line of stdout.
    """
    cmd = [
        "python", "render.py",
        "--agent-0", agent,
        "--agent-1", partner,
        "--level", level,
        "--mode", "video",
        "--horizon", "200",
        "--output", os.devnull,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # The renderer prints the final return as a number on the last line.
    for line in reversed(result.stdout.splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return 0.0

def main():
    rows = []
    for agent, partner, level in itertools.product(AGENTS, PARTNERS, LEVELS):
        score = run_one_episode(agent, partner, level)
        rows.append([agent, partner, level, score])
        print(f"{agent} vs {partner} on {level}: {score:.1f}")

    out_path = Path("results/matrix.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "partner", "level", "score"])
        writer.writerows(rows)
    print(f"Saved matrix → {out_path}")

if __name__ == "__main__":
    main()
