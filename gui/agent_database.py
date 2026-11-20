"""Agent database manager for tracking, searching, and deleting trained agents.

Maintains model registry and metadata for agent checkpoints trained with
Ray RLlib and the Overcooked-AI pipeline.

References:
- models/registry.json: Persistent JSON catalog of registered agents.
- HumanCompatibleAI/human_aware_rl: Checkpoint directory structure and metadata conventions.
- Project Guidance: Prof. Rodrigo Canaan (Cal Poly) - Overcooked-AI Research
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"


BENCHMARK_LAYOUT_NAMES = (
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
)


def _is_agent_checkpoint_dir(p: Path) -> bool:
    """Check if directory represents a concrete trained agent checkpoint collection."""
    if not p.is_dir() or p.name.startswith("."):
        return False
    has_meta = (p / "metadata.json").is_file()
    has_config = (p / "config.pkl").is_file()
    has_direct_ckpt = any(d.is_dir() for d in p.glob("checkpoint_*"))
    try:
        has_layout_subdirs = any(
            sub.is_dir() and sub.name in BENCHMARK_LAYOUT_NAMES and any(d.is_dir() for d in sub.glob("checkpoint_*"))
            for sub in p.iterdir()
        )
    except Exception:
        has_layout_subdirs = False
    return has_meta or has_config or has_direct_ckpt or has_layout_subdirs


class AgentDatabase:
    """Manages persistence, search, and deletion of custom trained Overcooked agents."""

    @classmethod
    def load(cls) -> list[dict[str, Any]]:
        """Load all registered agents, syncing with existing directories in models/."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []

        if REGISTRY_FILE.is_file():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records = data
            except Exception:
                records = []

        # Clean up records whose paths no longer exist on disk or are not valid agent dirs
        valid_records: list[dict[str, Any]] = []
        changed = False
        for r in records:
            p_str = r.get("checkpoint_path")
            if p_str and Path(p_str).exists() and _is_agent_checkpoint_dir(Path(p_str)):
                valid_records.append(r)
            else:
                changed = True
        records = valid_records

        # Sync disk directories: if any model directory exists in models/ or results/ without a record, auto-register it
        existing_ids = {r.get("id") for r in records}
        existing_paths = {str(Path(r.get("checkpoint_path", "")).resolve()) for r in records if r.get("checkpoint_path")}

        candidate_dirs: list[Path] = []
        if MODELS_DIR.is_dir():
            for p in sorted(MODELS_DIR.iterdir()):
                if _is_agent_checkpoint_dir(p):
                    candidate_dirs.append(p)

        results_dir = REPO_ROOT / "results"
        if results_dir.is_dir():
            for p in sorted(results_dir.iterdir()):
                if not p.is_dir() or p.name.startswith("."):
                    continue
                if _is_agent_checkpoint_dir(p):
                    candidate_dirs.append(p)
                # Check 1 level deeper (e.g. results/frame_stacking/ppo_greedy)
                try:
                    for sub in sorted(p.iterdir()):
                        if _is_agent_checkpoint_dir(sub):
                            candidate_dirs.append(sub)
                except Exception:
                    pass

        for p in candidate_dirs:
            resolved = str(p.resolve())
            if resolved not in existing_paths and p.name not in existing_ids:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                layout_val = "cramped_room"
                recipe_val = "ppo_custom"
                reward_mean_val = 0.0
                meta_file = p / "metadata.json"
                if meta_file.is_file():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            m_data = json.load(mf)
                            layout_val = m_data.get("layout_name", "cramped_room")
                            recipe_val = m_data.get("recipe", "ppo_custom")
                            reward_mean_val = float(m_data.get("reward_mean", 0.0))
                    except Exception:
                        pass
                records.append({
                    "id": p.name,
                    "name": p.name,
                    "date_trained": mtime.strftime("%m/%d/%Y"),
                    "time_trained": mtime.strftime("%H:%M"),
                    "checkpoint_path": resolved,
                    "layout": layout_val,
                    "recipe": recipe_val,
                    "reward_mean": reward_mean_val,
                })
                existing_paths.add(resolved)
                existing_ids.add(p.name)
                changed = True

        if changed:
            cls.save(records)

        records.sort(key=lambda r: (r.get("date_trained", ""), r.get("time_trained", "")), reverse=True)
        return records

    @classmethod
    def save(cls, records: list[dict[str, Any]]) -> None:
        """Write records to models/registry.json."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        tmp.replace(REGISTRY_FILE)

    @classmethod
    def add_agent(
        cls,
        name: str,
        layout: str,
        checkpoint_path: str | Path,
        agent_1: str = "ppo",
        agent_1_pct: int = 50,
        agent_2: str = "greedy",
        agent_2_pct: int = 50,
        reward_mean: float = 0.0,
    ) -> dict[str, Any]:
        """Add a newly trained agent record."""
        records = cls.load()
        now = datetime.now()
        safe_name = name.strip() or f"Agent_{now.strftime('%H%M')}"
        resolved_path = str(Path(checkpoint_path).resolve())

        existing = next((r for r in records if r.get("checkpoint_path") == resolved_path), None)
        if existing:
            existing["name"] = safe_name
            existing["layout"] = layout
            existing["date_trained"] = now.strftime("%m/%d/%Y")
            existing["time_trained"] = now.strftime("%H:%M")
            existing["agent_1"] = agent_1
            existing["agent_1_pct"] = agent_1_pct
            existing["agent_2"] = agent_2
            existing["agent_2_pct"] = agent_2_pct
            existing["reward_mean"] = reward_mean
            cls.save(records)
            return existing

        record = {
            "id": f"{safe_name.lower().replace(' ', '_')}_{now.strftime('%Y%m%d_%H%M%S')}",
            "name": safe_name,
            "layout": layout,
            "date_trained": now.strftime("%m/%d/%Y"),
            "time_trained": now.strftime("%H:%M"),
            "agent_1": agent_1,
            "agent_1_pct": agent_1_pct,
            "agent_2": agent_2,
            "agent_2_pct": agent_2_pct,
            "checkpoint_path": resolved_path,
            "reward_mean": reward_mean,
        }
        records.insert(0, record)
        cls.save(records)
        return record

    @classmethod
    def delete_agent(cls, agent_id: str) -> bool:
        """Delete an agent record and remove its files from disk."""
        records = cls.load()
        target = next((r for r in records if r.get("id") == agent_id), None)
        if not target:
            return False

        # Remove from disk
        path_str = target.get("checkpoint_path")
        if path_str:
            p = Path(path_str)
            if p.exists() and p.resolve() != MODELS_DIR.resolve():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except Exception as e:
                    print(f"[AgentDB] Error deleting path {p}: {e}")

        # Also remove directory by agent name if present in models/
        target_name = target.get("name")
        if target_name:
            named_p = MODELS_DIR / target_name
            if named_p.exists() and named_p.resolve() != MODELS_DIR.resolve():
                try:
                    if named_p.is_dir():
                        shutil.rmtree(named_p)
                    else:
                        named_p.unlink()
                except Exception as e:
                    print(f"[AgentDB] Error deleting named directory {named_p}: {e}")

        # Remove from registry
        new_records = [r for r in records if r.get("id") != agent_id]
        cls.save(new_records)
        return True

    @classmethod
    def search(cls, query: str = "") -> list[dict[str, Any]]:
        """Filter records by search query matching name or date."""
        records = cls.load()
        if not query.strip():
            return records
        q = query.strip().lower()
        return [
            r for r in records
            if q in r.get("name", "").lower()
            or q in r.get("date_trained", "").lower()
            or q in r.get("layout", "").lower()
        ]
