"""Agent database manager for tracking, searching, and deleting trained agents."""

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

        # Sync disk directories: if any model directory exists without a record, auto-register it
        existing_ids = {r.get("id") for r in records}
        existing_paths = {str(Path(r.get("checkpoint_path", "")).resolve()) for r in records if r.get("checkpoint_path")}

        changed = False
        for p in sorted(MODELS_DIR.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            has_checkpoint = any(p.glob("checkpoint_*")) or (p / "config.pkl").is_file()
            resolved = str(p.resolve())
            if has_checkpoint and resolved not in existing_paths and p.name not in existing_ids:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                records.append({
                    "id": p.name,
                    "name": p.name,
                    "date_trained": mtime.strftime("%m/%d/%Y"),
                    "time_trained": mtime.strftime("%H:%M"),
                    "checkpoint_path": resolved,
                    "layout": "cramped_room",
                    "recipe": "ppo_custom",
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
