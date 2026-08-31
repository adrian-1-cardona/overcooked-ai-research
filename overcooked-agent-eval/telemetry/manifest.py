"""Reproducibility manifest recording experiment configuration and environment metadata."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import overcooked_ai_py


def get_git_commit(cwd: Path | str | None = None) -> str:
    """Return the git commit SHA for the given directory or 'unknown'."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cwd) if cwd else None,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        if commit:
            return commit
    except Exception:
        pass
    return "unknown"


def get_overcooked_ai_version(project_root: Path) -> str:
    """Return the overcooked-ai package version or submodule git commit."""
    version = getattr(overcooked_ai_py, "__version__", None)
    if version and version != "unknown":
        return str(version)
    submodule_dir = project_root.parent / "external" / "overcooked_ai"
    if submodule_dir.exists():
        commit = get_git_commit(submodule_dir)
        if commit != "unknown":
            return f"submodule-{commit}"
    return "installed-local"


@dataclass(frozen=True)
class BatchManifest:
    """Complete metadata required to reproduce an experiment batch."""

    schema_version: int
    run_id: str
    layout: str
    agent_0_name: str
    agent_1_name: str
    episodes: int
    horizon: int
    base_seed: int
    episode_seeds: list[int]
    output_telemetry_file: str
    python_version: str
    overcooked_ai_version: str
    evaluation_commit: str
    created_at: str

    def __post_init__(self) -> None:
        """Validate that all required manifest fields are populated."""
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("Manifest run_id must be a non-empty string")
        if not self.layout or not isinstance(self.layout, str):
            raise ValueError("Manifest layout must be a non-empty string")
        if not self.agent_0_name or not self.agent_1_name:
            raise ValueError("Manifest agent names must be non-empty strings")
        if self.episodes < 1 or self.horizon < 1 or self.base_seed < 0:
            raise ValueError("Manifest numbers must be valid positive values")
        if len(self.episode_seeds) != self.episodes:
            raise ValueError("Manifest episode_seeds length must match episodes count")
        if not self.output_telemetry_file:
            raise ValueError("Manifest output_telemetry_file must be specified")
        if not self.python_version or not self.overcooked_ai_version:
            raise ValueError("Manifest version metadata cannot be empty")
        if not self.evaluation_commit:
            raise ValueError("Manifest evaluation_commit cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BatchManifest:
        return cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            layout=str(data["layout"]),
            agent_0_name=str(data["agent_0_name"]),
            agent_1_name=str(data["agent_1_name"]),
            episodes=int(data["episodes"]),
            horizon=int(data["horizon"]),
            base_seed=int(data["base_seed"]),
            episode_seeds=[int(seed) for seed in data["episode_seeds"]],
            output_telemetry_file=str(data["output_telemetry_file"]),
            python_version=str(data["python_version"]),
            overcooked_ai_version=str(data["overcooked_ai_version"]),
            evaluation_commit=str(data["evaluation_commit"]),
            created_at=str(data["created_at"]),
        )


def create_batch_manifest(
    run_id: str,
    layout: str,
    agent_0_name: str,
    agent_1_name: str,
    episodes: int,
    horizon: int,
    base_seed: int,
    output_telemetry_path: Path | str,
    project_root: Path | None = None,
    created_at: str | None = None,
) -> BatchManifest:
    """Create a validated BatchManifest for an experiment run."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    python_version = sys.version.split()[0]
    overcooked_version = get_overcooked_ai_version(project_root)
    eval_commit = get_git_commit(project_root)
    timestamp = created_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

    return BatchManifest(
        schema_version=2,
        run_id=run_id,
        layout=layout,
        agent_0_name=agent_0_name,
        agent_1_name=agent_1_name,
        episodes=episodes,
        horizon=horizon,
        base_seed=base_seed,
        episode_seeds=[base_seed + i for i in range(episodes)],
        output_telemetry_file=str(output_telemetry_path),
        python_version=python_version,
        overcooked_ai_version=overcooked_version,
        evaluation_commit=eval_commit,
        created_at=timestamp,
    )


def save_batch_manifest(manifest: BatchManifest, manifest_path: Path | str) -> Path:
    """Write a batch manifest to JSON."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_batch_manifest(manifest_path: Path | str) -> BatchManifest:
    """Load and validate a batch manifest from JSON."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BatchManifest.from_dict(data)
