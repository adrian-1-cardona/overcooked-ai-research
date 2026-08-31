"""Reproducibility manifest recording experiment configuration and environment metadata.

What this file does (in simple terms):
Think of this like a recipe card or a birth certificate for each game experiment!
Whenever we run a batch of Overcooked games, this file creates a JSON summary
that records:
  - Which kitchen map (layout) was played
  - Which robot players (agents) were in the kitchen
  - How many games (episodes) were played
  - The exact random number seeds used
  - The Python version and git commit codes
This way, if anyone wants to run the exact same experiment years later,
they have all the exact ingredients and settings to get the exact same results.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import overcooked_ai_py


def get_git_commit(cwd: Path | str | None = None) -> str:
    """Find the current git commit code so we know the exact code version used.
    
    If git is not available or fails, it returns 'unknown'.
    """
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
    """Find the version or git commit code for the Overcooked-AI game package.
    
    This ensures we know exactly which version of the game simulator ran.
    """
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
    """The complete recipe card (manifest) for a batch of experiments."""

    # Schema version so future code knows what format this is
    schema_version: int
    
    # Unique ID for this exact configuration
    run_id: str
    
    # Kitchen layout name (like 'cramped_room')
    layout: str
    
    # Names of player 1 (chef 0) and player 2 (chef 1)
    agent_0_name: str
    agent_1_name: str
    
    # How many games were played and how many timesteps each game lasted
    episodes: int
    horizon: int
    
    # Starting seed and the full list of seeds (one for each game)
    base_seed: int
    episode_seeds: list[int]
    
    # Path to the CSV file where the game-by-game data was saved
    output_telemetry_file: str
    
    # Computer environment details: Python version, game version, git commit
    python_version: str
    overcooked_ai_version: str
    evaluation_commit: str
    
    # The date and time when this experiment was created
    created_at: str

    def __post_init__(self) -> None:
        """Check that no important information is missing or blank."""
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
        """Convert manifest to a standard Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BatchManifest:
        """Load manifest from a Python dictionary."""
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
    """Create a validated BatchManifest containing all environment details."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    # Grab python version (e.g. 3.10.21)
    python_version = sys.version.split()[0]
    
    # Grab simulator version and code commit
    overcooked_version = get_overcooked_ai_version(project_root)
    eval_commit = get_git_commit(project_root)
    timestamp = created_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Build the manifest
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
    """Save the manifest to a nice, readable JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_batch_manifest(manifest_path: Path | str) -> BatchManifest:
    """Read a saved manifest JSON file back into Python."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BatchManifest.from_dict(data)
