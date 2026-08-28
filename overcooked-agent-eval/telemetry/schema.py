"""Schema for one timestep of an Overcooked-AI experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TelemetryRow:
    """Structured data recorded after one environment step."""

    run_id: str
    episode_id: int
    timestep: int
    layout_name: str
    agent_0_name: str
    agent_1_name: str
    agent_0_action: str
    agent_1_action: str
    reward: float
    done: bool
    agent_0_position: str
    agent_1_position: str
    agent_0_orientation: str
    agent_1_orientation: str
    agent_0_held_object: str
    agent_1_held_object: str

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-ready dictionary with stable field ordering."""
        return asdict(self)
