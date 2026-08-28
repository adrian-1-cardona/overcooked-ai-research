"""Validated schema for one timestep of an Overcooked-AI experiment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from typing import ClassVar


@dataclass(frozen=True)
class TelemetryRow:
    """Structured data recorded after one environment step."""

    ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {"north", "south", "east", "west", "stay", "interact"}
    )

    run_id: str
    episode_id: int
    episode_seed: int
    timestep: int
    layout_name: str
    agent_0_id: int
    agent_1_id: int
    agent_0_name: str
    agent_1_name: str
    agent_0_action: str
    agent_1_action: str
    reward: float
    agent_0_sparse_reward: float
    agent_1_sparse_reward: float
    agent_0_shaped_reward: float
    agent_1_shaped_reward: float
    done: bool
    agent_0_position: str
    agent_1_position: str
    agent_0_orientation: str
    agent_1_orientation: str
    agent_0_held_object: str
    agent_1_held_object: str
    agent_0_events: str
    agent_1_events: str

    def __post_init__(self) -> None:
        """Reject malformed values before they reach an experiment CSV."""
        required_text = {
            "run_id": self.run_id,
            "layout_name": self.layout_name,
            "agent_0_name": self.agent_0_name,
            "agent_1_name": self.agent_1_name,
            "agent_0_action": self.agent_0_action,
            "agent_1_action": self.agent_1_action,
            "agent_0_position": self.agent_0_position,
            "agent_1_position": self.agent_1_position,
            "agent_0_orientation": self.agent_0_orientation,
            "agent_1_orientation": self.agent_1_orientation,
            "agent_0_held_object": self.agent_0_held_object,
            "agent_1_held_object": self.agent_1_held_object,
        }
        invalid_text = [
            name
            for name, value in required_text.items()
            if not isinstance(value, str) or not value
        ]
        if invalid_text:
            raise ValueError(
                "Required telemetry text fields must be non-empty strings: "
                + ", ".join(invalid_text)
            )
        for name, value in {
            "agent_0_events": self.agent_0_events,
            "agent_1_events": self.agent_1_events,
        }.items():
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")

        identifiers = {
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "timestep": self.timestep,
            "agent_0_id": self.agent_0_id,
            "agent_1_id": self.agent_1_id,
        }
        invalid_identifiers = [
            name
            for name, value in identifiers.items()
            if not isinstance(value, Integral) or isinstance(value, bool)
        ]
        if invalid_identifiers:
            raise ValueError(
                "Telemetry identifiers must be integers: "
                + ", ".join(invalid_identifiers)
            )
        if (self.agent_0_id, self.agent_1_id) != (0, 1):
            raise ValueError("agent_0_id and agent_1_id must be 0 and 1")
        if self.episode_id < 1 or self.timestep < 1 or self.episode_seed < 0:
            raise ValueError(
                "episode_id and timestep must be positive; episode_seed cannot be negative"
            )
        numeric_values = {
            "reward": self.reward,
            "agent_0_sparse_reward": self.agent_0_sparse_reward,
            "agent_1_sparse_reward": self.agent_1_sparse_reward,
            "agent_0_shaped_reward": self.agent_0_shaped_reward,
            "agent_1_shaped_reward": self.agent_1_shaped_reward,
        }
        invalid_numbers = [
            name
            for name, value in numeric_values.items()
            if not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ]
        if invalid_numbers:
            raise ValueError(
                "Reward fields must be finite numbers: " + ", ".join(invalid_numbers)
            )
        if not isinstance(self.done, bool):
            raise ValueError("done must be a boolean")
        for name, value in {
            "agent_0_action": self.agent_0_action,
            "agent_1_action": self.agent_1_action,
            "agent_0_orientation": self.agent_0_orientation,
            "agent_1_orientation": self.agent_1_orientation,
        }.items():
            if value not in self.ACTIONS:
                raise ValueError(f"Invalid {name}: {value!r}")
        for name, value in {
            "agent_0_position": self.agent_0_position,
            "agent_1_position": self.agent_1_position,
        }.items():
            try:
                position = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"{name} must be a JSON [x, y] coordinate") from error
            if not (
                isinstance(position, list)
                and len(position) == 2
                and all(
                    isinstance(coordinate, int) and not isinstance(coordinate, bool)
                    for coordinate in position
                )
            ):
                raise ValueError(f"{name} must be a JSON [x, y] coordinate")

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-ready dictionary with stable field ordering."""
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> TelemetryRow:
        """Parse and validate a row read from CSV."""
        missing = [name for name in cls.fieldnames() if name not in row]
        extras = [name for name in row if name not in cls.fieldnames()]
        if missing or extras:
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if extras:
                details.append(f"unexpected columns: {', '.join(extras)}")
            raise ValueError("Invalid telemetry schema (" + "; ".join(details) + ")")
        try:
            return cls(
                **{
                    **row,
                    "episode_id": int(row["episode_id"]),
                    "episode_seed": int(row["episode_seed"]),
                    "timestep": int(row["timestep"]),
                    "agent_0_id": int(row["agent_0_id"]),
                    "agent_1_id": int(row["agent_1_id"]),
                    "reward": float(row["reward"]),
                    "agent_0_sparse_reward": float(row["agent_0_sparse_reward"]),
                    "agent_1_sparse_reward": float(row["agent_1_sparse_reward"]),
                    "agent_0_shaped_reward": float(row["agent_0_shaped_reward"]),
                    "agent_1_shaped_reward": float(row["agent_1_shaped_reward"]),
                    "done": {"True": True, "False": False}[row["done"]],
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed telemetry row: {error}") from error
