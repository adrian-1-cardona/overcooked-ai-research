"""In-memory collection and CSV output for experiment telemetry."""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import TelemetryRow


class TelemetryLogger:
    """Collect telemetry rows and write them to a reusable CSV format."""

    def __init__(self) -> None:
        self.rows: list[TelemetryRow] = []

    def log(self, row: TelemetryRow) -> None:
        self.rows.append(row)

    def save_csv(self, output_path: Path | str) -> Path:
        """Save all collected rows, creating the destination folder if needed."""
        if not self.rows:
            raise ValueError("Cannot save telemetry without any rows")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.rows[0].to_dict())
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.to_dict() for row in self.rows)
        return path
