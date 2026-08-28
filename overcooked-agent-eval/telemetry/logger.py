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
        if not isinstance(row, TelemetryRow):
            raise TypeError("TelemetryLogger only accepts TelemetryRow instances")
        self.rows.append(row)

    def save_csv(self, output_path: Path | str) -> Path:
        """Save all collected rows, creating the destination folder if needed."""
        if not self.rows:
            raise ValueError("Cannot save telemetry without any rows")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = TelemetryRow.fieldnames()
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.to_dict() for row in self.rows)
        return path

    @staticmethod
    def validate_csv(input_path: Path | str) -> int:
        """Validate a telemetry CSV and return its number of data rows."""
        path = Path(input_path)
        with path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames != TelemetryRow.fieldnames():
                missing = [
                    name
                    for name in TelemetryRow.fieldnames()
                    if name not in (reader.fieldnames or [])
                ]
                raise ValueError(
                    "Invalid telemetry CSV columns"
                    + (f"; missing: {', '.join(missing)}" if missing else "")
                )
            count = 0
            for line_number, row in enumerate(reader, start=2):
                try:
                    TelemetryRow.from_dict(row)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid telemetry at CSV line {line_number}: {error}"
                    ) from error
                count += 1
        if count == 0:
            raise ValueError("Telemetry CSV contains no data rows")
        return count
