"""Focused tests for the reusable telemetry components."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from telemetry import TelemetryLogger, TelemetryRow


def sample_row() -> TelemetryRow:
    return TelemetryRow(
        run_id="test-run",
        episode_id=1,
        timestep=1,
        layout_name="cramped_room",
        agent_0_name="RandomAgent",
        agent_1_name="RandomAgent",
        agent_0_action="north",
        agent_1_action="stay",
        reward=0,
        done=False,
        agent_0_position="(1, 2)",
        agent_1_position="(3, 2)",
        agent_0_orientation="north",
        agent_1_orientation="south",
        agent_0_held_object="none",
        agent_1_held_object="onion",
    )


class TelemetryTests(unittest.TestCase):
    def test_row_converts_to_expected_dictionary(self) -> None:
        row = sample_row().to_dict()
        self.assertEqual(row["layout_name"], "cramped_room")
        self.assertEqual(row["agent_1_held_object"], "onion")
        self.assertEqual(len(row), 16)

    def test_logger_creates_folder_and_writes_csv(self) -> None:
        logger = TelemetryLogger()
        logger.log(sample_row())

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "telemetry.csv"
            saved_path = logger.save_csv(output)
            with saved_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertTrue(saved_path.exists())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "test-run")


if __name__ == "__main__":
    unittest.main()
