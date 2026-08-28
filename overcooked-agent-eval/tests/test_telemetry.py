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
        episode_seed=42,
        timestep=1,
        layout_name="cramped_room",
        agent_0_id=0,
        agent_1_id=1,
        agent_0_name="RandomAgent",
        agent_1_name="RandomAgent",
        agent_0_action="north",
        agent_1_action="stay",
        reward=0,
        agent_0_sparse_reward=0,
        agent_1_sparse_reward=0,
        agent_0_shaped_reward=0,
        agent_1_shaped_reward=0,
        done=False,
        agent_0_previous_position="[1, 3]",
        agent_1_previous_position="[3, 3]",
        agent_0_position="[1, 2]",
        agent_1_position="[3, 2]",
        agent_0_orientation="north",
        agent_1_orientation="south",
        agent_0_held_object="none",
        agent_1_held_object="onion",
        agent_0_events="",
        agent_1_events="onion_pickup",
    )


class TelemetryTests(unittest.TestCase):
    def test_row_converts_to_expected_dictionary(self) -> None:
        row = sample_row().to_dict()
        self.assertEqual(row["layout_name"], "cramped_room")
        self.assertEqual(row["agent_1_held_object"], "onion")
        self.assertEqual(len(row), 27)

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
            self.assertEqual(TelemetryLogger.validate_csv(saved_path), 1)

    def test_row_rejects_malformed_position(self) -> None:
        values = sample_row().to_dict()
        values["agent_0_position"] = "not-a-coordinate"
        with self.assertRaisesRegex(ValueError, "JSON"):
            TelemetryRow(**values)

    def test_row_rejects_non_finite_reward(self) -> None:
        values = sample_row().to_dict()
        values["reward"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            TelemetryRow(**values)

    def test_csv_validation_reports_missing_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "missing-column.csv"
            values = sample_row().to_dict()
            values.pop("episode_seed")
            with output.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=values)
                writer.writeheader()
                writer.writerow(values)

            with self.assertRaisesRegex(ValueError, "episode_seed"):
                TelemetryLogger.validate_csv(output)

    def test_csv_validation_reports_malformed_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "malformed.csv"
            values = sample_row().to_dict()
            values["done"] = "sometimes"
            with output.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=TelemetryRow.fieldnames()
                )
                writer.writeheader()
                writer.writerow(values)

            with self.assertRaisesRegex(ValueError, "CSV line 2"):
                TelemetryLogger.validate_csv(output)


if __name__ == "__main__":
    unittest.main()
