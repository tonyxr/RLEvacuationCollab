import csv
import os
import tempfile
import unittest

from TrainingLogger import CSV_COLUMNS, CSV_SCHEMA_VERSION, trainingLog


class _ScalarRecorder:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, global_step):
        self.values[(name, int(global_step))] = float(value)

    def flush(self):
        pass

    def close(self):
        pass


class TrainingLoggerShelterLifecycleTests(unittest.TestCase):
    def test_progress_csv_persists_shelter_failure_and_reevacuation_metrics(self):
        metrics = {
            "shelter_closures": 3,
            "closed_this_step": 1,
            "current_sheltered_population": 47,
            "cumulative_shelter_admissions": 123,
            "forced_reevacuated": 19,
            "reactivated_this_step": 8,
            "total_shelter_capacity": 200.0,
            "operational_shelter_capacity": 125.0,
            "shelter_utilization": 0.376,
        }
        with tempfile.TemporaryDirectory() as directory:
            logger = trainingLog(directory)
            logger.log_step(17, 0.25, metrics)
            logger.close()

            with open(os.path.join(directory, "progress.csv"), newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader)

        self.assertEqual(CSV_SCHEMA_VERSION, 7)
        self.assertEqual(reader.fieldnames, CSV_COLUMNS)
        self.assertIsNone(row.get(None), "CSV row must align exactly with its header")
        self.assertEqual(row["shelter_closures"], "3")
        self.assertEqual(row["closed_this_step"], "1")
        self.assertEqual(row["current_sheltered_population"], "47")
        self.assertEqual(row["cumulative_shelter_admissions"], "123")
        self.assertEqual(row["forced_reevacuated"], "19")
        self.assertEqual(row["reactivated_this_step"], "8")
        self.assertEqual(row["total_shelter_capacity"], "200.0")
        self.assertEqual(row["operational_shelter_capacity"], "125.0")
        self.assertEqual(row["shelter_utilization"], "0.376")

    def test_tensorboard_receives_shelter_lifecycle_diagnostics(self):
        recorder = _ScalarRecorder()
        with tempfile.TemporaryDirectory() as directory:
            logger = trainingLog(directory)
            logger.tb = recorder
            logger.log_step(
                9,
                -0.5,
                {
                    "shelter_closures": 2,
                    "closed_this_step": 1,
                    "current_sheltered_population": 31,
                    "cumulative_shelter_admissions": 70,
                    "forced_reevacuated": 11,
                    "reactivated_this_step": 6,
                    "operational_shelter_capacity": 90.0,
                },
            )
            logger.close()

        expected = {
            "shelter/closures_cumulative": 2.0,
            "shelter/closed_this_step": 1.0,
            "shelter/current_sheltered_population": 31.0,
            "shelter/admissions_cumulative": 70.0,
            "shelter/forced_reevacuated_cumulative": 11.0,
            "shelter/reactivated_this_step": 6.0,
            "shelter/operational_capacity": 90.0,
        }
        for name, value in expected.items():
            self.assertEqual(recorder.values[(name, 9)], value)


if __name__ == "__main__":
    unittest.main()
