import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from CREvaluator.metrics import ContinualLearningMetrics
from CREvaluator.metrics.backward_transfer import bwt, bwt_metric


class ContinualLearningMetricsTests(unittest.TestCase):
    def setUp(self):
        self.accuracies = [
            [0.8, 0.4, 0.5],
            [0.6, 0.7, 0.6],
            [0.5, 0.8, 0.9],
        ]
        self.metrics = ContinualLearningMetrics(
            self.accuracies,
            baseline_accuracies=[0.1, 0.2, 0.3],
        )

    def test_accuracy_trajectory_and_summary(self):
        curve = self.metrics.average_accuracy_curve()
        self.assertAlmostEqual(curve[0], 0.8)
        self.assertAlmostEqual(curve[1], 0.65)
        self.assertAlmostEqual(curve[2], 2.2 / 3)
        summary = self.metrics.summary()
        self.assertAlmostEqual(summary["average_accuracy"], 2.2 / 3)
        self.assertAlmostEqual(summary["area_under_curve"], 0.7083333333333333)

    def test_transfer_forgetting_and_learning_metrics(self):
        self.assertAlmostEqual(self.metrics.backward_transfer(), -0.1)
        self.assertAlmostEqual(self.metrics.forward_transfer(), 0.25)
        forgetting = self.metrics.forgetting_measure()
        self.assertAlmostEqual(forgetting[0], 0.3)
        self.assertAlmostEqual(forgetting[1], -0.1)
        self.assertAlmostEqual(self.metrics.average_forgetting(), 0.1)
        self.assertAlmostEqual(self.metrics.learning_accuracy(), 0.8)
        self.assertAlmostEqual(bwt_metric(self.accuracies), -0.1)

    def test_forward_transfer_requires_baselines(self):
        metrics = ContinualLearningMetrics(self.accuracies)
        with self.assertRaisesRegex(ValueError, "baseline_accuracies"):
            metrics.forward_transfer()

    def test_one_task_summary_marks_undefined_metrics(self):
        metrics = ContinualLearningMetrics([[0.75]])
        summary = metrics.summary()
        self.assertEqual(summary["average_accuracy"], 0.75)
        self.assertIsNone(summary["area_under_curve"])
        self.assertIsNone(summary["backward_transfer"])
        self.assertIsNone(summary["average_forgetting"])
        self.assertIsNone(summary["forward_transfer"])

    def test_invalid_accuracy_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "square"):
            ContinualLearningMetrics([[0.8, 0.7]])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ContinualLearningMetrics([[1.2]])

    def test_legacy_json_backward_transfer_workflow(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = root / "results"
            ground_truth = root / "ground_truth"
            task_labels = [[0, 1], [1, 0]]
            task_predictions = [[0, 1], [0, 0, 1, 0]]

            for task_index, labels in enumerate(task_labels, start=1):
                truth_directory = ground_truth / f"task_{task_index}"
                result_directory = results / f"task_{task_index}"
                truth_directory.mkdir(parents=True)
                result_directory.mkdir(parents=True)
                (truth_directory / "test.json").write_text(json.dumps(labels))
                (result_directory / "seen_task_predict.json").write_text(
                    json.dumps(task_predictions[task_index - 1])
                )

            self.assertAlmostEqual(bwt(str(results), str(ground_truth), 2), -0.5)


if __name__ == "__main__":
    unittest.main()
