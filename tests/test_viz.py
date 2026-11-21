import unittest


class VisualizationTests(unittest.TestCase):
    def test_visualization_exports_and_plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from CREvaluator.metrics import ContinualLearningMetrics
        from CREvaluator.viz import (
            plot_accuracy_matrix,
            plot_average_accuracy_curve,
            plot_summary,
        )

        accuracies = [[0.8, 0.4], [0.6, 0.7]]
        metrics = ContinualLearningMetrics(accuracies)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        self.assertIs(plot_accuracy_matrix(accuracies, ax=axes[0]), axes[0])
        self.assertIs(plot_average_accuracy_curve(metrics, ax=axes[1]), axes[1])
        self.assertIs(plot_summary(metrics.summary(), ax=axes[2]), axes[2])
        plt.close(fig)

    def test_accuracy_matrix_rejects_non_square_rows(self):
        import matplotlib
        matplotlib.use("Agg")

        from CREvaluator.viz import plot_accuracy_matrix

        with self.assertRaisesRegex(ValueError, "square"):
            plot_accuracy_matrix([[0.8, 0.7], [0.6]])


if __name__ == "__main__":
    unittest.main()
