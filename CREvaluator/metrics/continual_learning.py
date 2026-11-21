"""Unified metrics interface for task-incremental learning experiments."""

from collections.abc import Iterable

from ._metric_utils import accuracy_matrix
from .area_under_curve import area_under_curve
from .average_accuracy import average_accuracy as mean_accuracy
from .average_forgetting import average_forgetting as calculate_average_forgetting
from .backward_transfer import bwt_metric
from .forward_transfer import forward_transfer as calculate_forward_transfer
from .forgetting_measure import forgetting_measure as calculate_forgetting
from .learning_accuracy import learning_accuracy as calculate_learning_accuracy


class ContinualLearningMetrics:
    """Calculate common continual-learning metrics from an accuracy matrix.

    Matrix rows are checkpoints after each task has been learned; columns are
    evaluation tasks. Cells for tasks not yet learned may be ``None``. Scores
    are fractions in [0, 1].
    """

    def __init__(
        self,
        accuracies: Iterable[Iterable[float | None]],
        baseline_accuracies: Iterable[float] | None = None,
    ) -> None:
        self.accuracies = tuple(tuple(row) for row in accuracy_matrix(accuracies))
        self.baseline_accuracies = (
            None if baseline_accuracies is None else tuple(float(value) for value in baseline_accuracies)
        )
        if self.baseline_accuracies is not None:
            if len(self.baseline_accuracies) != len(self.accuracies):
                raise ValueError("Provide one baseline accuracy per task")
            if any(not 0 <= value <= 1 for value in self.baseline_accuracies):
                raise ValueError("Baseline accuracies must be between 0 and 1")

    @property
    def number_of_tasks(self) -> int:
        """Number of tasks represented in the accuracy matrix."""
        return len(self.accuracies)

    def average_accuracy_curve(self) -> list[float]:
        """Return mean accuracy on seen tasks after every learning checkpoint."""
        curve = []
        for checkpoint, row in enumerate(self.accuracies):
            seen_scores = row[: checkpoint + 1]
            if any(score is None for score in seen_scores):
                raise ValueError("Every seen task needs a score at each checkpoint")
            curve.append(mean_accuracy(score for score in seen_scores if score is not None))
        return curve

    def average_accuracy(self) -> float:
        """Return final average accuracy over all tasks."""
        return self.average_accuracy_curve()[-1]

    def area_under_curve(self, normalize: bool = True) -> float:
        """Return area under the average-accuracy learning curve."""
        return area_under_curve(self.average_accuracy_curve(), normalize=normalize)

    def learning_accuracy(self) -> float:
        """Return mean accuracy immediately after each task is learned."""
        return calculate_learning_accuracy(self.accuracies)

    def backward_transfer(self) -> float:
        """Return mean final accuracy change on previously learned tasks."""
        return bwt_metric(self.accuracies)

    def forgetting_measure(self) -> list[float]:
        """Return forgetting per task, excluding the final task."""
        return calculate_forgetting(self.accuracies)

    def average_forgetting(self) -> float:
        """Return mean forgetting over tasks that had a later checkpoint."""
        return calculate_average_forgetting(self.accuracies)

    def forward_transfer(self) -> float:
        """Return forward transfer relative to supplied task baselines."""
        if self.baseline_accuracies is None:
            raise ValueError("baseline_accuracies are required to calculate forward transfer")
        return calculate_forward_transfer(self.accuracies, self.baseline_accuracies)

    def summary(self) -> dict[str, float | list[float] | None]:
        """Return all applicable aggregate metrics in a serializable mapping."""
        has_prior_tasks = self.number_of_tasks > 1
        return {
            "average_accuracy": self.average_accuracy(),
            "average_accuracy_curve": self.average_accuracy_curve(),
            "area_under_curve": self.area_under_curve() if has_prior_tasks else None,
            "learning_accuracy": self.learning_accuracy(),
            "backward_transfer": self.backward_transfer() if has_prior_tasks else None,
            "average_forgetting": self.average_forgetting() if has_prior_tasks else None,
            "forward_transfer": (
                self.forward_transfer()
                if has_prior_tasks and self.baseline_accuracies is not None
                else None
            ),
        }