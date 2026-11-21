from collections.abc import Iterable

from ._metric_utils import accuracy_matrix, mean
from .forgetting_measure import forgetting_measure


def average_forgetting(accuracies: Iterable[Iterable[float | None]]) -> float:
	"""Return mean forgetting over all tasks except the final task."""
	matrix = accuracy_matrix(accuracies)
	return mean(forgetting_measure(matrix), "forgetting scores")
