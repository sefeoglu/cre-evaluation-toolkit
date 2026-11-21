from collections.abc import Iterable

from ._metric_utils import mean


def average_accuracy(accuracies: Iterable[float]) -> float:
	"""Return the mean of accuracy scores, expressed as fractions or percentages."""
	return mean(accuracies, "accuracies")
