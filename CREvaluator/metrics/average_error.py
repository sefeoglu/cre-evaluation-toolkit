from collections.abc import Iterable

from ._metric_utils import numeric_values


def average_error(accuracies: Iterable[float]) -> float:
	"""Return classification error as one minus mean accuracy.

	Accuracy values must be fractions in the range [0, 1].
	"""
	values = numeric_values(accuracies, "accuracies")
	if any(not 0 <= accuracy <= 1 for accuracy in values):
		raise ValueError("Accuracy values must be between 0 and 1")
	return 1 - sum(values) / len(values)
