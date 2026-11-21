from collections.abc import Iterable

from ._metric_utils import paired_labels


def whole_accuracy(y_true: Iterable[object], y_pred: Iterable[object]) -> float:
	"""Return the fraction of all predictions that exactly match their labels."""
	actual, predicted = paired_labels(y_true, y_pred)
	return sum(truth == prediction for truth, prediction in zip(actual, predicted)) / len(actual)