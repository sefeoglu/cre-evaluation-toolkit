from collections.abc import Iterable
from math import sqrt

from ._metric_utils import numeric_values, paired_labels


def root_mean_square_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
	"""Return the root mean squared difference between paired numeric values."""
	actual, predicted = paired_labels(y_true, y_pred)
	actual_values = numeric_values(actual, "y_true")
	predicted_values = numeric_values(predicted, "y_pred")
	return sqrt(sum((truth - prediction) ** 2 for truth, prediction in zip(actual_values, predicted_values)) / len(actual_values))