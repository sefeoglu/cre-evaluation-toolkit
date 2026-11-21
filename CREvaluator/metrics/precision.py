from collections.abc import Iterable

from sklearn.metrics import precision_score

from ._metric_utils import paired_labels


def precision(
	y_true: Iterable[object],
	y_pred: Iterable[object],
	average: str | None = "macro",
) -> float | dict[object, float]:
	"""Compute precision; use ``average=None`` for per-class scores."""
	actual, predicted = paired_labels(y_true, y_pred)
	if average is None:
		labels = sorted(set(actual) | set(predicted))
		scores = precision_score(actual, predicted, labels=labels, average=None, zero_division=0)
		return dict(zip(labels, scores.tolist()))
	scores = precision_score(actual, predicted, average=average, zero_division=0)
	return float(scores)