from collections.abc import Iterable

from ._metric_utils import paired_labels


def unseen_class_accuracy(
	y_true: Iterable[object],
	y_pred: Iterable[object],
	seen_classes: Iterable[object],
) -> float:
	"""Return accuracy on examples whose true class is not in ``seen_classes``."""
	actual, predicted = paired_labels(y_true, y_pred)
	seen = set(seen_classes)
	selected = [(truth, prediction) for truth, prediction in zip(actual, predicted) if truth not in seen]
	if not selected:
		raise ValueError("No examples belong to unseen classes")
	return sum(truth == prediction for truth, prediction in selected) / len(selected)