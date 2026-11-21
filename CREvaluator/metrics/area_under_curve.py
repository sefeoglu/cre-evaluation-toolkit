from collections.abc import Iterable
from math import isfinite


def area_under_curve(scores: Iterable[float], normalize: bool = False) -> float:
	"""Return the trapezoidal area over equally spaced evaluation steps.

	Set ``normalize=True`` to divide by the span of the evaluation steps,
	yielding the mean score over that span.
	"""
	values = [float(score) for score in scores]
	if len(values) < 2:
		raise ValueError("At least two scores are required to calculate area")
	if not all(isfinite(score) for score in values):
		raise ValueError("Scores must all be finite numbers")

	area = sum((left + right) / 2 for left, right in zip(values, values[1:]))
	return area / (len(values) - 1) if normalize else area
