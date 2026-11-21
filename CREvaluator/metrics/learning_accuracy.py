from collections.abc import Iterable

from ._metric_utils import accuracy_matrix, mean


def learning_accuracy(accuracies: Iterable[Iterable[float | None]]) -> float:
	"""Return mean accuracy on each task immediately after it is learned."""
	matrix = accuracy_matrix(accuracies)
	diagonal = [matrix[index][index] for index in range(len(matrix))]
	if any(score is None for score in diagonal):
		raise ValueError("The accuracy matrix must contain all diagonal scores")
	return mean((score for score in diagonal if score is not None), "learning accuracies")