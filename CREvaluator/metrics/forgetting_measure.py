from collections.abc import Iterable

from ._metric_utils import accuracy_matrix


def forgetting_measure(accuracies: Iterable[Iterable[float | None]]) -> list[float]:
	"""Return per-task forgetting: best prior accuracy minus final accuracy.

	Rows are evaluation checkpoints and columns are tasks. The newest task is
	omitted because it has no later checkpoint at which it could be forgotten.
	"""
	matrix = accuracy_matrix(accuracies)
	if len(matrix) < 2:
		raise ValueError("At least two tasks are required to measure forgetting")

	forgetting = []
	final_row = matrix[-1]
	for task_index in range(len(matrix) - 1):
		prior_scores = [
			score
			for row in matrix[task_index:-1]
			if (score := row[task_index]) is not None
		]
		final_score = final_row[task_index]
		if not prior_scores or final_score is None:
			raise ValueError("Each previously learned task needs prior and final scores")
		forgetting.append(max(prior_scores) - final_score)
	return forgetting
