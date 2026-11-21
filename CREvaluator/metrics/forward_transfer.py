from collections.abc import Iterable

from ._metric_utils import accuracy_matrix, mean, numeric_values


def forward_transfer(
	accuracies: Iterable[Iterable[float | None]],
	baseline_accuracies: Iterable[float],
) -> float:
	"""Average pre-learning task accuracy minus each task's baseline.

	``accuracies[i][j]`` is the score on task ``j`` after checkpoint ``i``;
	baseline values are indexed by task.
	"""
	matrix = accuracy_matrix(accuracies)
	baselines = numeric_values(baseline_accuracies, "baseline_accuracies")
	if len(matrix) < 2 or len(baselines) != len(matrix):
		raise ValueError("A square matrix and one baseline per task are required")
	if any(not 0 <= baseline <= 1 for baseline in baselines):
		raise ValueError("Baseline accuracies must be between 0 and 1")

	transfers = []
	for task_index in range(1, len(matrix)):
		prelearning_score = matrix[task_index - 1][task_index]
		if prelearning_score is None:
			raise ValueError("Pre-learning scores must be provided for every task")
		transfers.append(prelearning_score - baselines[task_index])
	return mean(transfers, "transfer scores")