# CREvaluator

CREvaluator is a Python metric library for evaluating continual-learning experiments. It includes task-sequence metrics, label-based classification metrics, and the original JSON-backed BWT workflow.

## Installation

```bash
pip install .
```

## Continual learning metrics

Use a square accuracy matrix. Row `i` contains evaluation results after learning task `i`; column `j` contains results on task `j`. Use `None` when a task has not been evaluated yet. Accuracies and baselines are fractions in `[0, 1]`.

```python
from CREvaluator.metrics import ContinualLearningMetrics

accuracies = [
    [0.80, 0.35, 0.25],  # after task 1; future-task scores are the FWT pre-learning scores
    [0.68, 0.75, 0.40],  # after task 2
    [0.61, 0.70, 0.82],  # after task 3
]
baselines = [0.10, 0.20, 0.30]

metrics = ContinualLearningMetrics(accuracies, baseline_accuracies=baselines)
print(metrics.summary())
```

The summary contains the average-accuracy trajectory and its normalized area, final average accuracy, learning accuracy, backward transfer, average forgetting, and forward transfer. Forward transfer requires one baseline score per task. BWT, forgetting, FWT, and AUC are undefined for a single task; the summary reports `None` for those values.

Individual functions are also available from their modules, for example `CREvaluator.metrics.backward_transfer.bwt_metric(matrix)`, `CREvaluator.metrics.precision.precision(y_true, y_pred)`, and `CREvaluator.metrics.recall.recall(y_true, y_pred)`.

The legacy JSON workflow remains available:

```python
from CREvaluator.metrics import backward_transfer

score = backward_transfer.bwt("results", "data", number_of_tasks=10)
```

## Metrics

The library provides average accuracy and error, learning accuracy, AUC, backward and forward transfer, per-task and average forgetting, precision, recall, whole/seen/unseen-class accuracy, RMSE, reward averaging, and elapsed-time measurement.

