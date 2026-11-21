<div align="center">
    <img src="docs/assets/crevaluator-logo.png" width="640" height="320" alt="CREvaluator: Continual learning">
    <h1>CREvaluator</h1>
    <p>Continual learning evaluation metrics for Python</p>
    <p>
        <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB?logo=python&logoColor=white"></a>
        <a href="https://github.com/sefeoglu/cre-evaluation-toolkit/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-2A7A62.svg"></a>
        <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-426A8C.svg">
        <img alt="Development status: alpha" src="https://img.shields.io/badge/status-alpha-C47A32.svg">
    </p>
</div>

CREvaluator is a lightweight library for measuring performance across sequential learning tasks. It provides a unified API for accuracy trajectories, transfer, forgetting, and additional classification and regression metrics.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Metric conventions](#metric-conventions)
- [Citation](#citation)
- [License](#license)

## Installation

Install from a local checkout:

```bash
git clone https://github.com/sefeoglu/cre-evaluation-toolkit.git
cd cre-evaluation-toolkit
python -m pip install .
```

For development, use `python -m pip install -e .`.

## Quickstart

Pass a square matrix to `ContinualLearningMetrics`. Each row is an evaluation checkpoint after learning a task; each column is an evaluation task. Scores are fractions from 0 to 1.

```python
from CREvaluator.metrics import ContinualLearningMetrics

accuracies = [
        [0.80, 0.35, 0.25],  # after task 1; future-task scores support FWT
        [0.68, 0.75, 0.40],  # after task 2
        [0.61, 0.70, 0.82],  # after task 3
]
baselines = [0.10, 0.20, 0.30]

metrics = ContinualLearningMetrics(
        accuracies,
        baseline_accuracies=baselines,
)
report = metrics.summary()
print(report["average_accuracy"])
print(report["backward_transfer"])
print(report["forward_transfer"])
```

The summary includes final and per-checkpoint average accuracy, normalized AUC, learning accuracy, backward transfer, average forgetting, and forward transfer. Forward transfer requires one baseline accuracy per task. A one-task sequence has no transfer, forgetting, or multi-checkpoint AUC; those summary values are `None`.

Standalone metrics remain available from their modules:

```python
from CREvaluator.metrics.precision import precision
from CREvaluator.metrics.recall import recall

macro_precision = precision(y_true, y_pred)
macro_recall = recall(y_true, y_pred)
```

The original JSON-backed BWT workflow is also supported:

```python
from CREvaluator.metrics import backward_transfer

bwt = backward_transfer.bwt("results", "data", number_of_tasks=10)
```

## Metric conventions

| Metric | Definition |
| --- | --- |
| Average accuracy | Mean accuracy on tasks learned at each checkpoint; the final value is the final average accuracy. |
| Learning accuracy | Mean diagonal score, measured immediately after each task is learned. |
| Backward transfer (BWT) | Mean final accuracy minus accuracy immediately after learning, over all earlier tasks. |
| Forward transfer (FWT) | Mean pre-learning accuracy minus task baseline for each later task. |
| Forgetting | Best prior post-learning accuracy minus final accuracy, reported per task and averaged. |
| AUC | Trapezoidal area under the average-accuracy trajectory; normalized by default in the aggregate API. |
| Additional metrics | Precision, recall, whole/seen/unseen-class accuracy, average error, RMSE, reward, and elapsed time. |

For FWT, provide pre-learning scores in the matrix cells for future tasks and pass the corresponding baseline scores as `baseline_accuracies`. Use `None` only for matrix cells that were not evaluated.

## Citation

If you use CREvaluator in research, cite the software and include the version used. No DOI or separate publication is currently assigned.

```bibtex
@software{efeoglu2025crevaluator,
    author  = {Efeoglu, Sefika},
    title   = {CREvaluator: Continual Learning Evaluation Metrics},
    year    = {2025},
    version = {0.1.0},
    url     = {https://github.com/sefeoglu/cre-evaluation-toolkit}
}
```

## License

CREvaluator is distributed under the [MIT License](LICENSE). Copyright (c) 2025 Sefika Efeoglu. See the license file for the complete terms.

