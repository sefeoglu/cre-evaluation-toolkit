"""Visualization helpers for continual-learning evaluation results."""

from collections.abc import Iterable, Sequence
from pathlib import Path


def _axes_module():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization support requires matplotlib. Install CREvaluator with its dependencies."
        ) from exc
    return plt


def plot_accuracy_matrix(
    accuracies: Iterable[Iterable[float | None]],
    *,
    title: str = "Task Accuracy by Checkpoint",
    ax=None,
    cmap: str = "viridis",
):
    """Plot a continual-learning accuracy matrix as a heatmap.

    Rows are learning checkpoints and columns are evaluation tasks. ``None``
    cells are left transparent in the heatmap.
    """
    plt = _axes_module()
    rows = [list(row) for row in accuracies]
    if not rows:
        raise ValueError("Accuracy matrix cannot be empty")
    if len(rows) != len(rows[0]):
        raise ValueError("Accuracy matrix must be square")
    if any(len(row) != len(rows) for row in rows):
        raise ValueError("Accuracy matrix must be square")

    import numpy as np

    matrix = np.array(
        [[np.nan if value is None else float(value) for value in row] for row in rows],
        dtype=float,
    )
    if np.any((~np.isnan(matrix)) & ((matrix < 0) | (matrix > 1))):
        raise ValueError("Accuracy scores must be between 0 and 1")

    if ax is None:
        _, ax = plt.subplots()

    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Learning checkpoint")
    ax.set_xticks(range(len(rows)), labels=[f"T{i + 1}" for i in range(len(rows))])
    ax.set_yticks(range(len(rows)), labels=[f"C{i + 1}" for i in range(len(rows))])

    for row_index in range(len(rows)):
        for column_index in range(len(rows)):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                continue
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value < 0.55 else "black",
            )

    ax.figure.colorbar(image, ax=ax, label="Accuracy")
    return ax


def plot_average_accuracy_curve(
    metrics,
    *,
    title: str = "Average Accuracy",
    ax=None,
):
    """Plot average accuracy at each learning checkpoint."""
    plt = _axes_module()
    curve = metrics.average_accuracy_curve()

    if ax is None:
        _, ax = plt.subplots()

    checkpoints = range(1, len(curve) + 1)
    ax.plot(checkpoints, curve, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Learning checkpoint")
    ax.set_ylabel("Average accuracy")
    ax.set_xticks(list(checkpoints))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_summary(summary: dict, *, title: str = "Continual-Learning Summary", ax=None):
    """Plot scalar values from a metric summary as a bar chart."""
    plt = _axes_module()
    labels = (
        "average_accuracy",
        "area_under_curve",
        "learning_accuracy",
        "backward_transfer",
        "average_forgetting",
        "forward_transfer",
    )
    display = [label.replace("_", " ").title() for label in labels]
    values = [summary.get(label) for label in labels]

    available_labels = [label for label, value in zip(display, values) if value is not None]
    available_values = [float(value) for value in values if value is not None]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.bar(available_labels, available_values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=25)
    return ax


def save_figure(ax, path: str | Path, **save_kwargs):
    """Save the figure associated with an Axes to ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(destination, bbox_inches="tight", **save_kwargs)
    return destination
