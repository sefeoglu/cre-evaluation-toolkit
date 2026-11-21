from collections.abc import Iterable
from math import isfinite


def numeric_values(values: Iterable[float], name: str = "values") -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def score_matrix(values: Iterable[Iterable[float | None]]) -> list[list[float | None]]:
    matrix = [[None if value is None else float(value) for value in row] for row in values]
    if not matrix or not matrix[0]:
        raise ValueError("The score matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("The score matrix must be rectangular")
    if any(value is not None and not isfinite(value) for row in matrix for value in row):
        raise ValueError("The score matrix must contain only finite numbers or None")
    return matrix


def mean(values: Iterable[float], name: str = "values") -> float:
    scores = numeric_values(values, name)
    return sum(scores) / len(scores)


def paired_labels(y_true: Iterable[object], y_pred: Iterable[object]) -> tuple[list[object], list[object]]:
    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        raise ValueError("Labels must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    return actual, predicted


def accuracy_matrix(values: Iterable[Iterable[float | None]]) -> list[list[float | None]]:
    matrix = score_matrix(values)
    if any(value is not None and not 0 <= value <= 1 for row in matrix for value in row):
        raise ValueError("Accuracy values must be between 0 and 1")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The accuracy matrix must be square")
    return matrix