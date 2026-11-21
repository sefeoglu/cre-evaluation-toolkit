"""Compute Backward Transfer (BWT) metric."""


import os
import json
import logging
from sklearn.metrics import accuracy_score
from ._metric_utils import accuracy_matrix, mean

logger = logging.getLogger("BWTlogger")

def compute_accuracy(y_true, results):
    return accuracy_score(y_true, results)

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def bwt(result_folder, ground_truth_folder, number_of_tasks):
    try:
        if number_of_tasks < 2:
            raise ValueError("At least two tasks are required to calculate BWT")
        ground_truths = [
            read_json(os.path.join(ground_truth_folder, f"task_{task_id}", "test.json"))
            for task_id in range(1, number_of_tasks + 1)
        ]
        matrix = [[None] * number_of_tasks for _ in range(number_of_tasks)]

        for checkpoint in range(1, number_of_tasks + 1):
            predictions = read_json(
                os.path.join(result_folder, f"task_{checkpoint}", "seen_task_predict.json")
            )
            expected_count = sum(len(ground_truths[index]) for index in range(checkpoint))
            if len(predictions) != expected_count:
                raise ValueError(
                    f"Task {checkpoint} has {len(predictions)} predictions; expected {expected_count}"
                )

            offset = 0
            for task_index in range(checkpoint):
                task_labels = ground_truths[task_index]
                end = offset + len(task_labels)
                matrix[checkpoint - 1][task_index] = compute_accuracy(
                    task_labels, predictions[offset:end]
                )
                offset = end

        return bwt_metric(matrix)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        logger.warning("BWT could not be calculated: %s", error)
        return None

def bwt_metric(accuracies):
    """Return mean final-minus-learned accuracy for all earlier tasks."""
    matrix = accuracy_matrix(accuracies)
    if len(matrix) < 2:
        raise ValueError("At least two tasks are required to calculate BWT")
    differences = []
    final_row = matrix[-1]
    for task_index in range(len(matrix) - 1):
        learned_score = matrix[task_index][task_index]
        final_score = final_row[task_index]
        if learned_score is None or final_score is None:
            raise ValueError("The matrix must include learned and final scores for prior tasks")
        differences.append(final_score - learned_score)
    return mean(differences, "backward-transfer scores")

# if __name__ == "__main__":
#     result_folder = "./results"
#     ground_truth_folder ="./data"
#     number_of_tasks = 10
#     bwt = bwt(result_folder, ground_truth_folder, number_of_tasks)