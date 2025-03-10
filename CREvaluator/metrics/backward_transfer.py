"""Compute Backward Transfer (BWT) metric."""


import os
import sys
import json
import numpy as np
import logging
from sklearn.metrics import accuracy_score


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"
sys.path.append(PREFIX_PATH)

logging.basicConfig(filename=PREFIX_PATH + 'logs/backward_transfer.log', format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger("BWTlogger")

def compute_accuracy(y_true, results):
    y_pred = [ pred for pred in results]
    return accuracy_score(y_true, y_pred)
    

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def bwt(result_folder, ground_truth_folder, number_of_tasks):

    results = []
    last_task_results = []
    run = []

    try:
        for task_id in range(1, number_of_tasks+1):
            task = result_folder + f"/task_{task_id}/current_result.json"
            result = read_json(task)
            results.append(result[0]['acc'])
            
            #last task seen predictions
            task = result_folder+f"/task_{task_id}/seen_task_predict.json"
            last_task = read_json(task)
            end = 0
            start = 0
            for task_id in range(1, number_of_tasks+1):
                y_true_path = f"{ground_truth_folder}/task_{task_id}/test.json"
                y_true = read_json(y_true_path)
                end = start + len(y_true)
                acc = compute_accuracy(y_true, last_task[start:end])
                start = end
                last_task_results.append(acc)
            results.append(last_task_results)
            
            run.append({'results':results})
    except:
        if not os.path.exists(result_folder):
            logger.error(f"BWT: Folder {result_folder} does not exist")
        if not os.path.exists(ground_truth_folder):
            logger.error(f"BWT: Folder {ground_truth_folder} does not exist")
    
        logger.warn("BWT: Error while reading results")
        logger.warn("BWT: Please make sure that the results are in the correct format")
        
        return None
    if len(run) == 0:
        logger.warn("BWT: No results found")
        return None
    else:
        bwt = bwt_metric(run)
        return bwt
    
    

def bwt_metric(accuracies):
    """
    Calculate the Backward Transfer (BWT) metric.

    Parameters:
    - accuracies: A 2D list or numpy array where accuracies[N-1][t] is the test accuracy
                  on task t after sequential training on all N tasks, and accuracies[t][t]
                  is the test accuracy on task t immediately after it was learned.

    Returns:
    - BWT: The Backward Transfer metric.
    """
    N = len(accs)
    accuracies = np.array(accuracies)
    accs = accuracies['results'][:N-1]
    seen_task = accuracies['results'][N-1]


    bwt = 0
    for t in range(0, N):
        A_N_t = accs[t]  # Accuracy on task t after all tasks
        A_t_t = seen_task[t]
        bwt += (A_N_t - A_t_t)


    bwt /= (N - 1)  # Averaging the differences

    return bwt

# if __name__ == "__main__":
#     result_folder = "./results"
#     ground_truth_folder ="./data"
#     number_of_tasks = 10
#     bwt = bwt(result_folder, ground_truth_folder, number_of_tasks)