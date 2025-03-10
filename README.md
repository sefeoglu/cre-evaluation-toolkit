# CREvaluator: An Evaluation Toolkit for Continual Relation Extraction
[![Python 3.12.7](https://img.shields.io/badge/python-3.12.7-blue.svg)](https://www.python.org/downloads/release/python-3127/)


CREvaluator provides various continual learning metrics to evaluate continual learning performance. Additionally, it 
provides task incremental learning models and visualization module for evaluation of previously seen tasks across task incremantal learning.



## CREvaluator 

### Installation
````xml
$ git clone https://github.com/sefeoglu/cre-evaluation-toolkit.git
$ cd  cre-evaluation-toolkit
$ pip install .
$ pip install --upgrade .
````

### Uninstall
````xml
$ pip uninstall CREvaluator

````
### Usage
````python
from CREvaluator.metrics import backward_transfer

result_folder = "results"
ground_truth_folder ="data"
number_of_tasks = 10
bwt = backward_transfer.bwt(result_folder, ground_truth_folder, number_of_tasks)
````

## Folder Structure
````xml
.
├── LICENSE
├── README.md
├── logs            
├── data                                      -> data (ground truths) will be here for evaluation or give your ground truth path to metrics_settings.yaml
├── results                                   -> you can put your results (predictions) or training your data with our models.
├── settings                                  -> task incremental training and metrics settings.
│   ├── metrics_settings.yaml
│   └── training_settings.yaml
└── src
    ├── metrics                               ->support various metrics for evaluation of continual relation extraction.
    │   ├── average_accuracy.py
    │   ├── average_error.py
    │   ├── average_forgetting.py
    │   ├── bwt.py
    │   ├── forgetting_measure.py
    │   ├── forward_transfer.py
    │   ├── learning_accuracy.py
    │   ├── whole_accuracy.py
    │   ├── time_evaluator.py
    │   ├── seen_class_accuracy.py
    │   ├── unseen_class_accuracy.py
    │   └── ....
    ├── task_preparation
    │   └── task_data.py
    ├── training                              -> You can include your own trainer and models here.
    │   ├── decoder
    │   ├── encoder
    │   ├── encoder_decoder
    │   └── trainer.py
    ├── ui                                    -> User Interface for interative evaluation
    │   └── evaluator_ui.py
    └── visualization                         -> prepare figures for results.
        └── model_viz.py
````

## Result

## Demo
## Cite

