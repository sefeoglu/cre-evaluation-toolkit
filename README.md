# CREvaluator: An Evaluation Toolkit for Continual Relation Extraction

CREvaluator provides various continual learning metrics to evaluate continual learning performance. Additionally, it 
provides task incremental learning models and visualization module for evaluation of previously seen tasks across task incremantal learning.

````xml
.
├── LICENSE
├── README.md
├── data                                      -> data (ground truths) will be here for evaluation or give your ground truth path to metrics_settings.yaml
├── results                                   -> you can put your results (predictions) or training your data with our models.
├── settings                                  -> task incremental training and metrics settings.
│   ├── metrics_settings.yaml
│   └── training_settings.yaml
└── src
    ├── metrics                                       -> support various metrics for evaluation of continual relation extraction.
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
    ├── training                                         -----> You can include your own trainer and models here.
    │   ├── decoder
    │   ├── encoder
    │   ├── encoder_decoder
    │   └── trainer.py
    ├── ui
    │   └── evaluator_ui.py
    └── visualization
        └── model_viz.py
````

## Usage

## Result

## Demo
## Cite

