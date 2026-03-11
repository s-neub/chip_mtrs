# Performance Monitor: Classification
This ModelOp Center monitor computes classification metrics such as **AUC**, **Accuracy**, **Precision**, **Recall**, and **F1_score**.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **0**  |                                                       |
| Comparator Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - `BUSINESS_MODEL` is a **classification** model.
 - Input data must contain:
     - 1 column with **role=label** (ground truth) and **dataClass=categorical**
     - 1 column with **role=score** (model output) and **dataClass=categorical**

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Model Evaluator** class and uses the job json asset to determine the `label_column` and `score_column` accordingly.
3. The **classification performance** test is run.
4. Test results are appended to the list of `performance` tests to be returned by the model, and key:value pairs are added to the top-level of the output dictionary.

## Monitor Output

```JSON
{
    "accuracy": <accuracy>,
    "auc": <auc>,
    "f1_score": <f1_score>,
    "precision": <precision>,
    "recall": <recall>,
    "confusion_matrix": <confusion_matrix>,
    "performance": [
        {
            "test_category": "performance",
            "test_name": "Classification Metrics",
            "test_type": "classification_metrics",
            "test_id": "performance_classification_metrics",
            "values": {
                "accuracy": <accuracy>,
                "auc": <auc>,
                "f1_score": <f1_score>,
                "precision": <precision>,
                "recall": <recall>,
                "confusion_matrix": <confusion_matrix>
            }
        }
    ]
}
```