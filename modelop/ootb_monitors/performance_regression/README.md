# Performance Monitor: Regression
This ModelOp Center monitor computes regression metrics such as **MAE**, **RMSE**, and **r2_score**.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **0**  |                                                       |
| Comparator Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - `BUSINESS_MODEL` is a **regression** model.
 - Input data must contain:
     - 1 column with **role=label** (ground truth) and **dataClass=numerical**
     - 1 column with **role=score** (model output) and **dataClass=numerical**

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Model Evaluator** class and uses the job json asset to determine the `label_column` and `score_column` accordingly.
3. The **regression performance** test is run.
4. Test results are appended to the list of `performance` tests to be returned by the model, and key:value pairs are added to the top-level of the output dictionary.

## Monitor Output

```JSON
{
    "mae": <mae>,
    "rmse": <rmse>,
    "r2_score": <r2_score>,
    "performance": [
        {
            "test_category": "performance",
            "test_name": "Regression Metrics",
            "test_type": "regression_metrics",
            "test_id": "performance_regression_metrics",
            "values": {
                "mae": <mae>,
                "rmse": <rmse>,
                "r2_score": <r2_score>
            }
        }
    ]
}
```