# Linearity Metrics: Box-Tidwell

This ModelOp Center monitor computes the Box-Tidwell test.

## Input Assets

| Type | Number | Description |
| ------ | ------ | ------ |
| Baseline Data | **0** | |
| Comparator Data | **1** |  A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - `BUSINESS_MODEL` is a **regression** model.
 - Input data must contain:
     - 1 column with **role=label** (ground truth) 
     - At least 1 column with **dataclass=numerical**.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Linearity Metrics** class and uses the job json asset to determine the `label_column` and `numerical_columns` (predictors) accordingly.
3. The **statsmodels** custom **Box-Tidwell** test is run.
4. Test results are appended to the list of `linearity` tests to be returned by the model.

## Monitor Output

```JSON
{
    "linearity":[
        {
            "test_name": "Box-Tidwell",
            "test_category": "linearity",
            "test_type": "box_tidwell",
            "test_id": "linearity_box_tidwell",
            "metric": "p_value",
            "values": {
                <predictor_1_log_int>: <p_value>,
                ...:...,
                <predictor_N_log_int>: <p_value>
            }
        }
    ]
}
```