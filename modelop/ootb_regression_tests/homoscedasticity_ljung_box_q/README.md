# Homoscedasticity Metrics: Ljung-Box Q Test

This ModelOp Center monitor performs a **Ljung-Box Q** test.

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
     - 1 column with **role=score** (model output) 
     - At least 1 column with **dataclass=numerical**.

## Execution

1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Homoscedasticity Metrics** class and uses the job json asset to determine the `label_column`, `score_column`, and `numerical_columns` (predictors) accordingly.
3. The **statsmodels** **Ljung-Box Q** test is run.
4. Test results are appended to the list of `homoscedasticity` tests to be returned by the model.

## Monitor Output

```JSON
{
    "homoscedasticity":[
        {
            "test_name": "Ljung-Box Q",
            "test_category": "homoscedasticity",
            "test_type": "ljung_box_q",
            "test_id": "homoscedasticity_ljung_box_q",
            "values": {
                [
                    {"lag": <min_lag_value>, "lb_statistic": <lb_statistic>, "lb_p_value": <lb_p_value>},
                    ...,
                    {"lag": <max_lag_value>, "lb_statistic": <lb_statistic>, "lb_p_value": <lb_p_value>}
                ]
            }
        }
    ]
}
```