# Homoscedasticity Metrics: Engle's Lagrange Multiplier Test

This ModelOp Center monitor performs an **Engle's Lagrange Multiplier** test.

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
3. The **statsmodels** **Engle's Lagrange Multiplier** test is run.
4. Test results are appended to the list of `homoscedasticity` tests to be returned by the model.

## Monitor Output

```JSON
{
    "engle_lm_statistic": <engle_lagrange_multiplier_statistic>, 
    "engle_lm_p_value": <engle_lagrange_multiplier_p_value>,
    "homoscedasticity":[
        {
            "test_name": "Engle's Lagrange Multiplier",
            "test_category": "homoscedasticity",
            "test_type": "engle_lagrange_multiplier",
            "test_id": "homoscedasticity_engle_lagrange_multiplier",
            "values": {
                "lm_statistic": <engle_lagrange_multiplier_statistic>, 
                "lm_p_value": <engle_lagrange_multiplier_p_value>
            }
        }
    ]
}
```