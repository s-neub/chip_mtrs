# Homoscedasticity Metrics: Breusch-Pagan Test

This ModelOp Center monitor performs a **Breusch-Pagan** test.

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
3. The **statsmodels** **Breusch-Pagan** test is run.
4. Test results are appended to the list of `homoscedasticity` tests to be returned by the model.

## Monitor Output

```JSON
{
    "breusch_pagan_lm_statistic": <breusch_pagan_lagrange_multiplier_statistic>, 
    "breusch_pagan_lm_p_value": <breusch_pagan_lagrange_multiplier_p_value>,
    "breusch_pagan_f_statistic": <breusch_pagan_f_statistic>,
    "breusch_pagan_f_p_value":  <breusch_pagan_f_p_value>,
    "homoscedasticity":[
        {
            "test_name": "Breusch-Pagan",
            "test_category": "homoscedasticity",
            "test_type": "breusch_pagan",
            "test_id": "homoscedasticity_breusch_pagan",
            "values": {
                "lm_statistic": <breusch_pagan_lagrange_multiplier_statistic>,
                "lm_p_value": <breusch_pagan_lagrange_multiplier_p_value>,
                "f_statistic": <breusch_pagan_f_statistic>,
                "f_p_value":  <breusch_pagan_f_p_value>
            }
        }
    ]
}
```