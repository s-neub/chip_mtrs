# Normality Metrics: Anderson-Darling Test

This ModelOp Center monitor performs a **Anderson-Darling** test.

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

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Normality Metrics** class and uses the job json asset to determine the `label_column` and `scorE_column` accordingly.
3. The **statsmodels** **Anderson-Darling** test is run.
4. Test results are appended to the list of `normality` tests to be returned by the model.

## Monitor Output

```JSON
{
    "ks_statistic": <anderson_darling_statistic>, 
    "ks_p_value": <anderson_darling_p_value>,
    "normality":[
        {
            "test_name": "Anderson-Darling",
            "test_category": "normality",
            "test_type": "anderson_darling",
            "test_id": "normality_anderson_darling",
            "values": {
                "ks_statistic": <anderson_darling_statistic>,
                "ks_p_value": <anderson_darling_p_value>,
            }
        }
    ]
}
```