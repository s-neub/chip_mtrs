# Normality Metrics: Kolmogorov-Smirnov Test

This ModelOp Center monitor performs a **Kolmogorov-Smirnov** test.

## Input Assets

| Type | Number | Description |
| ------ | ------ | ------ |
| Baseline Data | **0** | |
| Comparator Data | **1** |  A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has an **extended input schema** asset.
 - `BUSINESS_MODEL` is a **regression** model.
 - Input data must contain:
     - 1 column with **role=label** (ground truth) 
     - 1 column with **role=score** (model output) 

## Execution

1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Normality Metrics** class and uses the job json asset to determine the `label_column` and `scorE_column` accordingly.
3. The **statsmodels** **Kolmogorov-Smirnov** test is run.
4. Test results are appended to the list of `normality` tests to be returned by the model.

## Monitor Output

```JSON
{
    "ks_statistic": <kolmogorov_smirnov_statistic>, 
    "ks_p_value": <kolmogorov_smirnov_p_value>,
    "normality":[
        {
            "test_name": "Kolmogorov-Smirnov",
            "test_category": "normality",
            "test_type": "kolmogorov_smirnov",
            "test_id": "normality_kolmogorov_smirnov",
            "values": {
                "ks_statistic": <kolmogorov_smirnov_statistic>,
                "ks_p_value": <kolmogorov_smirnov_p_value>,
            }
        }
    ]
}
```