# Autocorrelation Metrics: Durbin-Watson Test

This ModelOp Center monitor performs a **Durbin-Watson** test.

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
2. `metrics` function instantiates the **Autocorrelation Metrics** class and uses the job json asset to determine the `label_column` and `score_column` accordingly.
3. The **statsmodels** **Durbin-Watson** test is run.
4. Test results are appended to the list of `autocorrelation` tests to be returned by the model.

## Monitor Output

```JSON
{
    "dw_statistic": <durbin_watson_statistic>, 
    "autocorrelation":[
        {
            "test_name": "Durbin-Watson",
            "test_category": "autocorrelation",
            "test_type": "durbin_watson",
            "test_id": "autocorrelation_durbin_watson",
            "values": {
                "dw_statistic": <durbin_watson_statistic>,
            }
        }
    ]
}
```