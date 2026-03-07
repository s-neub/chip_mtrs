# Concept Drift Monitor: Kolmogorov-Smirnov
This ModelOp Center monitor computes the **Kolmogorov-Smirnov p-value** on **output** data.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | A dataset corresponding to training/reference data    |
| Comparator Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - Input data must contain:
     - 1 column with **role=score** (model output) 

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Concept Drift Monitor** class and uses the job json asset to set the `target_column` (score column) and `output_type` (`numerical` vs. `categorical`).
3. The **Kullback-Leibler** concept drift test is run.
4. Test result is returned under the list of `concept_drift` tests.

## Monitor Output

```JSON
{
    "concept_drift":[
        {
            "test_name": "Kolmogorov-Smirnov",
            "test_category": "concept_drift",
            "test_type": "kolmogorov_smirnov",
            "metric": "p_value",
            "test_id": "concept_drift_kolmogorov_smirnov_p_value",
            "values": {<score_column>: <p-value>},
        }
    ]
}
```