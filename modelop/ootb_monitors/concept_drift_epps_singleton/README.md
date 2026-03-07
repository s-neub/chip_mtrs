# Concept Drift Monitor: Epps-Singleton
This ModelOp Center monitor computes the **Epps-Singleton p-value** on **output** data.

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
3. The **Epps-Singleton** concept drift test is run.
4. Test result is returned under the list of `concept_drift` tests.

## Monitor Output

```JSON
{
    "concept_drift":[
        {
            "test_name": "Epps-Singleton",
            "test_category": "concept_drift",
            "test_type": "epps_singleton",
            "metric": "p_value",
            "test_id": "concept_drift_epps_singleton_p_value",
            "values": {<score_column>: <p-value>},
        }
    ]
}
```