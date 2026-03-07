# Concept Drift Monitor: Kullback-Leibler
This ModelOp Center monitor computes the **Kullback-Leibler divergence** on **output** data.

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
3. The **Kolmogorov-Smirnov** concept drift test is run.
   - If `output_type` is `numerical`, data is bucketed into 5 bins by default.
   - If `output_type` is `categorical`, data is bucketed by category.
4. Test result is returned under the list of `concept_drift` tests.

## Monitor Output

```JSON
{
    "concept_drift":[
        {
            "test_name": "Kullback-Leibler",
            "test_category": "concept_drift",
            "test_type": "kullback_leibler",
            "metric": "divergence",
            "test_id": "concept_drift_kullback_leibler_divergence",
            "values": {<score_column>: <divergence>},
        }
    ]
}
```