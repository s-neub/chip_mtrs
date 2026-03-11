# Stability Monitor: PSI/CSI
This ModelOp Center monitor computes **stability** metrics, including Population Stability Index (**PSI**) and Characteristic Stability Indices (**CSI**), and their breakdown by buckets.

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
2. `metrics` function instantiates the **Stability Monitor** class and uses the job json asset to determine the `predictors`, `feature_dataclass`, `special_values`, `score_column`, and `weight_column` accordingly.
3. The **stability analysis** test is run.
   - For each `categorical` feature, the number of groups (`n_groups`) to break the data into is set by default to be equal to the number of unique values of this feature.
   - For each `numerical` feature, `n_groups` is set to **2** if this feature has more than 1 unique value. Otherwise, `n_groups` is set to **1**.
4. Test results are appended to the list of `stability` tests to be returned by the model.

## Monitor Output

```JSON
{
    "stability": [
        {
            "test_name": "Stability Analysis",
            "test_category": "stability",
            "test_type": "stability_analysis",
            "test_id": "stability_stability_analysis",
            "values": {
                <predictive_feature_1>: {
                    "stability_analysis_table": <stability_analysis_table>,
                    "stability_index": <stability_index>,
                    "stability_chisq": <stability_chisq>,
                    "stability_ks": <stability_ks>
                },
                ...:...,
                <predictive_feature_n>: {
                    "stability_analysis_table": <stability_analysis_table>,
                    "stability_index": <stability_index>,
                    "stability_chisq": <stability_chisq>,
                    "stability_ks": <stability_ks>
                },
                <score_column>: {
                    "stability_analysis_table": <stability_analysis_table>,
                    "stability_index": <stability_index>,
                    "stability_chisq": <stability_chisq>,
                    "stability_ks": <stability_ks>
                }
            }
        }
    ]
}
```