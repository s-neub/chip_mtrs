# Concept Drift Monitor: Comprehensive Analysis
This ModelOp Center monitor runs and compares **Kolmogorov-Smirnov**, **Epps-Singleton**, **Jensen-Shannon**, **Kullback-Leibler**, and **Pandas summary** on **output** data.

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
3. The **Epps-Singleton**, **Jensen-Shannon**, **Kullback-Leibler**, **Kolmogorov-Smirnov**, and **Pandas.describe()** concept drift tests are run.
4. Each test result is appended to the list of `concept_drift` tests to be returned by the model.

## Monitor Output

```JSON
{
    "concept_drift": [
        <epps_singleton_test_result>,
        <jensen_shannon_test_result>,
        <kullback_leibler_test_result>,
        <kolmogorov_smirnov_test_result>,
        <pandas_summary_test_result>
    ]
}
```