# Multicollinearity Metrics: Variance Inflation Factor

This ModelOp Center monitor computes Variance Inflation Factors.

## Input Assets

| Type | Number | Description |
| ------ | ------ | ------ |
| Baseline Data | **0** | |
| Comparator Data | **1** |  A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has a **job json** asset.
 - `BUSINESS_MODEL` is a **regression** model.
 - Input data must contain:
     - At least 1 column with **dataclass=numerical**.

## Execution
1. `init` function accepts the job json asset and validates the input schema (corresponding to the `BUSINESS_MODEL` being monitored).
2. `metrics` function instantiates the **Multicollinearity Metrics** class and uses the job json asset to determine the `label_column` and `numerical_columns` (predictors) accordingly.
3. The **statsmodels** **Variance Inflation Factor** test is run.
4. Test results are appended to the list of `multicollinearity` tests to be returned by the model.

## Monitor Output

```JSON
{
    "multicollinearity":[
        {
            "test_name": "Variance Inflation Factor",
            "test_category": "multicollinearity",
            "test_type": "variance_inflation_factor",
            "test_id": "multicollinearity_variance_inflation_factorn",
            "values": {
                <predictor_1>: <variance_inflation_factor>,
                ...:...,
                <predictor_N>: <variance_inflation_factor>
            }
        }
    ]
}
```