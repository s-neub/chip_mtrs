# CHIP_MTR_1 Monitor

Tracks the behavior and drift of the Claude AI model's output over time using Population Stability Index (PSI) and Data Drift methods.

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Execution
1. The `init` function reads the schema asset to identify predictors, scores, and labels.
2. The `metrics` function computes the test results and yields the JSON payload.
