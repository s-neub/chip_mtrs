# CHIP_MTR_2 Monitor

Evaluates the operational performance of the AI by calculating Concordance (Accuracy, Precision, Recall) against Human-In-The-Loop Ground Truth.

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Execution
1. The `init` function reads the schema asset to identify predictors, scores, and labels.
2. The `metrics` function computes the test results and yields the JSON payload.
