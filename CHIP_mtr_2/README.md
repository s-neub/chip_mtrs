# CHIP_MTR_2 Monitor

Evaluates operational performance (Accuracy, Precision, Recall) against Human-In-The-Loop Ground Truth.

## Required Assets
- **Baseline Data:** Historical dataset for establishing the baseline.
- **Comparator Data:** Production dataset to be evaluated.
- **Schema Asset:** Used by `infer.validate_schema()` to identify role assignments.

## Full-dimensional data
Baseline and comparator contain all batch-related columns. This monitor may pre-filter to the columns it needs in `init`/`metrics`.
