# CHIP_MTR_2 Monitor

Evaluates AI-vs-HITL concordance using classification metrics and class balance context.

## What this monitor tells you
- Measures concordance between AI and HITL labels using classification metrics.
- Surfaces class balance and reviewer-level activity context for interpretability.

## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Schema Asset:** Parsed by `infer.validate_schema()` to map score/label/predictor roles.
- **Test Results Output:** `CHIP_mtr_2_test_results.json` written by local smoke runs.

## Runtime Initialization Contract
- Entry points are `init(job_json)` and `metrics(dataframe)`.
- `init()` stores the runtime `JOB` object, validates schema via `infer.validate_schema(job_json)`, and parses `rawJson.jobParameters` with defaults.
- Optional job parameters:
  - `AI_FAIL_VALUES` (default: `["FAIL"]`) for AI predicted positive class mapping.
  - `HITL_POSITIVE_VALUES` (default: `["REJECTED", "REPROCESS", "PENDING"]`) for HITL positive class mapping.
- Runtime assets in this folder:
  - baseline: `CHIP_mtr_2_baseline.json` / `CHIP_mtr_2_baseline.csv`
  - comparator (used by `metrics`): `CHIP_mtr_2_comparator.json` / `CHIP_mtr_2_comparator.csv`
  - test results: `CHIP_mtr_2_test_results.json`

## UI Output Interpretation
- **Generic Table:** Confusion entries, key scores (accuracy/precision/recall/F1/AUC), reviewer/activity totals, date window.
- **Generic Bar Graph / Horizontal Bar Graph:** Primary concordance metrics (`data1`) for quick comparison.
- **Generic Pie/Donut:** Comparator class balance.

## Known Caveat
- AUC can be `null` when comparator contains only one effective class.

## Data Notes
- Baseline and comparator exports are full-dimensional batch data.
- Monitor scripts pre-filter columns needed by their metrics functions.
- `CHIP_data/CHIP_master.*` is always refreshed by preprocess runs.

