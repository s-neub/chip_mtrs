# CHIP_MTR_2 Monitor

Evaluates AI-vs-HITL concordance using classification metrics and class balance context.

## What this monitor tells you
- Measures concordance between AI and HITL labels using classification metrics.
- Surfaces class balance and reviewer-level activity context for interpretability.

## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Schema Asset:** Parsed by `infer.validate_schema()` to map score/label/predictor roles.

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

