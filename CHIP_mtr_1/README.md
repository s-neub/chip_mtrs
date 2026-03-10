# CHIP_MTR_1 Monitor

Tracks AI output stability and drift between baseline and comparator windows.

## What this monitor tells you
- Detects shifts between baseline and comparator for AI outputs and related dimensions.
- Highlights which features are most unstable (CSI) and how that aligns with drift distance (JS).

## Required Assets
- **Baseline Data:** Historical records used to establish expected behavior.
- **Comparator Data:** Recent production records used for current evaluation.
- **Schema Asset:** Parsed by `infer.validate_schema()` to map score/label/predictor roles.

## UI Output Interpretation
- **Generic Table:** High-level summary (largest/smallest CSI, overall PSI, date windows).
- **Generic Bar Graph / Horizontal Bar Graph:** Side-by-side CSI (`data1`) and JS distance (`data2`) by feature.
- **Generic Scatter Plot:** CSI vs JS relationship across top features.
- **Generic Pie/Donut:** Comparator AI outcome mix.

## Data Notes
- Baseline and comparator exports are full-dimensional batch data.
- Monitor scripts pre-filter columns needed by their metrics functions.
- `CHIP_data/CHIP_master.*` is always refreshed by preprocess runs.

