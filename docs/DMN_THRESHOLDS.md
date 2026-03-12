# CHIP Monitor DMN Thresholds

CHIP monitors use CHIP-specific DMN files for pass/fail and category/action. Thresholds are aligned with ModelOp OOTB monitors and with model ops guidance.

## Input variables (from monitor `metrics()` output)

- **M1 / M3:** `stability` (array with `[0].values` = map of feature → `{ stability_index }`), `data_drift` (array of tests; Kolmogorov-Smirnov has `.values` with feature → p-value), optional `modelRisk` ("HIGH" | "MEDIUM" | "LOW").
- **M2:** `auc`, `precision`, `recall`, `f1_score` (numbers).

## Thresholds

| Monitor | DMN file | Criteria | Thresholds |
|--------|----------|----------|------------|
| M1 (AI stability/drift) | `CHIP_mtr_1/CHIP_M1_Stability_Drift.dmn` | Stability (CSI/PSI) | Index > 0.15 (HIGH), > 0.2 (MEDIUM), > 0.25 (LOW) → fail |
| M1 | same | Data drift (Kolmogorov–Smirnov) | Any feature p-value < 0.05 → fail |
| M2 (Performance) | `CHIP_mtr_2/CHIP_M2_Performance.dmn` | Classification | auc ≤ 0.7, precision ≤ 0.6, recall ≤ 0.6, f1_score ≤ 0.7 → fail |
| M3 (HITL stability) | `CHIP_mtr_3/CHIP_M3_HITL_Stability_Drift.dmn` | Same as M1 | Same stability and KS drift thresholds |

## cm_term and threshold evolution

- **cm_term:** In ModelOp, the comparator term (e.g. baseline vs comparator window) is set by the split (e.g. `date-30`, `volume-5000`) in the preprocess. The same term is reflected in the monitor assets and in CHIP_master (`split_method`, `dataset`).
- **Evolving thresholds:** As production data and ModelOp Customer Success (CS) guidance evolve, update the DMN files (or the platform-linked DMN) and, if needed, job parameters or constants. Document any change and the rationale (e.g. “KS p-value threshold relaxed to 0.01 for feature X per CS”) in this doc or in a change log.
