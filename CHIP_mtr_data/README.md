# CHIP_mtr_data: ETL / Preprocessing Monitor

This monitor performs ETL (extract, transform, load) specific to the BMS CHIP business model. It consumes raw business data and produces master, baseline, and comparator CSV datasets that feed the downstream monitors CHIP_mtr_1, CHIP_mtr_2, and CHIP_mtr_3.

## What this monitor does

- **Inputs (configure in Add monitor wizard):** Batch activity log JSON, AI feedback JSON, and a directory (or archive) of AI response JSONs. These assets live on the implementation.
- **Outputs (CSV only):**  
  - `CHIP_master.csv` – full dataset with `dataset` and `split_method` columns  
  - `CHIP_baseline.csv` – baseline slice for stability/drift  
  - `CHIP_comparator.csv` – comparator slice for stability/drift/performance  

Outputs are written to the directory specified by job parameters (`output_dir`) or the platform’s output location, and become implementation assets. When adding CHIP_mtr_1, CHIP_mtr_2, and CHIP_mtr_3 to the same implementation, select these CSVs as BASELINE_DATA / COMPARATOR_DATA in each monitor’s wizard.

## Chaining with downstream monitors

1. Add this **CHIP_mtr_data** monitor to the implementation and set its input assets (activity log, feedback, AI responses).
2. Run the preprocessing monitor so it produces the CSVs.
3. Add **CHIP_mtr_1**, **CHIP_mtr_2**, **CHIP_mtr_3** and, in each wizard, choose the preprocessing monitor outputs as the required input assets.

No baseline or comparator data should live inside the monitor repos; it all comes from the implementation.

## Job parameters (optional)

- `output_dir` – where to write the CSV files (default: script directory or `CHIP_ETL_OUTPUT_DIR`)  
- `activity_file`, `feedback_file`, `ai_responses_dir` – override input paths  
- `split_method` – `"DATE"` or `"VOLUME"`  
- `days_threshold`, `volume_threshold`, `baseline_start_date`, `min_records_baseline`, `min_records_comparator`, `config_path` – same as the root preprocess pipeline  

## Required assets

- **Baseline/Reference:** Batch activity log (top-level key `batch_activity_log`).  
- **Comparator 1:** AI feedback (top-level key `ai_feedback`).  
- **Comparator 2:** Directory or archive of AI response JSONs.  

Configure these in the ModelOp Center Add monitor wizard for this monitor.
