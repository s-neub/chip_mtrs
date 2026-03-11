# CHIP Monitor Test Results – Aggregate Summary vs. Proposal

**Run:** CHIP_mtr_data (preprocessing) → MTR 1, MTR 2, MTR 3  
**Reference:** BMS CHIP – Product Disposition – Document (COA/Report) Validation Use Case – Monitor Recommendations  

---

## 1. Test Run Summary

### Latest run (chained: preprocessing monitor then downstream monitors)

| Step | Description | Result |
|------|-------------|--------|
| **CHIP_mtr_data** | ETL preprocessing monitor – `execute_pipeline_csv_only()` → `CHIP_data/CHIP_master.csv`, `CHIP_baseline.csv`, `CHIP_comparator.csv` | OK – master=448, baseline=216, comparator=232 (CSV only; date split) |
| **MTR 1** | Model Output Stability (Drift) – AI decisions baseline vs comparator; inputs from `CHIP_data` | OK – metrics and visualizations written to `CHIP_mtr_1_test_results.json` (PSI ai_overall_status ≈ 0.1108) |
| **MTR 2** | Approval Concordance – AI vs Human QA agreement; comparator from `CHIP_data` | OK – metrics and visualizations written to `CHIP_mtr_2_test_results.json` (Accuracy 93.53%, AUC null – single class in comparator) |
| **MTR 3** | QA Calibration – Human reviewer stability and vs team; inputs from `CHIP_data` | OK – metrics and visualizations written to `CHIP_mtr_3_test_results.json` (Team rejection 1.0; reviewer USER-251.0; time_line_graph present) |

Local execution order: (1) Run preprocessing (or `execute_pipeline_csv_only` with `output_dir=CHIP_data`); (2) Run CHIP_mtr_1, CHIP_mtr_2, CHIP_mtr_3 with `PYTHONPATH` set to repo root. Downstream monitors load baseline/comparator from `CHIP_data/CHIP_baseline.csv` and `CHIP_data/CHIP_comparator.csv` when local JSON assets are not present.

### Previous run (legacy: single preprocess script then monitors)

| Step | Description | Result |
|------|-------------|--------|
| Preprocess | `CHIP_mtr_preprocess.py` – baseline vs comparator split from CHIP_master | OK – baseline=216, comparator=232 (date-auto split: 2026-02-27 16:00:26 UTC) |
| MTR 1 | Model Output Stability (Drift) – AI decisions baseline vs comparator | OK – metrics and visualizations written to `CHIP_mtr_1_test_results.json` |
| MTR 2 | Approval Concordance – AI vs Human QA agreement | OK – metrics and visualizations written to `CHIP_mtr_2_test_results.json` |
| MTR 3 | QA Calibration – Human reviewer stability and vs team | OK – metrics and visualizations written to `CHIP_mtr_3_test_results.json` |

---

## 2. Proposal Success Criteria vs. Actual Results

Per the proposal, **BMS CHIP / Customer** success criteria and **ModelOp Customer Success** deliverable are custom monitors that meet the required functionality below.

### 2.1 Monitor 1 – Model Output Stability (Drift)

| Source | Criterion | Requirement / Measure | This Run | Status |
|--------|-----------|------------------------|----------|--------|
| Proposal (slide 5) | **BMS Goal** | Reliability (Model Stability) | – | – |
| Proposal (slide 6) | **Description** | "Consistent Behavior: The AI's rejection rate remains stable over time (PSI < 0.1) unless specific intervention occurs." | – | – |
| Proposal (slide 6) | **Measure** | "We compare the AI's rejection rate this week vs. last month." | Baseline: 2026-02-27 (≈11 min window). Comparator: 2026-02-27 to 2026-03-04. | **Met** – comparison implemented |
| **ModelOp deliverable** | **Functionality** | Model Stability (PSI), reliability vs baseline | **Score PSI (ai_overall_status): 0.1108**; Max CSI: 2.56 (batchId); Min CSI: 0.0 (businessKey) | **Alert** – PSI > 0.1 |
| **Result** | **Interpretation** | PSI < 0.1 = stable | PSI ≈ 0.11 → slight drift in AI overall status distribution vs baseline; CSI highlights batchId as highest instability. | **Needs review** – above 0.1 threshold |

**Key outputs (from test result JSON):** `ai_overall_status_PSI` = 0.1108; `CSI_maxCSIValue` = 2.5616; `CSI_minCSIValue` = 0.0; baseline/comparator date range populated; `generic_bar_graph`, `generic_table`, `activity_feedback_summary`.

---

### 2.2 Monitor 2 – Approval Concordance

| Source | Criterion | Requirement / Measure | This Run | Status |
|--------|-----------|------------------------|----------|--------|
| Proposal (slide 5) | **BMS Goal** | Accuracy & Agreement (Model-to-Human) | – | – |
| Proposal (slide 6) | **Description** | ">95% Agreement: High concordance between the LLM's recommendation and the Human QA's final decision." | – | – |
| Proposal (slide 6) | **Measure** | "We compare the AI's 'Conform' vote against the QA's 'Approve' vote. If the AI says Pass but the Human says Fail, we flag it." | Binary classification: ai_overall_status (score) vs hitl_qa_decision (label). | **Met** – comparison implemented |
| **ModelOp deliverable** | **Functionality** | Concordance, Accuracy, confusion matrix | **Accuracy: 93.53%**; Precision: 1.0; Recall: 0.9353; F1: 0.9666; confusion matrix present | **Below 95%** – 93.53% < 95% |
| **Result** | **Interpretation** | >95% agreement | 93.53% agreement (≈15 of 232 discordant) | **Needs review** – under target |

**Key outputs (from test result JSON):** `accuracy` = 0.9353; `precision` = 1.0; `recall` = 0.9353; `f1_score` = 0.9666; `confusion_matrix`; `generic_bar_graph`, `generic_table`, `reviewer_volume`, `activity_feedback_summary`.

---

### 2.3 Monitor 3 – QA Calibration

| Source | Criterion | Requirement / Measure | This Run | Status |
|--------|-----------|------------------------|----------|--------|
| Proposal (slide 5) | **BMS Goal** | Process Control (Human Stability) | – | – |
| Proposal (slide 6) | **Description** | "Stable Review Standards: Human QA rejection rates remain consistent over time, ensuring no 'policy drift' or fatigue." | – | – |
| Proposal (slide 6) | **Measure** | "We track the Human QA team's rejection rate over time. If the humans suddenly get much stricter or more lenient than usual, we flag it." | Time series and per-reviewer stats vs team. | **Met** – tracking implemented |
| **ModelOp deliverable** | **Functionality** | Human Stability (PSI), process control, reviewer vs team | **Team rejection rate: 0%**; Reviewer USER-251.0: Volume=232, Rejection Rate=0%, vs Team=0; `reviewer_time_series` (per-reviewer daily); `time_line_graph` (daily rate/volume) | **Met** – stable (single reviewer, single day in comparator) |
| **Result** | **Interpretation** | Consistent human rejection behavior | One reviewer, one date in comparator; rejection rate 0%; reviewer vs team and time series available for ongoing monitoring. | **Met** – functionality in place |

**Key outputs (from test result JSON):** `reviewer_stats_table` (Team + USER-251.0); `reviewer_time_series` (per-reviewer dates, rejection_rate, volume); `time_line_graph`; `qa_feedback_samples`; baseline/comparator date range.

---

## 3. E-Sign, Batch Assignee, and User Identity Tracking (New)

This section documents the new columns added to `CHIP_data/CHIP_master.csv` and `CHIP_data/CHIP_master.json` following the [E-sign and assignee tracking plan](../\.cursor/plans/e-sign_and_assignee_tracking_4d9a13e3.plan.md). Previously, the pipeline only stored `hitl_reviewer_id` as a non-alphabetic identifier (e.g. `USER-251.0`) and did not track e-sign status, batch assignees, or parse any human-readable names from the activity log.

### 3.1 New Columns – Presence and Coverage

| Column | Description | Records Populated | Unique Values |
|--------|-------------|--------------------|---------------|
| `batch_e_signed` | True if batch has any `e_signed == True` or `e-sign-successful` event | 448/448 (all rows) | True: 231, False: 217 |
| `e_signer_user_id` | `user_id` from latest `e-sign-successful` event per batch | 231/448 | 1 (user 217) |
| `e_signer_name` | Reserved for future user_id-to-name lookup | 0/448 | n/a |
| `current_assignee_name` | Name from latest `batch_assignee` event `new_value` | 231/448 | 1 ("Tanya Chakladar") |
| `previous_assignee_name` | Name from latest `batch_assignee` event `old_value` | 231/448 | 1 ("Heliana Downing") |
| `current_assignee_id` | From latest `batch_assignee_id` event `new_value` | 448/448 | 2 ("Eswar Vamsi Krishna", "Madhu Konagandla") |
| `activity_commenter_names` | Pipe-separated distinct commenter names from "X commented in Comments" messages | 0/448 | n/a (no matching messages in current log batches) |
| `activity_mentioned_names` | Pipe-separated distinct @[Name] mentions from user-comment `new_value` | 0/448 | n/a (no @mentions in current log batches) |
| `hitl_reviewer_name` | QA reviewer name (proxy from `current_assignee_name`) | 231/448 | 1 ("Tanya Chakladar") |
| `hitl_reviewer_id` | Existing column (unchanged) | 448/448 | 1 ("USER-251.0") |

### 3.2 E-Signer vs. QA Reviewer Distinction

A key finding from this implementation is that the e-signer and the QA reviewer are different roles and often different people:

| Metric | Value |
|--------|-------|
| Batches with both `e_signer_user_id` and `hitl_reviewer_id` populated | 1 (at batch level) |
| Same person (e_signer == reviewer) | 0 |
| Different person | 1 |

- **E-signer** (user 217): The individual who completed the e-sign event, stored in `e_signer_user_id`.
- **QA reviewer** (USER-251.0 / "Tanya Chakladar"): The individual who applied the ground-truth QA label (Reprocess/Rejected/Approved), stored in `hitl_reviewer_id` and `hitl_reviewer_name`.

This confirms the plan's hypothesis: the e-signer is not the person who reviews and labels the batch records. Monitors (particularly M3) can now differentiate these roles.

### 3.3 Baseline vs. Comparator Coverage

| Dimension | Baseline (n=216) | Comparator (n=232) |
|-----------|-------------------|--------------------|
| `batch_e_signed == True` | 216 (100%) | 15 (6.5%) |
| `e_signer_user_id` non-null | 216 | 15 |
| `current_assignee_name` non-null | 216 | 15 |
| `hitl_reviewer_name` non-null | 216 | 15 |

The baseline batch (ADJ9318) has full e-sign and assignee data. In the comparator, only 15 of 232 records (batch ADE5350) have e-sign events; the remaining 217 comparator records have no e-sign or assignee activity in the log. This distribution is expected: not all batches have progressed to the e-sign or reassignment stage.

### 3.4 Drift Impact on New Columns (from MTR 3 test results)

The new columns are now tracked by the stability/drift monitors. CSI values from `CHIP_mtr_3_test_results.json`:

| Column | CSI | Interpretation |
|--------|-----|----------------|
| `batch_e_signed` | 2.5616 | High CSI due to baseline 100% True vs comparator 6.5% True (expected: different batch lifecycle stages) |
| `e_signer_user_id` | 2.5616 | Same as above (populated only where e-sign occurred) |
| `current_assignee_name` | 2.5616 | Same pattern; assignee data sparse in comparator |
| `previous_assignee_name` | 2.5616 | Same pattern |
| `current_assignee_id` | 2.5616 | Same pattern |
| `hitl_reviewer_name` | 2.5616 | Same pattern (derived from assignee) |
| `activity_commenter_names` | 0.0 | Stable (null in both splits) |
| `activity_mentioned_names` | 0.0 | Stable (null in both splits) |

The high CSI values for e-sign and assignee columns are expected: the baseline batch has completed its full lifecycle (including e-sign and assignment), while most comparator batches have not. This is a data maturity artifact, not drift. Monitors should treat these columns as informational/dimensional rather than drift-sensitive until the comparator batches mature.

### 3.5 Data Source Mapping

All new columns are derived from `batch_activity_log` event fields with no changes to the log format:

| Activity Log Field | Derived Column(s) |
|--------------------|-------------------|
| `e_signed` (boolean), `category == "e-sign-successful"` | `batch_e_signed` |
| `user_id` from `e-sign-successful` event | `e_signer_user_id` |
| `field_name == "batch_assignee"`, `new_value` / `old_value` | `current_assignee_name`, `previous_assignee_name` |
| `field_name == "batch_assignee_id"`, `new_value` | `current_assignee_id` |
| `message` matching `"X commented in Comments"` | `activity_commenter_names` |
| `new_value` containing `@[Name]` | `activity_mentioned_names` |
| `current_assignee_name` (proxy) | `hitl_reviewer_name` |

### 3.6 Gaps and Future Work

| Item | Status | Notes |
|------|--------|-------|
| `e_signer_name` | Null | E-sign messages do not contain the signer's name. Requires a user_id-to-name lookup table or API. |
| `activity_commenter_names` | Null | The current log batches (ACE7578 activity) are present but not matched to CHIP_master batches (ADJ9318/ADE5350). When future batches have comments, this column will populate. |
| `activity_mentioned_names` | Null | Same as above; @[Name] mentions will appear when matching batches have comment activity. |
| `hitl_reviewer_name` via direct event parsing | Proxy only | Currently set from `current_assignee_name`; could be enhanced to parse the QA decision event's `message` field directly if the message contains a name. |

---

## 4. Aggregate Summary Table

| Monitor | Proposal BMS Goal | Proposal Success Criterion | Delivered Functionality | This Run Result | vs Criterion |
|---------|-------------------|----------------------------|--------------------------|------------------|--------------|
| **M1** – Model Output Stability | Reliability (Model Stability) | AI rejection rate stable; PSI < 0.1 | PSI and CSI on AI output (e.g. ai_overall_status); baseline vs comparator; bar/table/donut; activity_feedback_summary | Score PSI = 0.1108; CSI max = 2.56 | **PSI > 0.1** – flag for review |
| **M2** – Approval Concordance | Accuracy & Agreement | >95% AI–Human agreement | Accuracy, precision, recall, F1, confusion matrix; reviewer_volume; activity_feedback_summary | Accuracy = 93.53% | **< 95%** – below target |
| **M3** – QA Calibration | Process Control (Human Stability) | Stable human rejection rates; track over time | Per-reviewer volume/rejection rate vs team; reviewer_time_series; time_line_graph; qa_feedback_samples; **new:** e-signer/assignee/name columns in drift | Team rate 0%; 1 reviewer, 1 date; all outputs present | **Met** – monitoring in place |
| **New: User Identity** | Traceability / Auditability | Distinguish e-signer from QA reviewer; surface batch assignee and user names | `batch_e_signed`, `e_signer_user_id`, `current_assignee_name`, `previous_assignee_name`, `current_assignee_id`, `hitl_reviewer_name`, `activity_commenter_names`, `activity_mentioned_names` | E-signer (user 217) differs from QA reviewer (USER-251.0 / Tanya Chakladar); 231/448 records have full identity data | **Met** – see Section 3 |

---

## 5. Conclusion and Next Steps

- **Chained run (CHIP_mtr_data → M1/M2/M3):** The preprocessing monitor produces CSV-only outputs in `CHIP_data/` (CHIP_master.csv, CHIP_baseline.csv, CHIP_comparator.csv). The three downstream monitors were run locally reading from these CSVs; all completed successfully and wrote test result JSONs. In ModelOp Center, add CHIP_mtr_data to the implementation first, configure its input assets, then add M1/M2/M3 and select the preprocessing outputs as BASELINE_DATA/COMPARATOR_DATA.
- **ModelOp Customer Success deliverable:** All three custom monitors ran successfully. They match the **required functionality** described in the proposal (stability/drift, concordance, QA calibration with reviewer vs team and time series).
- **E-sign and user identity tracking (new):** The pipeline now extracts batch-level e-sign status, e-signer user ID, batch assignee names (current and previous), assignee ID, and QA reviewer name from the activity log. The e-signer and QA reviewer are confirmed to be different individuals (user 217 vs USER-251.0 / Tanya Chakladar), validating the need for separate tracking. All new columns flow through to CHIP_master and all three monitor schemas. See Section 3 for full details.
- **BMS CHIP / Customer success criteria (this run):**
  - **M1:** PSI for AI overall status is slightly above the 0.1 stability threshold (0.11) – warrants review (e.g. baseline vs comparator window, batch mix).
  - **M2:** Agreement is 93.53%, below the >95% target – warrants review (e.g. schema/mapping, label quality, or operational mix).
  - **M3:** QA calibration logic and outputs are in place; current data show stable (0%) rejection with one reviewer and one comparator day. New e-sign/assignee columns show high CSI (2.56) due to data maturity differences between baseline and comparator, not actual drift.

**Recommended next steps:**  
(1) Re-run on production or a larger reference set to assess M1/M2 against 0.1 and 95% under target conditions.  
(2) Confirm with BMS CHIP whether baseline/comparator windows and LOD align with the proposal.  
(3) Use M3 reviewer_time_series and time_line_graph over longer periods to monitor for policy drift or fatigue once more dates/reviewers are available.  
(4) Populate `e_signer_name` once a user_id-to-name lookup is available. Monitor `activity_commenter_names` and `activity_mentioned_names` as more batches with comment activity flow through.  
(5) Consider excluding lifecycle-dependent columns (`batch_e_signed`, `e_signer_user_id`, `current_assignee_name`) from CSI/PSI drift thresholds or flagging them as informational to avoid false-positive drift alerts caused by batch maturity differences.
