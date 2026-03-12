"""
BMS CHIP Individual Source Data Discovery Script
------------------------------------------------
This script parses the Batch Activity Log, AI Feedback Log, AND Claude AI Responses.
It flattens nested JSON attributes and generates a comprehensive PDF report
detailing the categorical distributions and nested relationships (trees) 
within each source to help identify potential Ground Truth labels and AI behavior.

Requirements:
    pip install pandas reportlab
"""

import json
import os
import glob
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DATA_DIR = os.path.join(_REPO_ROOT, "CHIP_mtr_data")

def safe_json_loads(x):
    """Safely parse JSON strings, returning an empty dict on failure."""
    if pd.isna(x) or str(x).strip() == "":
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return {}

def load_and_flatten_data():
    """Loads the JSON DB extracts and flattens nested structures."""
    activity_file = os.path.join(_DATA_DIR, 'batch_activity_log_202603042226.json')
    feedback_file = os.path.join(_DATA_DIR, 'ai_feedback_202603042225.json')
    
    # 1. Load Activity Log
    try:
        with open(activity_file, 'r') as f:
            df_act = pd.DataFrame(json.load(f).get('batch_activity_log', []))
    except FileNotFoundError:
        print(f"[!] Warning: {activity_file} not found.")
        df_act = pd.DataFrame()

    # 2. Load Feedback Log
    try:
        with open(feedback_file, 'r') as f:
            df_fb = pd.DataFrame(json.load(f).get('ai_feedback', []))
            
        # Flatten the nested 'context' JSON string if it exists
        if not df_fb.empty and 'context' in df_fb.columns:
            parsed_context = df_fb['context'].apply(safe_json_loads)
            df_context = pd.json_normalize(parsed_context) #type: ignore
            # Prefix columns to avoid collisions
            df_context.columns = [f"context.{c}" for c in df_context.columns]
            df_fb = pd.concat([df_fb.drop('context', axis=1), df_context], axis=1)
    except FileNotFoundError:
        print(f"[!] Warning: {feedback_file} not found.")
        df_fb = pd.DataFrame()

    return df_act, df_fb

def load_claude_data(directory=os.path.join(_DATA_DIR, "AI Responses")):
    """Scans the AI Responses directory to profile raw Claude outputs."""
    print(f"Loading Claude JSONs from '{directory}'...")
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    batch_records = []
    row_records = []
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
            
        # Schema 1: BG
        if "headerData" in data and "rows" in data:
            schema = "BG (Background Data)"
            raw_val_status = str(data.get("summary", {}).get("validation_status", "Missing"))
            
            batch_records.append({
                "File": filename,
                "Schema": schema,
                "Raw_Validation_Status": raw_val_status
            })
            
            for row in data.get("rows", []):
                r_data = row.get("data", {})
                row_records.append({
                    "File": filename,
                    "Schema": schema,
                    "Row_Status": str(r_data.get("overall_batch_result", "Missing"))
                })
                
        # Schema 2: CCA
        elif "qeList" in data:
            schema = "CCA (Quality Events)"
            raw_val_status = str(data.get("overallBatchResult", "Missing"))
            
            batch_records.append({
                "File": filename,
                "Schema": schema,
                "Raw_Validation_Status": raw_val_status
            })
            
            for qe in data.get("qeList", []):
                row_records.append({
                    "File": filename,
                    "Schema": schema,
                    "Row_Status": str(qe.get("submissionStatus", "Missing"))
                })
        else:
            batch_records.append({
                "File": filename,
                "Schema": "UNKNOWN/OTHER",
                "Raw_Validation_Status": "N/A"
            })

    return pd.DataFrame(batch_records), pd.DataFrame(row_records)

def get_categorical_distribution(df, column_name):
    if df.empty or column_name not in df.columns:
        return []
    
    counts = df[column_name].fillna("NULL").value_counts()
    total = len(df)
    dist = []
    for val, count in counts.items():
        pct = f"{(count / total) * 100:.1f}%"
        dist.append([str(val), str(count), pct])
    return dist

def create_distribution_table(headers, data, styles):
    table_data = [headers] + data
    t = Table(table_data, colWidths=[250, 100, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4053')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F4F6F7')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
    ]))
    return t

def build_nested_tree_table(df, level1, level2, level3, styles):
    if df.empty or not all(col in df.columns for col in [level1, level2, level3]):
        return Paragraph("<i>Missing columns for tree generation</i>", styles['Normal'])

    grouped = df.fillna("NULL").groupby([level1, level2, level3]).size().reset_index(name='Count')
    grouped = grouped.sort_values([level1, 'Count'], ascending=[True, False])
    
    tree_data = [[level1.split('.')[-1].upper(), level2.split('.')[-1].upper(), level3.split('.')[-1].upper(), "Count"]]
    for _, row in grouped.iterrows():
        tree_data.append([str(row[level1]), str(row[level2]), str(row[level3]), str(row['Count'])])

    t = Table(tree_data, colWidths=[120, 150, 120, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874A6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (3, 0), (3, -1), 'CENTER')
    ]))
    return t

def generate_pdf_report(df_act, df_fb, df_claude_batch, df_claude_row, pdf_filename="BMS_Data_Discovery_Report.pdf"):
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # TITLE
    elements.append(Paragraph("BMS CHIP Data Discovery & Profiling Report", styles['Title']))
    elements.append(Paragraph("Automated extraction of Categories, UI Contexts, and AI Outputs", styles['Normal']))
    elements.append(Spacer(1, 20))

    # ==========================================
    # SECTION 1: Batch Activity Log
    # ==========================================
    elements.append(Paragraph("1. Batch Activity Log Analysis", styles['Heading2']))
    elements.append(Paragraph(f"Total Records: {len(df_act)}", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("1A. Distribution of 'category'", styles['Heading3']))
    cat_dist = get_categorical_distribution(df_act, 'category')
    if cat_dist:
        elements.append(create_distribution_table(['Category', 'Count', '% of Total'], cat_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("1B. Distribution of 'field_name'", styles['Heading3']))
    field_dist = get_categorical_distribution(df_act, 'field_name')
    if field_dist:
        elements.append(create_distribution_table(['Field Name', 'Count', '% of Total'], field_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(PageBreak())

    # ==========================================
    # SECTION 2: AI Feedback Log
    # ==========================================
    elements.append(Paragraph("2. AI Feedback Log Analysis", styles['Heading2']))
    elements.append(Paragraph(f"Total Records: {len(df_fb)}", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("2A. Distribution of 'feedback_type'", styles['Heading3']))
    fb_dist = get_categorical_distribution(df_fb, 'feedback_type')
    if fb_dist:
        elements.append(create_distribution_table(['Feedback Type', 'Count', '% of Total'], fb_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("2B. Distribution of 'context.page_type'", styles['Heading3']))
    page_dist = get_categorical_distribution(df_fb, 'context.page_type')
    if page_dist:
        elements.append(create_distribution_table(['Page Type', 'Count', '% of Total'], page_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("2C. Nested Tree: Page Type -> UI Tab -> Action", styles['Heading3']))
    elements.append(Paragraph("Shows where human intervention occurs within the application architecture.", styles['Normal']))
    elements.append(Spacer(1, 5))
    tree_fb = build_nested_tree_table(df_fb, 'context.page_type', 'context.page_info.tab', 'action', styles)
    elements.append(tree_fb)
    elements.append(PageBreak())

    # ==========================================
    # SECTION 3: Claude AI Responses
    # ==========================================
    elements.append(Paragraph("3. Claude AI Response Profile (RAW STRINGS)", styles['Heading2']))
    elements.append(Paragraph("Analyzes the exact string values returned by Claude before PASS/FAIL mapping.", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("3A. Discovered Document Schemas", styles['Heading3']))
    schema_dist = get_categorical_distribution(df_claude_batch, 'Schema')
    if schema_dist:
        elements.append(create_distribution_table(['Detected Schema', 'Files', '% of Total'], schema_dist, styles))
    else:
        elements.append(Paragraph("<i>No AI files found.</i>", styles['Normal']))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("3B. Raw 'Validation Status' Strings (Batch Level)", styles['Heading3']))
    elements.append(Paragraph("This is the raw string returned by the AI (e.g., 'partially_valid', 'false').", styles['Normal']))
    elements.append(Spacer(1, 5))
    val_status_dist = get_categorical_distribution(df_claude_batch, 'Raw_Validation_Status')
    if val_status_dist:
        elements.append(create_distribution_table(['Raw String Output', 'Count', '% of Total'], val_status_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("3C. Raw 'Row Status' Strings (Test Level)", styles['Heading3']))
    elements.append(Paragraph("Displays how AI grades individual tests (e.g. 'overall_batch_result' or 'submissionStatus').", styles['Normal']))
    elements.append(Spacer(1, 5))
    row_status_dist = get_categorical_distribution(df_claude_row, 'Row_Status')
    if row_status_dist:
        elements.append(create_distribution_table(['Raw Row Status', 'Count', '% of Total'], row_status_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))

    # Build PDF
    doc.build(elements)
    print(f"\n[*] Successfully generated PDF report: {pdf_filename}")

if __name__ == "__main__":
    print("--- Starting Discovery Script ---")
    
    # Load Logs
    df_act, df_fb = load_and_flatten_data()
    
    # Load AI Responses
    df_claude_batch, df_claude_row = load_claude_data()
    
    # Generate PDF
    generate_pdf_report(df_act, df_fb, df_claude_batch, df_claude_row)