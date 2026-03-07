"""
BMS CHIP Advanced Data Discovery Script
----------------------------------------
This script flattens hybrid JSON/CSV data into a pure CSV format using dot notation.
It merges the Batch Activity Log and AI Feedback Log, outputting a bidirectional CSV.
Finally, it generates a comprehensive Data Discovery PDF report with charts to aid 
in baseline generation for ModelOp Monitors 1, 2, and 3.

Requirements:
    pip install pandas reportlab
"""

import json
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend

def safe_json_loads(x):
    """Safely parse JSON strings, returning an empty dict on failure or null."""
    if pd.isna(x) or str(x).strip() == "":
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return {}

def load_and_flatten_data():
    """Loads the JSON extracts and flattens nested structures to pure tabular columns."""
    activity_file = 'batch_activity_log_202603042226.json'
    feedback_file = 'ai_feedback_202603042225.json'
    
    print(f"Loading and flattening {activity_file}...")
    with open(activity_file, 'r') as f:
        activity_data = json.load(f)['batch_activity_log']
    df_activity = pd.DataFrame(activity_data)
    
    print(f"Loading and flattening {feedback_file}...")
    with open(feedback_file, 'r') as f:
        feedback_data = json.load(f)['ai_feedback']
    df_feedback = pd.DataFrame(feedback_data)
    
    # ---------------------------------------------------------
    # FLATTENING LOGIC: Convert nested JSON to dot-notation CSV
    # ---------------------------------------------------------
    # Safely convert the stringified JSON in 'context' to actual dictionaries
    parsed_context = df_feedback['context'].apply(safe_json_loads)
    
    # Normalize the dictionaries into a flat dataframe with dot notation
    df_flat_context = pd.json_normalize(parsed_context) #type:ignore
    
    # Prepend 'context.' to clarify lineage and avoid column collisions
    df_flat_context.columns = [f"context.{col}" for col in df_flat_context.columns]
    
    # Drop the original nested column and concatenate the flattened columns
    df_feedback = pd.concat([df_feedback.drop(columns=['context']), df_flat_context], axis=1)
    
    return df_activity, df_feedback

def process_and_join(df_activity, df_feedback):
    """Normalizes keys, converts dates, and joins the datasets."""
    # Standardize the join key
    df_feedback = df_feedback.rename(columns={'batch_id': 'batch_number'})
    
    # Convert timestamps
    df_activity['timestamp'] = pd.to_datetime(df_activity['timestamp'])
    df_feedback['created_at'] = pd.to_datetime(df_feedback['created_at'])
    
    # Create the unified, pure CSV Outer Join (Historical Master Record)
    df_merged = pd.merge(
        df_activity, 
        df_feedback, 
        on='batch_number', 
        how='outer', 
        suffixes=('_activity', '_feedback')
    )
    
    # Save the pure CSV format dataset
    merged_filename = 'bms_chip_merged_pure_historical.csv'
    df_merged.to_csv(merged_filename, index=False)
    print(f"Successfully saved pure flattened CSV to {merged_filename}")
    
    return df_activity, df_feedback, df_merged

# --- REPORTLAB CHART GENERATORS ---

def create_bar_chart(data_series, title):
    """Generates a ReportLab Bar Chart from a pandas value_counts Series."""
    d = Drawing(460, 240)
    chart = VerticalBarChart()
    chart.width = 380
    chart.height = 130
    chart.x = 40
    chart.y = 70  # Shifted up to give labels room to wrap/angle below
    
    # Format data for reportlab
    labels = [str(x)[:20] + ".." if len(str(x)) > 20 else str(x) for x in data_series.index.tolist()]
    values = data_series.values.tolist()
    
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.bars[0].fillColor = colors.HexColor('#006630') # BMS Green
    chart.valueAxis.valueMin = 0
    
    # Angle text to prevent overlap in portrait view
    chart.categoryAxis.labels.angle = 35
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.dy = -8
    chart.categoryAxis.labels.dx = -5
    
    d.add(String(230, 220, title, fontSize=12, fontName="Helvetica-Bold", textAnchor="middle"))
    d.add(chart)
    return d

def create_pie_chart(data_series, title):
    """Generates a ReportLab Pie Chart from a pandas value_counts Series."""
    d = Drawing(460, 200)
    pie = Pie()
    pie.x = 40
    pie.y = 20
    pie.width = 140
    pie.height = 140
    
    # Truncate labels to prevent legend overflow on portrait orientation
    labels = [str(x)[:25] + ".." if len(str(x)) > 25 else str(x) for x in data_series.index.tolist()]
    values = data_series.values.tolist()
    
    pie.data = values
    pie.labels = labels
    
    # Custom color palette for slices
    color_palette = [
        colors.HexColor('#006630'), colors.HexColor('#008844'),
        colors.HexColor('#33aa66'), colors.HexColor('#66cc88'),
        colors.HexColor('#99eeaa'), colors.HexColor('#cceecc')
    ]
    for i in range(len(pie.data)):
        pie.slices[i].fillColor = color_palette[i % len(color_palette)]
        
    legend = Legend()
    legend.x = 220 # Shifted inward for Portrait constraints
    legend.y = 150
    legend.dy = 10
    legend.fontName = 'Helvetica'
    legend.fontSize = 9
    legend.colorNamePairs = [(pie.slices[i].fillColor, labels[i]) for i in range(len(labels))]
    
    d.add(String(230, 180, title, fontSize=12, fontName="Helvetica-Bold", textAnchor="middle"))
    d.add(pie)
    d.add(legend)
    return d

def generate_pdf_report(df_activity, df_feedback, df_merged):
    """Generates a rich PDF data discovery report with visual distributions."""
    pdf_filename = "BMS_CHIP_Monitors_Discovery_Report.pdf"
    
    # Changed to Portrait layout
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter) 
    styles = getSampleStyleSheet()
    elements = []
    
    # Custom Text Wrapping Styles for Tables
    tbl_header_style = ParagraphStyle(name='TblHeader', parent=styles['Normal'], fontName='Helvetica-Bold', textColor=colors.whitesmoke, fontSize=10)
    tbl_body_style = ParagraphStyle(name='TblBody', parent=styles['Normal'], fontSize=9, leading=11)
    
    # TITLE
    elements.append(Paragraph("BMS CHIP: Data Discovery & Monitor Readiness", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # --- 1. OVERVIEW ---
    elements.append(Paragraph("1. Data Integration Overview", styles['Heading2']))
    exec_summary = (
        f"The data extraction and normalization process is complete. The hybrid JSON context was successfully "
        f"flattened using pure dot notation (e.g., <i>context.page_info.tab</i>). The result is a bidirectional, "
        f"machine-readable CSV containing <b>{len(df_merged)}</b> total events."
    )
    elements.append(Paragraph(exec_summary, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # --- 2. GROUND TRUTH LABELS (MONITOR 2) ---
    elements.append(Paragraph("2. Ground Truth & Batch Status Analysis (Monitor 2)", styles['Heading2']))
    gt_desc = (
        "To evaluate Approval Concordance, we must isolate the discrete 'Ground Truth' states generated by the "
        "QA process. The distribution below shows the discrete categories available in the `new_value` field "
        "when `field_name` == 'batch_status'."
    )
    elements.append(Paragraph(gt_desc, styles['Normal']))
    
    # Isolate batch status changes
    df_status = df_activity[df_activity['field_name'] == 'batch_status']
    if not df_status.empty:
        status_counts = df_status['new_value'].value_counts().head(6)
        elements.append(create_pie_chart(status_counts, "Discrete Ground Truth Labels (Batch Status)"))
    else:
        elements.append(Paragraph("<i>No 'batch_status' data found in activity log.</i>", styles['Normal']))
    
    elements.append(Spacer(1, 20))
    
    # --- 3. HITL BASELINE CALIBRATION (MONITOR 3) ---
    elements.append(Paragraph("3. HITL Activity Baseline (Monitor 3)", styles['Heading2']))
    hitl_desc = (
        "Monitor 3 tracks Human 'Policy Drift' over time. Establishing a 'Business As Usual' (BAU) baseline "
        "requires understanding the volume and types of human interventions. Below is the volume of activity categories."
    )
    elements.append(Paragraph(hitl_desc, styles['Normal']))
    
    if not df_activity.empty:
        cat_counts = df_activity['category'].value_counts().head(8)
        elements.append(create_bar_chart(cat_counts, "Top Activity Categories (BAU Tracking)"))
    
    elements.append(Spacer(1, 20))
    
    # AI Feedback Types
    if not df_feedback.empty:
        fb_counts = df_feedback['feedback_type'].value_counts().head(5)
        elements.append(create_pie_chart(fb_counts, "AI Feedback Dist (Model Output Adjustments)"))
    
    elements.append(PageBreak())
    
    # --- APPENDIX: DATA DICTIONARY ---
    elements.append(Paragraph("Appendix: Derived Data Dictionary", styles['Heading1']))
    dict_desc = "The following defines the pure tabular schema generated by flattening the hybrid JSON/CSV extracts."
    elements.append(Paragraph(dict_desc, styles['Normal']))
    elements.append(Spacer(1, 10))
    
    # Table data for dictionary - Wrapped in Paragraphs to prevent overflow
    dict_data = [
        [Paragraph("Column Name (Pure CSV)", tbl_header_style), Paragraph("Source", tbl_header_style), Paragraph("Description / Definition", tbl_header_style)],
        [Paragraph("id_activity", tbl_body_style), Paragraph("Activity Log", tbl_body_style), Paragraph("Unique identifier for the audit trail event.", tbl_body_style)],
        [Paragraph("batch_number", tbl_body_style), Paragraph("Join Key", tbl_body_style), Paragraph("The core document identifier connecting Camunda and AI.", tbl_body_style)],
        [Paragraph("action_type", tbl_body_style), Paragraph("Activity Log", tbl_body_style), Paragraph("The type of state change (e.g., CREATE, MODIFY).", tbl_body_style)],
        [Paragraph("old_value / new_value", tbl_body_style), Paragraph("Activity Log", tbl_body_style), Paragraph("State transition values. 'new_value' holds the ground truth label.", tbl_body_style)],
        [Paragraph("category", tbl_body_style), Paragraph("Activity Log", tbl_body_style), Paragraph("The classification of the event (e.g., user-comment, batch-status-changed).", tbl_body_style)],
        [Paragraph("field_name", tbl_body_style), Paragraph("Activity Log", tbl_body_style), Paragraph("The database field that was altered.", tbl_body_style)],
        [Paragraph("feedback_type", tbl_body_style), Paragraph("Feedback Log", tbl_body_style), Paragraph("The explicit classification of the QA intervention (e.g., ai-correction).", tbl_body_style)],
        [Paragraph("context.row_id", tbl_body_style), Paragraph("Feedback Log", tbl_body_style), Paragraph("(Flattened) The specific row index the QA corrected.", tbl_body_style)],
        [Paragraph("context.page_info.tab", tbl_body_style), Paragraph("Feedback Log", tbl_body_style), Paragraph("(Flattened) The UI tab where the feedback was submitted.", tbl_body_style)],
        [Paragraph("context.page_type", tbl_body_style), Paragraph("Feedback Log", tbl_body_style), Paragraph("(Flattened) The overarching app module (e.g., CDP, QPRP).", tbl_body_style)],
        [Paragraph("context.section_title", tbl_body_style), Paragraph("Feedback Log", tbl_body_style), Paragraph("(Flattened) The document section targeted (e.g., 'Product').", tbl_body_style)]
    ]
    
    # Style and build the Data Dictionary table - Constrained to 460 width for Portrait
    dict_table = Table(dict_data, colWidths=[130, 80, 250])
    dict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#006630')),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), # Top alignment is better for wrapped text
    ]))
    
    elements.append(dict_table)
    
    # Build PDF
    doc.build(elements)
    print(f"Successfully generated PDF report: {pdf_filename}")

if __name__ == "__main__":
    try:
        # Load and Flatten
        df_act, df_fb = load_and_flatten_data()
        
        # Process and Join
        df_act, df_fb, df_merged = process_and_join(df_act, df_fb)
        
        # Generate Visualization PDF
        generate_pdf_report(df_act, df_fb, df_merged)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the JSON extracts are in the same directory.")