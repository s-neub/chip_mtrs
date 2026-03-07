"""
BMS CHIP Individual Source Data Discovery Script
------------------------------------------------
This script parses the Batch Activity Log and AI Feedback Log independently.
It flattens nested JSON attributes and generates a comprehensive PDF report
detailing the categorical distributions and nested relationships (trees) 
within each source to help identify potential Ground Truth labels.

Requirements:
    pip install pandas reportlab
"""

import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def safe_json_loads(x):
    """Safely parse JSON strings, returning an empty dict on failure."""
    if pd.isna(x) or str(x).strip() == "":
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return {}

def load_and_flatten_data():
    """Loads the JSON extracts and flattens nested structures for independent analysis."""
    activity_file = 'batch_activity_log_202603042226.json'
    feedback_file = 'ai_feedback_202603042225.json'
    
    print(f"Loading {activity_file}...")
    with open(activity_file, 'r') as f:
        activity_data = json.load(f)['batch_activity_log']
    df_activity = pd.DataFrame(activity_data)
    
    print(f"Loading and flattening {feedback_file}...")
    with open(feedback_file, 'r') as f:
        feedback_data = json.load(f)['ai_feedback']
    df_feedback = pd.DataFrame(feedback_data)
    
    # Flatten Feedback Context
    parsed_context = df_feedback['context'].apply(safe_json_loads)
    df_flat_context = pd.json_normalize(parsed_context) #type:ignore
    df_flat_context.columns = [f"context.{col}" for col in df_flat_context.columns]
    df_feedback = pd.concat([df_feedback.drop(columns=['context']), df_flat_context], axis=1)
    
    return df_activity, df_feedback

def get_categorical_distribution(df, column_name):
    """Returns a list of lists containing value, count, and percentage for ALL distinct values."""
    if column_name not in df.columns:
        return []
    
    counts = df[column_name].value_counts(dropna=False)
    total = len(df)
    
    data = []
    for val, count in counts.items():
        pct = (count / total) * 100
        val_str = str(val) if not pd.isna(val) else "NULL"
        data.append([val_str, str(count), f"{pct:.1f}%"])
        
    return data

def build_nested_tree_table(df, parent_col, child_col1, child_col2, styles):
    """Builds a nested view (Parent -> All Children) to show categorical relationships."""
    tbl_body_style = ParagraphStyle(name='TblBody', parent=styles['Normal'], fontSize=8, leading=10)
    tbl_header_style = ParagraphStyle(name='TblHead', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', textColor=colors.whitesmoke)
    
    table_data = [[
        Paragraph(f"Parent: {parent_col}", tbl_header_style), 
        Paragraph(f"Child: {child_col1}", tbl_header_style), 
        Paragraph(f"Child: {child_col2}", tbl_header_style)
    ]]
    
    # Group by parent to find relationships (ALL parents)
    all_parents = df[parent_col].value_counts().index
    
    for parent in all_parents:
        parent_df = df[df[parent_col] == parent]
        
        # Get all child 1
        all_child1 = parent_df[child_col1].value_counts().index.tolist()
        
        # Get all child 2
        all_child2 = parent_df[child_col2].value_counts().index.tolist()
        
        # Chunking lists into groups of 5 to prevent LayoutError from cells overflowing a PDF page
        chunk_size = 5
        child1_chunks = [all_child1[i:i + chunk_size] for i in range(0, len(all_child1), chunk_size)]
        child2_chunks = [all_child2[i:i + chunk_size] for i in range(0, len(all_child2), chunk_size)]
        
        max_chunks = max(len(child1_chunks), len(child2_chunks), 1)
        
        # Iterate and create a new row for each chunk so ReportLab can paginate properly
        for i in range(max_chunks):
            # Only print the parent name on the first row of the chunk set
            parent_str = str(parent) if i == 0 else ""
            
            c1_str = ", ".join([str(x) for x in child1_chunks[i] if pd.notna(x)]) if i < len(child1_chunks) else ""
            if not c1_str and i == 0 and not all_child1:
                c1_str = "N/A"
                
            c2_str = ", ".join([str(x) for x in child2_chunks[i] if pd.notna(x)]) if i < len(child2_chunks) else ""
            if not c2_str and i == 0 and not all_child2:
                c2_str = "N/A"
            
            table_data.append([
                Paragraph(parent_str, tbl_body_style),
                Paragraph(c1_str, tbl_body_style),
                Paragraph(c2_str, tbl_body_style)
            ])
            
    # repeatRows=1 ensures headers print on new pages if split
    t = Table(table_data, colWidths=[150, 150, 200], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#006630')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ]))
    return t

def create_distribution_table(headers, data, styles):
    """Helper to create a standard distribution table."""
    tbl_header_style = ParagraphStyle(name='TblHead', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', textColor=colors.whitesmoke)
    tbl_body_style = ParagraphStyle(name='TblBody', parent=styles['Normal'], fontSize=8, leading=10)
    
    formatted_data = [[Paragraph(h, tbl_header_style) for h in headers]]
    for row in data:
        formatted_data.append([Paragraph(str(cell), tbl_body_style) for cell in row])
        
    # repeatRows=1 ensures headers print on new pages if split
    t = Table(formatted_data, colWidths=[280, 100, 100], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#006630')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
    ]))
    return t

def generate_pdf_report(df_act, df_fb):
    """Generates the comprehensive PDF Report."""
    pdf_filename = "BMS_CHIP_Source_Categorical_Discovery.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # --- TITLE ---
    elements.append(Paragraph("BMS CHIP: Independent Source Categorical Discovery", styles['Title']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("This report isolates and maps the categorical structures of the two distinct data extracts to identify potential alternate definitions for 'Ground Truth' and 'Model Failure'.", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # ==========================================
    # PART 1: BATCH ACTIVITY LOG
    # ==========================================
    elements.append(Paragraph("Part 1: Batch Activity Log Analysis", styles['Heading1']))
    elements.append(Paragraph(f"Total Records: {len(df_act)}", styles['Normal']))
    elements.append(Spacer(1, 10))
    
    # Distribution: Category
    elements.append(Paragraph("1A. Distribution of 'category'", styles['Heading3']))
    elements.append(Paragraph("The broad classification of user and system events.", styles['Normal']))
    cat_dist = get_categorical_distribution(df_act, 'category')
    elements.append(create_distribution_table(['Category', 'Count', '% of Total'], cat_dist, styles))
    elements.append(Spacer(1, 15))
    
    # Distribution: Field Name
    elements.append(Paragraph("1B. Distribution of 'field_name'", styles['Heading3']))
    elements.append(Paragraph("The specific database attributes being created or modified.", styles['Normal']))
    field_dist = get_categorical_distribution(df_act, 'field_name')
    elements.append(create_distribution_table(['Field Name', 'Count', '% of Total'], field_dist, styles))
    elements.append(Spacer(1, 15))
    
    # Nested Tree: Activity Log
    elements.append(Paragraph("1C. Nested Tree: Category -> Field Name -> New Value", styles['Heading3']))
    elements.append(Paragraph("This shows how broad categories map to specific fields and what values those fields take. This is critical for finding alternate Ground Truth indicators.", styles['Normal']))
    elements.append(Spacer(1, 5))
    tree_act = build_nested_tree_table(df_act, 'category', 'field_name', 'new_value', styles)
    elements.append(tree_act)
    
    elements.append(PageBreak())
    
    # ==========================================
    # PART 2: AI FEEDBACK LOG
    # ==========================================
    elements.append(Paragraph("Part 2: AI Feedback Log Analysis (Flattened)", styles['Heading1']))
    elements.append(Paragraph(f"Total Records: {len(df_fb)}", styles['Normal']))
    elements.append(Spacer(1, 10))
    
    # Distribution: Feedback Type
    elements.append(Paragraph("2A. Distribution of 'feedback_type'", styles['Heading3']))
    fb_type_dist = get_categorical_distribution(df_fb, 'feedback_type')
    if fb_type_dist:
        elements.append(create_distribution_table(['Feedback Type', 'Count', '% of Total'], fb_type_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(Spacer(1, 15))
    
    # Distribution: Page Type
    elements.append(Paragraph("2B. Distribution of 'context.page_type'", styles['Heading3']))
    page_dist = get_categorical_distribution(df_fb, 'context.page_type')
    if page_dist:
        elements.append(create_distribution_table(['Page Type', 'Count', '% of Total'], page_dist, styles))
    else:
        elements.append(Paragraph("<i>No data</i>", styles['Normal']))
    elements.append(Spacer(1, 15))
    
    # Nested Tree: Feedback Log
    elements.append(Paragraph("2C. Nested Tree: Page Type -> UI Tab -> Action", styles['Heading3']))
    elements.append(Paragraph("Shows where human intervention occurs within the application architecture.", styles['Normal']))
    elements.append(Spacer(1, 5))
    if not df_fb.empty and 'context.page_type' in df_fb.columns:
        tree_fb = build_nested_tree_table(df_fb, 'context.page_type', 'context.page_info.tab', 'action', styles)
        elements.append(tree_fb)
    else:
        elements.append(Paragraph("<i>Insufficient nested data to build tree.</i>", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    print(f"Successfully generated PDF report: {pdf_filename}")

if __name__ == "__main__":
    try:
        # Load and Flatten
        df_act, df_fb = load_and_flatten_data()
        
        # Generate Visualization PDF
        generate_pdf_report(df_act, df_fb)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the JSON extracts are in the same directory.")