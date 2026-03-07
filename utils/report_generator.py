import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_report(task_id: str, analysis_data: dict) -> str:
    """
    Vygeneruje PDF report z posbíraných dat.
    Vrací url cestu k hotovému reportu.
    """
    output_filename = f"data/reports/{task_id}.pdf"
    
    doc = SimpleDocTemplate(output_filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = styles['Heading1']
    title_style.textColor = colors.HexColor('#3b82f6')
    
    normal_style = styles['Normal']
    
    elements = []
    
    # Hlavička
    elements.append(Paragraph(f"OSINT Session Report", title_style))
    elements.append(Paragraph(f"Task ID: {task_id}", normal_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Syntéza (Verdict)
    synth = analysis_data.get("synthesis", {})
    is_authentic = synth.get("is_authentic", False)
    score = synth.get("authenticity_score", 0.0)
    
    verdict_text = "AUTHENTIC" if is_authentic else "MANIPULATED / UNRELIABLE"
    verdict_color = colors.green if is_authentic else colors.red
    
    v_style = ParagraphStyle('Verdict', parent=styles['Heading2'], textColor=verdict_color)
    elements.append(Paragraph(f"FINAL VERDICT: {verdict_text} (Score: {score*100:.0f}%)", v_style))
    elements.append(Spacer(1, 10))
    
    # Poznámky syntézy
    notes = synth.get("reliability_notes", [])
    if notes:
        elements.append(Paragraph("System Notes:", styles['Heading3']))
        for n in notes:
            elements.append(Paragraph(f"• {n}", normal_style))
        elements.append(Spacer(1, 15))
        
    # Forensics Data
    forensic = analysis_data.get("forensic", {})
    if forensic:
        elements.append(Paragraph("Forensic Data", styles['Heading2']))
        meta = forensic.get("metadata", {})
        data = [
            ["Attribute", "Value"],
            ["DateTime Original", meta.get("datetime_original", "N/A")],
            ["Camera Make/Model", f'{meta.get("make", "N/A")} / {meta.get("model", "N/A")}']
        ]
        t = Table(data, colWidths=[150, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

    # Environment
    env = analysis_data.get("environment", {})
    if env:
        elements.append(Paragraph("Environmental Analysis", styles['Heading2']))
        wea = env.get("sky_weather_match", {})
        elements.append(Paragraph(f"Weather Match Confidence: {wea.get('confidence', 0)*100:.0f}%", normal_style))
        elements.append(Paragraph(f"Details: {wea.get('details', 'N/A')}", normal_style))
        elements.append(Spacer(1, 20))

    # Entities
    ent = analysis_data.get("entities", {})
    if ent:
        elements.append(Paragraph("Identified Entities", styles['Heading2']))
        persons = ent.get("persons", [])
        for p in persons:
            elements.append(Paragraph(f"Person: {p.get('description')} ({p.get('confidence', 0)*100:.0f}%)", normal_style))
        objs = ent.get("objects", [])
        if objs:
            elements.append(Paragraph(f"Objects: {', '.join(objs)}", normal_style))
            
    # Sestavení PDF
    doc.build(elements)
    
    return f"/download-report/{task_id}"
