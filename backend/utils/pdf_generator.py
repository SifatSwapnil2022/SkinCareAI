from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io, base64
from PIL import Image as PILImage


def PDF_generator_report(
    user_name:       str,
    user_email:      str,
    disease:         str,
    confidence:      float,
    all_predictions: dict,
    recommendations: str,
    next_steps:      str,
    tips:            str,
    model_used:      str,
    image_base64:    str,
    analysis_date:   str
) -> bytes:

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               topMargin=0.5*inch, bottomMargin=0.5*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 fontSize=22, textColor=colors.HexColor("#FF6B6B"),
                                 spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("Sub", parent=styles["Normal"],
                                 fontSize=10, textColor=colors.HexColor("#636e72"),
                                 alignment=TA_CENTER, spaceAfter=12)
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"],
                                 fontSize=13, textColor=colors.HexColor("#4ECDC4"),
                                 spaceBefore=12, spaceAfter=6)
    body_style  = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=16,
                                 textColor=colors.HexColor("#2D3436"))

    # ── Header 
    story.append(Paragraph("🏥 Skin Disease Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {analysis_date}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#FF6B6B")))
    story.append(Spacer(1, 12))

    # ── Patient Info 
    story.append(Paragraph("Patient Information", h2_style))
    patient_data = [
        ["Name",          user_name],
        ["Email",         user_email],
        ["Model Used",    model_used],
        ["Analysis Date", analysis_date],
    ]
    t = Table(patient_data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (0, -1), colors.HexColor("#4ECDC4")),
        ("TEXTCOLOR",      (0, 0), (0, -1), colors.white),
        ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1),
         [colors.HexColor("#F7F9FC"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe6e9")),
        ("PADDING",        (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Uploaded Image 
    try:
        img_data = base64.b64decode(image_base64)
        pil_img  = PILImage.open(io.BytesIO(img_data)).convert("RGB")
        img_buf  = io.BytesIO()
        pil_img.save(img_buf, format="JPEG")
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=2.5*inch, height=2.5*inch)
        story.append(Paragraph("Uploaded Image", h2_style))
        story.append(rl_img)
        story.append(Spacer(1, 12))
    except:
        pass

    # ── Diagnosis 
    story.append(Paragraph("Diagnosis Result", h2_style))
    diag_data = [
        ["Detected Condition", disease],
        ["Confidence",         f"{confidence * 100:.1f}%"],
    ]
    t2 = Table(diag_data, colWidths=[2*inch, 4*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FF6B6B")),
        ("TEXTCOLOR",  (0, 0), (0, -1), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 11),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe6e9")),
        ("PADDING",    (0, 0), (-1, -1), 10),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # ── All Predictions 
    story.append(Paragraph("All Class Probabilities", h2_style))
    sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    pred_data    = [["Disease", "Confidence"]] + \
                   [[k, f"{v*100:.2f}%"] for k, v in sorted_preds]
    t3 = Table(pred_data, colWidths=[4*inch, 2*inch])
    t3.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2D3436")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F7F9FC"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe6e9")),
        ("PADDING",        (0, 0), (-1, -1), 8),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))

    # ── LLM Sections 
    for title, content in [
        ("Recommendations", recommendations),
        ("Next Steps",      next_steps),
        ("Daily Tips",      tips),
    ]:
        if content and content.strip():
            story.append(Paragraph(title, h2_style))
            story.append(Paragraph(content.replace("\n", "<​br/>"), body_style))
            story.append(Spacer(1, 8))

    # ── Disclaimer 
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#dfe6e9")))
    disclaimer_style = ParagraphStyle("Disc", parent=styles["Normal"],
                                      fontSize=8, textColor=colors.HexColor("#b2bec3"),
                                      alignment=TA_CENTER, spaceBefore=8)
    story.append(Paragraph(
        "⚠️ This report is AI-generated and for informational purposes only. "
        "Please consult a certified dermatologist for medical advice.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()