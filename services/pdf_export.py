"""Exports a generated 18-section Cynbit AutoBlueprint to a styled PDF using ReportLab."""

from __future__ import annotations

import os
import re
from typing import Any, Dict
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer


def export_blueprint_to_pdf(blueprint_record: Dict[str, Any], output_pdf_path: str) -> str:
    """Renders the complete 18-section Technical Blueprint to a multi-page PDF file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Cynbit AutoBlueprint - {blueprint_record.get('project_title', 'Technical Blueprint')}",
        author="Cynbit AutoBlueprint Multi-Agent System",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BlueprintTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0F2942"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "BlueprintMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "BlueprintH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0F2942"),
        spaceBefore=10,
        spaceAfter=4,
    )
    h3_style = ParagraphStyle(
        "BlueprintH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0D9488"),
        spaceBefore=6,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BlueprintBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "BlueprintBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3.5,
    )
    code_style = ParagraphStyle(
        "BlueprintCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#0F172A"),
        leftIndent=10,
        spaceAfter=2,
    )

    story = []
    md_text = blueprint_record.get("blueprint_markdown", "")
    if not md_text:
        from services.blueprint_generator import format_blueprint_markdown
        md_text = format_blueprint_markdown(blueprint_record)

    in_code_block = False
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            story.append(Spacer(1, 3))
            continue

        if in_code_block:
            story.append(Paragraph(escape(line), code_style))
            continue

        if not line.strip():
            story.append(Spacer(1, 3))
            continue

        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:].strip()), title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F2942"), spaceAfter=6))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:].strip()), h2_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=4))
        elif line.startswith("### "):
            story.append(Paragraph(escape(line[4:].strip()), h3_style))
        elif line.startswith("- "):
            clean_b = escape(line[2:].strip())
            clean_b = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean_b)
            clean_b = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", clean_b)
            story.append(Paragraph(f"• {clean_b}", bullet_style))
        elif line.startswith("> "):
            clean_q = escape(line[2:].strip())
            clean_q = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean_q)
            story.append(Paragraph(f"<i>{clean_q}</i>", meta_style))
        else:
            clean_p = escape(line.strip())
            clean_p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean_p)
            clean_p = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", clean_p)
            style_to_use = meta_style if ("Client Organization:" in line or "Industry:" in line or "Critic Quality Score:" in line) else body_style
            story.append(Paragraph(clean_p, style_to_use))

    doc.build(story)
    return output_pdf_path
