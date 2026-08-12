"""
report_generator.py - PDF inspection report generation.

Turns one or more `inspection_db.InspectionRecord`s into a PDF report
using `reportlab` (pure-Python, no external binary dependency - safe to
bundle into the PyInstaller executable like the rest of the app).

Two entry points:

    generate_single_report(record, out_path)
        One-page-per-crack report for a single inspection event.

    generate_full_report(records, out_path)
        Multi-crack report covering a batch/session of inspections
        (e.g. everything captured on a given day/flight).

GPS is optional everywhere: if `record.gps_available` is False (or
latitude/longitude are None), the report prints "GPS: Unavailable"
instead of blank/zero coordinates, and generation proceeds normally -
missing GPS must never block report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from inspection_db import InspectionRecord

_styles = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "MatanglawinTitle", parent=_styles["Title"], textColor=colors.HexColor("#b91c1c")
)
_HEADING_STYLE = _styles["Heading2"]
_BODY_STYLE = _styles["BodyText"]


def _gps_text(record: InspectionRecord) -> str:
    if not record.gps_available or record.latitude is None or record.longitude is None:
        return "GPS: Unavailable"
    parts = [f"Latitude: {record.latitude:.6f}", f"Longitude: {record.longitude:.6f}"]
    if record.altitude is not None:
        parts.append(f"Altitude: {record.altitude:.1f} m")
    return " | ".join(parts)


def _crack_flowables(record: InspectionRecord, index: Optional[int] = None) -> list:
    flowables = []
    label = f"Crack #{index:03d}" if index is not None else f"Inspection #{record.id}"
    flowables.append(Paragraph(label, _HEADING_STYLE))

    info_rows = [
        ["Timestamp", record.timestamp],
        ["Detection Confidence", f"{record.confidence * 100:.1f}%"],
        ["Crack Instances", str(record.num_instances)],
        ["GPS", _gps_text(record)],
    ]
    table = Table(info_rows, colWidths=[1.8 * inch, 4.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#374151")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
            ]
        )
    )
    flowables.append(table)
    flowables.append(Spacer(1, 10))

    image_path = record.cropped_image_path
    if image_path and Path(image_path).exists():
        try:
            img = Image(image_path, width=3.2 * inch, height=3.2 * inch, kind="proportional")
            flowables.append(Paragraph("Captured Image:", _BODY_STYLE))
            flowables.append(img)
        except Exception:
            flowables.append(Paragraph("Captured Image: (unable to load image)", _BODY_STYLE))
    else:
        flowables.append(Paragraph("Captured Image: (not available)", _BODY_STYLE))

    flowables.append(Spacer(1, 18))
    return flowables


def _header_flowables(title: str, subtitle: str = "") -> list:
    flowables = [Paragraph("MATANGLAWIN", _TITLE_STYLE)]
    flowables.append(Paragraph(title, _HEADING_STYLE))
    if subtitle:
        flowables.append(Paragraph(subtitle, _BODY_STYLE))
    flowables.append(Spacer(1, 16))
    return flowables


def generate_single_report(record: InspectionRecord, out_path: str) -> str:
    """
    Generate a one-crack PDF inspection report. Never raises just
    because GPS is unavailable; only raises for genuine I/O/library
    failures (which the caller should surface as "PDF generation
    failed", per the error-handling requirements).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)

    story = _header_flowables("Crack Inspection Report")
    story.extend(_crack_flowables(record))

    doc.build(story)
    return str(out_path)


def generate_full_report(records: Iterable[InspectionRecord], out_path: str) -> str:
    """
    Generate a multi-crack PDF report covering every record in
    `records` (e.g. a full inspection session). Records without GPS are
    included exactly like records with GPS - only the GPS line differs.
    If `records` is empty, a valid PDF stating "No inspections
    recorded" is still produced (report generation must not fail just
    because there's nothing to report yet).
    """
    records: List[InspectionRecord] = list(records)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)

    story = _header_flowables(
        "Full Inspection Report",
        subtitle=f"Total cracks recorded: {len(records)}",
    )

    if not records:
        story.append(Paragraph("No inspections recorded.", _BODY_STYLE))
    else:
        for i, record in enumerate(records, start=1):
            story.extend(_crack_flowables(record, index=i))
            if i < len(records):
                story.append(PageBreak())

    doc.build(story)
    return str(out_path)
