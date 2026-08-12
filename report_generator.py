"""
report_generator.py - PDF inspection report generation.

Turns one or more `inspection_db.InspectionRecord`s into a PDF report
using `reportlab` (pure-Python, no external binary dependency - safe to
bundle into the PyInstaller executable like the rest of the app).

Two entry points:

    generate_single_report(record, out_path)
        One inspection (original + red-highlighted + crack-only images).

    generate_full_report(records, out_path)
        A report covering many inspections (e.g. a whole session).

The report is intentionally simple and mirrors the result page: it shows
the inspection status ("CRACK DETECTED" / "NO CRACK DETECTED"), the
timestamp, GPS (when available), and the result images. It deliberately
does NOT show confidence percentages, probability scores, or any raw
model statistics - matching the "no confidence in the UI" requirement.

GPS is optional everywhere: if a record has no GPS, the report prints
"GPS: Unavailable" and generation proceeds normally.
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

from inspection_db import InspectionRecord, STATUS_CRACK

_styles = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "MatanglawinTitle", parent=_styles["Title"], textColor=colors.HexColor("#b91c1c")
)
_HEADING_STYLE = _styles["Heading2"]
_BODY_STYLE = _styles["BodyText"]
_STATUS_CRACK_STYLE = ParagraphStyle(
    "StatusCrack", parent=_styles["Heading3"], textColor=colors.HexColor("#b91c1c")
)
_STATUS_OK_STYLE = ParagraphStyle(
    "StatusOk", parent=_styles["Heading3"], textColor=colors.HexColor("#15803d")
)

_IMG_W = 3.0 * inch
_IMG_H = 3.0 * inch


def _gps_text(record: InspectionRecord) -> str:
    if not record.gps_available or record.latitude is None or record.longitude is None:
        return "GPS: Unavailable"
    parts = [f"Latitude: {record.latitude:.6f}", f"Longitude: {record.longitude:.6f}"]
    if record.altitude is not None:
        parts.append(f"Altitude: {record.altitude:.1f} m")
    return " | ".join(parts)


def _image_flowable(path: Optional[str], caption: str) -> list:
    flowables = [Paragraph(caption, _BODY_STYLE)]
    if path and Path(path).exists():
        try:
            flowables.append(Image(path, width=_IMG_W, height=_IMG_H, kind="proportional"))
        except Exception:
            flowables.append(Paragraph("(unable to load image)", _BODY_STYLE))
    else:
        flowables.append(Paragraph("(not available)", _BODY_STYLE))
    flowables.append(Spacer(1, 8))
    return flowables


def _inspection_flowables(record: InspectionRecord, index: Optional[int] = None) -> list:
    flowables = []
    label = f"Inspection #{index:03d}" if index is not None else f"Inspection #{record.id}"
    flowables.append(Paragraph(label, _HEADING_STYLE))

    status_style = _STATUS_CRACK_STYLE if record.status == STATUS_CRACK else _STATUS_OK_STYLE
    flowables.append(Paragraph(record.status, status_style))

    info_rows = [
        ["Timestamp", record.timestamp],
        ["Source", record.source],
        ["Cracks detected", str(record.num_instances)],
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

    # Original + highlighted always shown.
    flowables.extend(_image_flowable(record.original_image_path, "Original photo:"))
    if record.status == STATUS_CRACK:
        flowables.extend(_image_flowable(record.highlighted_image_path, "Crack highlighted (red):"))
        for i, crack_path in enumerate(record.crack_image_paths, start=1):
            flowables.extend(_image_flowable(crack_path, f"Crack only #{i}:"))

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
    Generate a one-inspection PDF report. Never raises just because GPS
    is unavailable or an image is missing; only raises for genuine
    I/O/library failures (which the caller surfaces as "PDF generation
    failed").
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)

    story = _header_flowables("Crack Inspection Report")
    story.extend(_inspection_flowables(record))

    doc.build(story)
    return str(out_path)


def generate_full_report(records: Iterable[InspectionRecord], out_path: str) -> str:
    """
    Generate a PDF covering every record in `records`. If `records` is
    empty, a valid PDF stating "No inspections recorded" is still
    produced (report generation must not fail just because there's
    nothing to report yet).
    """
    records: List[InspectionRecord] = list(records)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)

    crack_count = sum(1 for r in records if r.status == STATUS_CRACK)
    story = _header_flowables(
        "Full Inspection Report",
        subtitle=f"Total inspections: {len(records)} | With cracks: {crack_count}",
    )

    if not records:
        story.append(Paragraph("No inspections recorded.", _BODY_STYLE))
    else:
        for i, record in enumerate(records, start=1):
            story.extend(_inspection_flowables(record, index=i))
            if i < len(records):
                story.append(PageBreak())

    doc.build(story)
    return str(out_path)
