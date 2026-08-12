"""
Tests for report_generator.py - status-based PDF generation, no
confidence in output, GPS optional, missing images tolerated.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection_db import InspectionRecord, STATUS_CRACK, STATUS_NO_CRACK  # noqa: E402
from report_generator import generate_single_report, generate_full_report  # noqa: E402


@pytest.fixture()
def img(tmp_path):
    p = tmp_path / "img.jpg"
    a = np.zeros((80, 80, 3), dtype=np.uint8)
    a[:] = (0, 0, 255)
    cv2.imwrite(str(p), a)
    return str(p)


def _crack(img, **over):
    kwargs = dict(
        id=1,
        timestamp="2026-08-12T14:35:21",
        status=STATUS_CRACK,
        source="import",
        original_image_path=img,
        highlighted_image_path=img,
        crack_image_paths=[img, img],
        num_instances=2,
        latitude=14.6,
        longitude=121.0,
        altitude=30.0,
        gps_available=True,
    )
    kwargs.update(over)
    return InspectionRecord(**kwargs)


def _valid_pdf(p):
    with open(p, "rb") as f:
        return f.read(5) == b"%PDF-"


def test_single_crack_report(tmp_path, img):
    out = tmp_path / "s.pdf"
    generate_single_report(_crack(img), str(out))
    assert _valid_pdf(out)


def test_single_no_crack_report(tmp_path, img):
    rec = InspectionRecord(
        id=2, timestamp="t", status=STATUS_NO_CRACK, source="upload",
        original_image_path=img, highlighted_image_path=img, crack_image_paths=[], num_instances=0,
    )
    out = tmp_path / "n.pdf"
    generate_single_report(rec, str(out))
    assert _valid_pdf(out)


def test_report_with_missing_images_does_not_crash(tmp_path):
    rec = _crack("/nope.jpg", original_image_path="/nope.jpg", highlighted_image_path="/no.jpg", crack_image_paths=["/no.png"])
    out = tmp_path / "m.pdf"
    generate_single_report(rec, str(out))
    assert _valid_pdf(out)


def test_full_report_mixed(tmp_path, img):
    recs = [
        _crack(img, id=1),
        InspectionRecord(id=2, timestamp="t", status=STATUS_NO_CRACK, source="upload",
                         original_image_path=img, highlighted_image_path=img, crack_image_paths=[], num_instances=0),
    ]
    out = tmp_path / "f.pdf"
    generate_full_report(recs, str(out))
    assert _valid_pdf(out)


def test_full_report_empty(tmp_path):
    out = tmp_path / "e.pdf"
    generate_full_report([], str(out))
    assert _valid_pdf(out)


def test_report_source_does_not_embed_confidence_text(tmp_path, img):
    """
    The report code path must not reference a confidence attribute
    (records don't even have one) - smoke test that generation works
    from a record whose to_dict() has no confidence.
    """
    rec = _crack(img)
    assert "confidence" not in rec.to_dict()
    out = tmp_path / "s2.pdf"
    generate_single_report(rec, str(out))
    assert _valid_pdf(out)
