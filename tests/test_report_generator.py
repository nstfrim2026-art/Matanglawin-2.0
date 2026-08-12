"""
Tests for report_generator.py - PDF generation with and without GPS,
and other edge cases (missing image, empty record set).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection_db import InspectionRecord  # noqa: E402
from report_generator import generate_single_report, generate_full_report  # noqa: E402


@pytest.fixture()
def sample_image(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    path = tmp_path / "crop.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def _record(sample_image, **overrides):
    defaults = dict(
        id=1,
        timestamp="2026-08-12 14:35:21",
        cropped_image_path=sample_image,
        confidence=0.942,
        num_instances=1,
        latitude=None,
        longitude=None,
        altitude=None,
        gps_available=False,
    )
    defaults.update(overrides)
    return InspectionRecord(**defaults)


def _is_valid_pdf(path: Path) -> bool:
    with open(path, "rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


def test_single_report_with_gps(tmp_path, sample_image):
    record = _record(
        sample_image, latitude=14.5995, longitude=120.9842, altitude=35.2, gps_available=True
    )
    out_path = tmp_path / "single_gps.pdf"
    generate_single_report(record, str(out_path))
    assert out_path.exists()
    assert _is_valid_pdf(out_path)
    assert out_path.stat().st_size > 0


def test_single_report_without_gps_still_generates(tmp_path, sample_image):
    record = _record(sample_image, gps_available=False)
    out_path = tmp_path / "single_no_gps.pdf"
    generate_single_report(record, str(out_path))
    assert out_path.exists()
    assert _is_valid_pdf(out_path)


def test_single_report_with_missing_image_does_not_crash(tmp_path):
    record = _record("/definitely/does/not/exist.jpg")
    out_path = tmp_path / "single_missing_image.pdf"
    generate_single_report(record, str(out_path))
    assert out_path.exists()
    assert _is_valid_pdf(out_path)


def test_full_report_mixed_gps_and_no_gps(tmp_path, sample_image):
    records = [
        _record(sample_image, id=1, latitude=14.6, longitude=121.0, gps_available=True),
        _record(sample_image, id=2, gps_available=False),
    ]
    out_path = tmp_path / "full.pdf"
    generate_full_report(records, str(out_path))
    assert out_path.exists()
    assert _is_valid_pdf(out_path)


def test_full_report_with_empty_list_still_generates_valid_pdf(tmp_path):
    out_path = tmp_path / "empty.pdf"
    generate_full_report([], str(out_path))
    assert out_path.exists()
    assert _is_valid_pdf(out_path)


def test_full_report_creates_parent_directories(tmp_path, sample_image):
    out_path = tmp_path / "nested" / "dir" / "full.pdf"
    generate_full_report([_record(sample_image)], str(out_path))
    assert out_path.exists()


def test_pdf_generation_failure_is_a_clear_exception(tmp_path, sample_image, monkeypatch):
    """
    Callers (app.py routes) are expected to catch exceptions from this
    module and surface them as a 500 with a clear message - verify that
    a genuine failure (e.g. reportlab raising) propagates as an
    exception rather than silently producing a corrupt file.
    """
    import report_generator

    def broken_build(self, *args, **kwargs):
        raise RuntimeError("simulated reportlab failure")

    monkeypatch.setattr(report_generator.SimpleDocTemplate, "build", broken_build)

    record = _record(sample_image)
    out_path = tmp_path / "should_fail.pdf"
    with pytest.raises(RuntimeError):
        generate_single_report(record, str(out_path))
