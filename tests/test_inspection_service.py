"""
Tests for inspection_service.py - the single central pipeline shared by
manual upload and automatic DJI import. Verifies exactly two artifacts
(original + red-highlighted), never a crack-only image, and correct
CRACK / NO CRACK status. Uses a stubbed predict_masks (no real model).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from tests import stubs  # noqa: E402
from inspection_db import InspectionDB, STATUS_CRACK, STATUS_NO_CRACK  # noqa: E402
from inspection_service import InspectionService, InvalidImageError  # noqa: E402


def _service(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    return InspectionService(db, str(tmp_path / "inspections"), weights="x"), db


def test_no_crack_preserves_original_and_makes_clean_highlighted(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "in.jpg", value=90, size=80)

    rec = svc.analyze_file(str(src), source="upload")
    assert rec.status == STATUS_NO_CRACK
    assert rec.has_crack is False
    # Original is preserved and both files exist.
    assert Path(rec.original_image_path).exists()
    assert Path(rec.highlighted_image_path).exists()
    # For no-crack the "highlighted" is essentially the clean original (no red).
    orig = cv2.imread(rec.original_image_path)
    hi = cv2.imread(rec.highlighted_image_path)
    assert orig is not None and hi is not None
    assert np.array_equal(orig, hi)


def test_crack_produces_red_highlighted_different_from_original(tmp_path, monkeypatch):
    stubs.stub_one_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "in.jpg", value=90, size=80)

    rec = svc.analyze_file(str(src), source="import")
    assert rec.status == STATUS_CRACK
    assert rec.has_crack is True
    orig = cv2.imread(rec.original_image_path)
    hi = cv2.imread(rec.highlighted_image_path)
    # The highlighted image must differ from the original (red painted on).
    assert not np.array_equal(orig, hi)
    # Some pixels should be pushed toward red (BGR: high R channel).
    assert int(hi[:, :, 2].sum()) > int(orig[:, :, 2].sum())


def test_no_crack_only_artifacts_are_written(tmp_path, monkeypatch):
    stubs.stub_one_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "in.jpg", value=90, size=80)
    rec = svc.analyze_file(str(src), source="import")

    base = Path(rec.original_image_path).parent.parent
    # Only 'original' and 'highlighted' subfolders exist - never 'crack'.
    subdirs = {p.name for p in base.iterdir() if p.is_dir()}
    assert subdirs == {"original", "highlighted"}
    assert not (base / "crack").exists()
    # No stray crack-only / mask files anywhere in the inspection folder.
    names = [p.name.lower() for p in base.rglob("*") if p.is_file()]
    assert not any("crack_0" in n or "mask" in n for n in names)


def test_invalid_image_raises(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    svc, db = _service(tmp_path)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image")
    try:
        svc.analyze_file(str(bad), source="upload")
        assert False, "expected InvalidImageError"
    except InvalidImageError:
        pass


def test_manual_and_import_converge_on_same_service(tmp_path, monkeypatch):
    stubs.stub_one_crack(inference_core)
    svc, db = _service(tmp_path)
    a = svc.analyze_file(str(stubs.write_jpg(tmp_path / "a.jpg")), source="upload")
    b = svc.analyze_file(str(stubs.write_jpg(tmp_path / "b.jpg")), source="import")
    assert a.source == "upload" and b.source == "import"
    # Both went through the same pipeline and produced the same status.
    assert a.status == b.status == STATUS_CRACK
    assert db.count() == 2
