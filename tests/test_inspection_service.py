"""
Tests for inspection_service.py - the single central photo-analysis
pipeline shared by manual upload and DJI import.

YOLO is stubbed via inference_core.predict_masks so no real model is
needed and detection is deterministic.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gps_provider  # noqa: E402
import inference_core  # noqa: E402
from inspection_db import InspectionDB, STATUS_CRACK, STATUS_NO_CRACK  # noqa: E402
from inspection_service import InspectionService, InvalidImageError  # noqa: E402


class _Conf:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.v)


class _Boxes:
    def __init__(self, c):
        self.conf = _Conf(c)


class _MD:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Masks:
    def __init__(self, a):
        self._a = a

    @property
    def data(self):
        return _MD(self._a)

    def __len__(self):
        return self._a.shape[0]


class _Result:
    def __init__(self, masks, boxes, orig):
        self.masks = masks
        self.boxes = boxes
        self.orig_img = orig


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    orig = inference_core.predict_masks
    gps_provider.set_gps_source(None)
    yield
    inference_core.predict_masks = orig
    gps_provider.set_gps_source(None)


@pytest.fixture()
def env(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    svc = InspectionService(db, str(tmp_path / "inspections"), weights="best.pt")
    yield db, svc, tmp_path
    db.close()


def _frame(size=200, value=190):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _thin_diagonal(size=200, box=(20, 20, 90, 90)):
    # A diagonal band ~4px wide: well above the 150px noise floor, but
    # still a small fraction of its bounding box (so crack-only output
    # should be mostly black, not a filled rectangle).
    m = np.zeros((size, size), dtype=np.float32)
    x1, y1, x2, y2 = box
    n = min(x2 - x1, y2 - y1)
    for i in range(n):
        for w in range(4):
            if x1 + i + w < size:
                m[y1 + i, x1 + i + w] = 1.0
    return m


def _set_masks(masks, confs):
    inference_core.predict_masks = lambda *a, **k: _Result(_Masks(np.array(masks)), _Boxes(confs), a[0])


def _set_no_crack():
    inference_core.predict_masks = lambda *a, **k: _Result(None, None, a[0])


def test_no_crack_status_and_preserved_original(env):
    db, svc, _ = env
    _set_no_crack()
    rec = svc.analyze_array(_frame(), source="upload")
    assert rec.status == STATUS_NO_CRACK
    assert rec.num_instances == 0
    assert rec.crack_image_paths == []
    assert Path(rec.original_image_path).exists()
    # For no-crack, highlighted is just a copy of the original.
    assert Path(rec.highlighted_image_path).exists()


def test_crack_status_generates_all_artifacts(env):
    db, svc, _ = env
    _set_masks([_thin_diagonal()], [0.9])
    rec = svc.analyze_array(_frame(), source="upload")
    assert rec.status == STATUS_CRACK
    assert rec.num_instances == 1
    assert len(rec.crack_image_paths) == 1
    assert Path(rec.original_image_path).exists()
    assert Path(rec.highlighted_image_path).exists()
    assert all(Path(p).exists() for p in rec.crack_image_paths)


def test_highlighted_differs_from_original_but_original_untouched(env):
    db, svc, _ = env
    frame = _frame()
    _set_masks([_thin_diagonal()], [0.9])
    rec = svc.analyze_array(frame, source="upload")
    orig = cv2.imread(rec.original_image_path)
    hl = cv2.imread(rec.highlighted_image_path)
    assert (orig != hl).any()               # red highlight applied
    assert (orig == frame).all()             # original preserved as captured


def test_crack_only_is_mask_based_not_rectangular(env):
    db, svc, _ = env
    _set_masks([_thin_diagonal()], [0.9])
    rec = svc.analyze_array(_frame(), source="upload")
    crop = cv2.imread(rec.crack_image_paths[0])
    nonblack = (crop.sum(axis=2) > 0).mean()
    assert nonblack < 0.3  # thin crack -> mostly black, not a slab of surface


def test_multiple_cracks_supported(env):
    db, svc, _ = env
    m1 = _thin_diagonal(box=(10, 10, 80, 80))
    m2 = np.zeros((200, 200), dtype=np.float32)
    m2[120:150, 120:170] = 1.0
    _set_masks([m1, m2], [0.9, 0.7])
    rec = svc.analyze_array(_frame(), source="import")
    assert rec.status == STATUS_CRACK
    assert rec.num_instances == 2
    assert len(rec.crack_image_paths) == 2


def test_upload_and_import_use_the_same_pipeline(env):
    """Same image + same model must yield the same status via either source."""
    db, svc, tmp = env
    _set_masks([_thin_diagonal()], [0.9])
    frame = _frame()
    p = tmp / "photo.jpg"
    cv2.imwrite(str(p), frame)

    up = svc.analyze_array(frame, source="upload")
    imp = svc.analyze_file(str(p), source="import")
    assert up.status == imp.status == STATUS_CRACK
    assert up.num_instances == imp.num_instances
    assert up.source == "upload" and imp.source == "import"


def test_invalid_image_raises(env):
    db, svc, tmp = env
    bad = tmp / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        svc.analyze_file(str(bad))


def test_gps_attached_when_available(env):
    db, svc, _ = env
    gps_provider.set_gps_source(lambda: gps_provider.GpsFix(14.6, 121.0, 12.0))
    _set_no_crack()
    rec = svc.analyze_array(_frame())
    assert rec.gps_available is True
    assert rec.latitude == 14.6


def test_gps_unavailable_does_not_block_analysis(env):
    db, svc, _ = env
    _set_masks([_thin_diagonal()], [0.9])
    rec = svc.analyze_array(_frame())
    assert rec.status == STATUS_CRACK
    assert rec.gps_available is False
    assert rec.latitude is None


def test_metadata_json_written(env):
    db, svc, _ = env
    _set_no_crack()
    rec = svc.analyze_array(_frame())
    meta = Path(rec.original_image_path).parent.parent / "metadata.json"
    assert meta.exists()
