"""
Tests for capture_manager.py - automatic crack capture with
cooldown/debounce, confidence filtering, GPS attachment, and the
original/crack/overlays folder layout.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gps_provider  # noqa: E402
from capture_manager import CaptureManager  # noqa: E402
from detector import DetectionResult, CrackInstance  # noqa: E402
from inspection_db import InspectionDB  # noqa: E402


@pytest.fixture()
def env(tmp_path):
    db_path = tmp_path / "inspections.db"
    capture_dir = tmp_path / "captures"
    db = InspectionDB(str(db_path))
    manager = CaptureManager(db, str(capture_dir), cooldown_sec=1.0, min_confidence=0.3)
    yield db, manager, capture_dir
    db.close()


@pytest.fixture(autouse=True)
def reset_gps():
    gps_provider.set_gps_source(None)
    yield
    gps_provider.set_gps_source(None)


def _frame():
    # Light gray "concrete" background so we can distinguish "masked out"
    # (black) pixels from "real background" pixels in assertions below.
    return np.full((200, 200, 3), 200, dtype=np.uint8)


def _mask_for_bbox(bbox, fill_ratio=1.0):
    """A simple filled-rectangle mask sized to `bbox`, for test fixtures."""
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    mask = np.zeros((h, w), dtype=np.uint8)
    if fill_ratio >= 1.0:
        mask[:, :] = 1
    else:
        # only fill a thin diagonal strip, to simulate a real crack mask
        # that does NOT cover the whole bounding box.
        n = min(h, w)
        for i in range(n):
            mask[i, i] = 1
    return mask


def _no_detection():
    return DetectionResult(has_crack=False, instances=[], max_confidence=0.0)


def _low_conf_detection():
    bbox = (10, 10, 50, 50)
    inst = CrackInstance(confidence=0.1, bbox=bbox, area_px=1600, mask=_mask_for_bbox(bbox))
    return DetectionResult(has_crack=True, instances=[inst], max_confidence=0.1)


def _good_detection(conf=0.9, thin_mask=False):
    bbox = (10, 10, 60, 60)
    mask = _mask_for_bbox(bbox, fill_ratio=(0.0 if thin_mask else 1.0))
    inst = CrackInstance(confidence=conf, bbox=bbox, area_px=2500, mask=mask)
    return DetectionResult(has_crack=True, instances=[inst], max_confidence=conf)


def test_no_detection_is_not_captured(env):
    _, manager, _ = env
    result = manager.maybe_capture(_frame(), _no_detection())
    assert result.saved is False
    assert result.reason == "no_detection"


def test_low_confidence_is_not_captured(env):
    _, manager, _ = env
    result = manager.maybe_capture(_frame(), _low_conf_detection())
    assert result.saved is False
    assert result.reason == "low_confidence"


def test_valid_detection_is_captured_and_saved_to_disk(env):
    db, manager, capture_dir = env
    result = manager.maybe_capture(_frame(), _good_detection())
    assert result.saved is True
    assert result.inspection_id is not None
    assert len(list((capture_dir / "original").glob("*.jpg"))) == 1
    assert len(list((capture_dir / "crack").glob("*.jpg"))) == 1


def test_overlay_folder_gets_an_internal_diagnostic_crop(env):
    """
    captures/overlays/ is populated for internal diagnostics, but this
    is never exposed through any route - see test_app_routes.py's
    test_stream_preview_route_does_not_exist.
    """
    _, manager, capture_dir = env
    manager.maybe_capture(_frame(), _good_detection())
    assert len(list((capture_dir / "overlays").glob("*.jpg"))) == 1


def test_capture_is_written_to_inspection_db(env):
    db, manager, _ = env
    result = manager.maybe_capture(_frame(), _good_detection(conf=0.87))
    record = db.get_inspection(result.inspection_id)
    assert record is not None
    assert abs(record.confidence - 0.87) < 1e-6


def test_inspection_record_has_both_original_and_crack_paths(env):
    db, manager, capture_dir = env
    result = manager.maybe_capture(_frame(), _good_detection())
    record = db.get_inspection(result.inspection_id)
    assert record.image_path is not None
    assert "original" in record.image_path
    assert record.cropped_image_path is not None
    assert "crack" in record.cropped_image_path


def test_crack_only_result_uses_segmentation_mask_not_full_bbox(env):
    """
    The saved crack-only image must follow the mask shape, not just
    fill the whole bounding-box rectangle: for a thin diagonal mask,
    most of the saved crop should be black (masked out), not the full
    gray "concrete" background.
    """
    import cv2

    _, manager, capture_dir = env
    result = manager.maybe_capture(_frame(), _good_detection(thin_mask=True))
    assert result.saved is True

    crack_files = list((capture_dir / "crack").glob("*.jpg"))
    assert len(crack_files) == 1
    saved = cv2.imread(str(crack_files[0]))
    assert saved is not None

    nonblack_fraction = (saved.sum(axis=2) > 0).mean()
    # A thin diagonal mask covers a small fraction of its bounding box;
    # the saved crack-only image must reflect that, not be ~fully gray.
    assert nonblack_fraction < 0.5


def test_cooldown_blocks_immediate_second_capture(env):
    _, manager, _ = env
    r1 = manager.maybe_capture(_frame(), _good_detection())
    assert r1.saved is True

    r2 = manager.maybe_capture(_frame(), _good_detection())
    assert r2.saved is False
    assert r2.reason == "cooldown"


def test_capture_allowed_again_after_cooldown_elapses(env):
    _, manager, _ = env
    r1 = manager.maybe_capture(_frame(), _good_detection())
    assert r1.saved is True

    time.sleep(1.1)  # cooldown_sec=1.0 in the fixture

    r2 = manager.maybe_capture(_frame(), _good_detection())
    assert r2.saved is True
    assert r2.inspection_id != r1.inspection_id


def test_reset_cooldown_allows_immediate_recapture(env):
    _, manager, _ = env
    r1 = manager.maybe_capture(_frame(), _good_detection())
    assert r1.saved is True

    manager.reset_cooldown()

    r2 = manager.maybe_capture(_frame(), _good_detection())
    assert r2.saved is True


def test_capture_without_gps_stores_null_coordinates(env):
    db, manager, _ = env
    result = manager.maybe_capture(_frame(), _good_detection())
    record = db.get_inspection(result.inspection_id)
    assert record.gps_available is False
    assert record.latitude is None
    assert record.longitude is None


def test_capture_with_gps_stores_real_coordinates(env):
    db, manager, _ = env
    gps_provider.set_gps_source(
        lambda: gps_provider.GpsFix(latitude=14.5995, longitude=120.9842, altitude=35.2)
    )
    result = manager.maybe_capture(_frame(), _good_detection())
    record = db.get_inspection(result.inspection_id)
    assert record.gps_available is True
    assert record.latitude == 14.5995
    assert record.longitude == 120.9842
    assert record.altitude == 35.2


def test_gps_never_fabricated_when_source_returns_partial_data(env):
    """A GpsFix missing longitude must still be treated as unavailable."""
    db, manager, _ = env
    gps_provider.set_gps_source(lambda: gps_provider.GpsFix(latitude=14.6, longitude=None))
    result = manager.maybe_capture(_frame(), _good_detection())
    record = db.get_inspection(result.inspection_id)
    assert record.gps_available is False
    assert record.latitude is None
    assert record.longitude is None
