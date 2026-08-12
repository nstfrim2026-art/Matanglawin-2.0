"""
Tests for capture_manager.py - automatic crack capture with
cooldown/debounce, confidence filtering, and GPS attachment.
"""

import shutil
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

TEST_DB_DIR = Path(__file__).resolve().parent / ".tmp_capture_test"


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
    return np.zeros((200, 200, 3), dtype=np.uint8)


def _no_detection():
    return DetectionResult(has_crack=False, instances=[], union_mask=None, overlay_frame=None, max_confidence=0.0)


def _low_conf_detection():
    inst = CrackInstance(confidence=0.1, bbox=(10, 10, 50, 50), area_px=1600)
    return DetectionResult(has_crack=True, instances=[inst], union_mask=None, overlay_frame=None, max_confidence=0.1)


def _good_detection(conf=0.9):
    inst = CrackInstance(confidence=conf, bbox=(10, 10, 60, 60), area_px=2500)
    return DetectionResult(has_crack=True, instances=[inst], union_mask=None, overlay_frame=None, max_confidence=conf)


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
    saved_files = list(capture_dir.glob("*.jpg"))
    assert len(saved_files) == 1


def test_capture_is_written_to_inspection_db(env):
    db, manager, _ = env
    result = manager.maybe_capture(_frame(), _good_detection(conf=0.87))
    record = db.get_inspection(result.inspection_id)
    assert record is not None
    assert abs(record.confidence - 0.87) < 1e-6


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
