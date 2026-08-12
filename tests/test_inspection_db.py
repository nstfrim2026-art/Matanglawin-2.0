"""
Tests for inspection_db.py - SQLite-backed inspection record storage,
including the GPS-unavailable path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection_db import InspectionDB  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    d = InspectionDB(str(tmp_path / "test.db"))
    yield d
    d.close()


def test_add_and_get_inspection_with_gps(db):
    inspection_id = db.add_inspection(
        timestamp="2026-08-12T14:35:21",
        cropped_image_path="/data/crops/crack_001.jpg",
        confidence=0.942,
        image_path="/data/frames/frame_001.jpg",
        num_instances=1,
        latitude=14.5995,
        longitude=120.9842,
        altitude=35.2,
        detection_info={"bbox": [10, 20, 50, 60]},
    )
    record = db.get_inspection(inspection_id)
    assert record.gps_available is True
    assert record.latitude == 14.5995
    assert record.longitude == 120.9842
    assert record.altitude == 35.2
    assert record.detection_info == {"bbox": [10, 20, 50, 60]}


def test_add_inspection_without_gps_stores_nulls(db):
    inspection_id = db.add_inspection(
        timestamp="2026-08-12T14:40:00",
        cropped_image_path="/data/crops/crack_002.jpg",
        confidence=0.75,
    )
    record = db.get_inspection(inspection_id)
    assert record.gps_available is False
    assert record.latitude is None
    assert record.longitude is None
    assert record.altitude is None


def test_partial_gps_is_treated_as_unavailable(db):
    """Latitude with no longitude must not be reported as gps_available."""
    inspection_id = db.add_inspection(
        timestamp="2026-08-12T14:41:00",
        cropped_image_path="/data/crops/crack_003.jpg",
        confidence=0.5,
        latitude=14.6,
        longitude=None,
    )
    record = db.get_inspection(inspection_id)
    assert record.gps_available is False


def test_get_nonexistent_inspection_returns_none(db):
    assert db.get_inspection(9999) is None


def test_list_inspections_newest_first(db):
    id1 = db.add_inspection(timestamp="t1", cropped_image_path="a.jpg", confidence=0.5)
    id2 = db.add_inspection(timestamp="t2", cropped_image_path="b.jpg", confidence=0.6)
    records = db.list_inspections()
    assert records[0].id == id2
    assert records[1].id == id1


def test_get_latest_returns_most_recent(db):
    db.add_inspection(timestamp="t1", cropped_image_path="a.jpg", confidence=0.5)
    id2 = db.add_inspection(timestamp="t2", cropped_image_path="b.jpg", confidence=0.6)
    latest = db.get_latest()
    assert latest.id == id2


def test_get_latest_with_no_records_is_none(db):
    assert db.get_latest() is None


def test_count(db):
    assert db.count() == 0
    db.add_inspection(timestamp="t1", cropped_image_path="a.jpg", confidence=0.5)
    assert db.count() == 1


def test_delete_inspection(db):
    inspection_id = db.add_inspection(timestamp="t1", cropped_image_path="a.jpg", confidence=0.5)
    assert db.delete_inspection(inspection_id) is True
    assert db.get_inspection(inspection_id) is None
    assert db.delete_inspection(inspection_id) is False  # already deleted


def test_to_dict_shape(db):
    inspection_id = db.add_inspection(timestamp="t1", cropped_image_path="a.jpg", confidence=0.5)
    record = db.get_inspection(inspection_id)
    d = record.to_dict()
    for key in [
        "id", "timestamp", "image_path", "cropped_image_path", "confidence",
        "num_instances", "latitude", "longitude", "altitude", "gps_available",
        "detection_info",
    ]:
        assert key in d
