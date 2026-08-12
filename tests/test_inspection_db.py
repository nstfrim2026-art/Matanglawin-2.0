"""
Tests for inspection_db.py - photo-inspection record storage, the
GPS-unavailable path, no-confidence serialization, and legacy migration.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection_db import InspectionDB, STATUS_CRACK, STATUS_NO_CRACK  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    d = InspectionDB(str(tmp_path / "test.db"))
    yield d
    d.close()


def _add_crack(db, **over):
    kwargs = dict(
        timestamp="2026-08-12T14:35:21",
        status=STATUS_CRACK,
        source="import",
        original_image_path="/i/a/original/photo.jpg",
        highlighted_image_path="/i/a/highlighted/h.jpg",
        crack_image_paths=["/i/a/crack/crack_001.png", "/i/a/crack/crack_002.png"],
        num_instances=2,
        latitude=14.5995,
        longitude=120.9842,
        altitude=35.2,
        detection_info={"instances": [{"confidence": 0.9}]},
    )
    kwargs.update(over)
    return db.add_inspection(**kwargs)


def test_add_and_get_crack_with_gps(db):
    rid = _add_crack(db)
    r = db.get_inspection(rid)
    assert r.status == STATUS_CRACK
    assert r.has_crack is True
    assert r.gps_available is True
    assert r.latitude == 14.5995
    assert r.num_instances == 2
    assert r.crack_image_names() == ["crack_001.png", "crack_002.png"]


def test_add_no_crack_without_gps(db):
    rid = db.add_inspection(
        timestamp="2026-08-12T14:40:00",
        status=STATUS_NO_CRACK,
        source="upload",
        original_image_path="/i/b/original/photo.jpg",
        highlighted_image_path="/i/b/original/photo.jpg",
        crack_image_paths=[],
        num_instances=0,
    )
    r = db.get_inspection(rid)
    assert r.status == STATUS_NO_CRACK
    assert r.has_crack is False
    assert r.gps_available is False
    assert r.latitude is None
    assert r.crack_image_names() == []


def test_partial_gps_is_treated_as_unavailable(db):
    rid = _add_crack(db, latitude=14.6, longitude=None)
    assert db.get_inspection(rid).gps_available is False


def test_to_dict_never_exposes_confidence(db):
    rid = _add_crack(db)
    d = db.get_inspection(rid).to_dict()
    assert "confidence" not in d
    for key in [
        "id", "timestamp", "status", "has_crack", "source", "num_instances",
        "latitude", "longitude", "altitude", "gps_available", "crack_image_names",
    ]:
        assert key in d


def test_list_newest_first_and_latest(db):
    id1 = _add_crack(db)
    id2 = db.add_inspection(timestamp="t2", status=STATUS_NO_CRACK)
    records = db.list_inspections()
    assert records[0].id == id2
    assert records[1].id == id1
    assert db.get_latest().id == id2


def test_count_and_delete(db):
    assert db.count() == 0
    rid = _add_crack(db)
    assert db.count() == 1
    assert db.delete_inspection(rid) is True
    assert db.get_inspection(rid) is None
    assert db.delete_inspection(rid) is False


def test_get_latest_empty_is_none(db):
    assert db.get_latest() is None


def test_legacy_schema_is_migrated_in_place(tmp_path):
    """A DB created by the OLD live-detection schema must upgrade cleanly."""
    legacy_path = tmp_path / "legacy.db"
    c = sqlite3.connect(str(legacy_path))
    c.execute(
        """CREATE TABLE inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            image_path TEXT, cropped_image_path TEXT NOT NULL, confidence REAL NOT NULL,
            num_instances INTEGER NOT NULL DEFAULT 1, latitude REAL, longitude REAL,
            altitude REAL, gps_available INTEGER NOT NULL DEFAULT 0, detection_info TEXT)"""
    )
    c.execute(
        "INSERT INTO inspections (timestamp, image_path, cropped_image_path, confidence, num_instances, latitude, longitude, gps_available) VALUES (?,?,?,?,?,?,?,?)",
        ("2026-01-01T00:00:00", "/o/orig.jpg", "/o/crack.jpg", 0.8, 1, 14.6, 121.0, 1),
    )
    c.commit()
    c.close()

    db = InspectionDB(str(legacy_path))
    recs = db.list_inspections()
    assert len(recs) == 1
    r = recs[0]
    assert r.status == STATUS_CRACK          # had a crack image -> CRACK
    assert r.original_image_path == "/o/orig.jpg"
    assert r.crack_image_paths == ["/o/crack.jpg"]
    assert r.gps_available is True

    # New-shape inserts must work after migration (old NOT NULL columns gone).
    new_id = db.add_inspection(timestamp="2026-02-02T00:00:00", status=STATUS_NO_CRACK)
    assert db.count() == 2

    # Reopening must be a no-op (idempotent migration).
    db.close()
    db2 = InspectionDB(str(legacy_path))
    assert db2.count() == 2
    db2.close()
