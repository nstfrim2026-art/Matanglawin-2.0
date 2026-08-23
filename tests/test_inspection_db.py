"""
Tests for inspection_db.py - storage of exactly the spec's history fields
(id, timestamp, source, original, highlighted, status), and nothing that
leaks a prohibited metric to the UI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection_db import InspectionDB, STATUS_CRACK, STATUS_NO_CRACK  # noqa: E402


def _db(tmp_path):
    return InspectionDB(str(tmp_path / "insp.db"))


def test_add_get_roundtrip(tmp_path):
    db = _db(tmp_path)
    iid = db.add_inspection(
        timestamp="2026-01-01T00:00:00", status=STATUS_CRACK, source="import",
        original_image_path="/o.jpg", highlighted_image_path="/h.jpg", num_instances=2,
    )
    rec = db.get_inspection(iid)
    assert rec.status == STATUS_CRACK
    assert rec.source == "import"
    assert rec.source_label == "DJI import"
    assert rec.has_crack is True
    assert rec.original_image_path == "/o.jpg"
    assert rec.highlighted_image_path == "/h.jpg"


def test_latest_and_ordering_and_count(tmp_path):
    db = _db(tmp_path)
    a = db.add_inspection(timestamp="t1", status=STATUS_NO_CRACK, source="upload")
    b = db.add_inspection(timestamp="t2", status=STATUS_CRACK, source="import")
    assert db.count() == 2
    assert db.get_latest().id == b
    ids = [r.id for r in db.list_inspections(limit=10)]
    assert ids == [b, a]  # newest first


def test_to_dict_only_exposes_allowed_fields(tmp_path):
    db = _db(tmp_path)
    iid = db.add_inspection(
        timestamp="t", status=STATUS_NO_CRACK, source="upload",
        original_image_path="/o.jpg", highlighted_image_path="/h.jpg", num_instances=0,
    )
    d = db.get_inspection(iid).to_dict()
    assert set(d.keys()) == {
        "id", "timestamp", "status", "has_crack", "source", "source_label",
        "gps_available", "latitude", "longitude",
    }
    assert d["status"] in (STATUS_CRACK, STATUS_NO_CRACK)


def test_source_label_for_upload(tmp_path):
    db = _db(tmp_path)
    iid = db.add_inspection(timestamp="t", status=STATUS_NO_CRACK, source="upload")
    assert db.get_inspection(iid).source_label == "Manual upload"


def test_delete(tmp_path):
    db = _db(tmp_path)
    iid = db.add_inspection(timestamp="t", status=STATUS_NO_CRACK)
    assert db.delete_inspection(iid) is True
    assert db.get_inspection(iid) is None
