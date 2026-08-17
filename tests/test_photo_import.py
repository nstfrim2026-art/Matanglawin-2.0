"""
Tests for photo_import.py - the automatic DJI photo bridge (watch folder).
Verifies automatic analysis on arrival, partial-copy safety, duplicate
protection, and invalid-file handling. Uses a stubbed predict_masks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from tests import stubs  # noqa: E402
from inspection_db import InspectionDB  # noqa: E402
from inspection_service import InspectionService  # noqa: E402
from photo_import import PhotoImportWatcher  # noqa: E402


def _watcher(tmp_path, stable_polls=1):
    db = InspectionDB(str(tmp_path / "insp.db"))
    svc = InspectionService(db, str(tmp_path / "inspections"), weights="x")
    imp = tmp_path / "import"
    w = PhotoImportWatcher(str(imp), svc, stable_polls=stable_polls)
    return w, db, imp


def test_photo_analyzed_automatically_on_arrival(tmp_path, monkeypatch):
    stubs.stub_one_crack(inference_core)
    w, db, imp = _watcher(tmp_path, stable_polls=1)
    stubs.write_jpg(imp / "DJI_0001.jpg")

    results = w.poll_once()
    assert len(results) == 1
    assert results[0]["action"] == "analyzed"
    assert results[0]["inspection_id"] is not None
    # The inspection was recorded with source 'import' (automatic path).
    rec = db.get_latest()
    assert rec.source == "import"


def test_partial_copy_is_not_analyzed_until_stable(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    w, db, imp = _watcher(tmp_path, stable_polls=2)
    stubs.write_jpg(imp / "DJI_0002.jpg")

    first = w.poll_once()   # size seen once - not yet stable
    assert first == []
    assert db.count() == 0

    second = w.poll_once()  # size unchanged -> stable -> analyzed
    assert any(r["action"] == "analyzed" for r in second)
    assert db.count() == 1


def test_duplicate_content_is_skipped(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    w, db, imp = _watcher(tmp_path, stable_polls=1)
    stubs.write_jpg(imp / "a.jpg", value=77, size=70)
    stubs.write_jpg(imp / "b.jpg", value=77, size=70)  # identical bytes

    results = w.poll_once()
    actions = sorted(r["action"] for r in results)
    assert actions == ["analyzed", "skipped_duplicate"]
    assert db.count() == 1  # only one analysis despite two files


def test_same_photo_not_reanalyzed_across_restarts(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    w, db, imp = _watcher(tmp_path, stable_polls=1)
    stubs.write_jpg(imp / "c.jpg", value=55)
    assert w.poll_once()[0]["action"] == "analyzed"

    # New watcher (simulated restart) reads the persisted processed-hash state.
    svc2 = InspectionService(db, str(tmp_path / "inspections"), weights="x")
    w2 = PhotoImportWatcher(str(imp), svc2, stable_polls=1)
    res2 = w2.poll_once()
    assert all(r["action"] != "analyzed" for r in res2)
    assert db.count() == 1


def test_invalid_file_is_handled_and_does_not_crash(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    w, db, imp = _watcher(tmp_path, stable_polls=1)
    (imp / "corrupt.jpg").write_bytes(b"not a real image at all")

    results = w.poll_once()
    assert results and results[0]["action"] == "invalid"
    assert db.count() == 0


def test_non_image_extensions_ignored(tmp_path, monkeypatch):
    stubs.stub_no_crack(inference_core)
    w, db, imp = _watcher(tmp_path, stable_polls=1)
    (imp / "notes.txt").write_text("hello")
    assert w.poll_once() == []
