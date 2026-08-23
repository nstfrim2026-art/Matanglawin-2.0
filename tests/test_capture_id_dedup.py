"""
Tests for the "one capture -> one inspection" fix.

A single physical capture is identified by a capture_id that flows through
the whole pipeline (image save -> import -> service -> DB), with a DB-level
UNIQUE constraint as the final guard. These tests verify:

  * processing the SAME capture_id twice yields exactly ONE record;
  * the DB rejects a duplicate capture_id (sqlite IntegrityError);
  * different captures (and manual uploads with capture_id=None) still
    create separate records;
  * the two automatic import transports (watch folder + HTTP /api/import)
    share the identity, so the same photo via both routes is stored once.

Segmentation is stubbed (no real model).
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from import_ledger import hash_bytes  # noqa: E402
from inspection_db import InspectionDB, STATUS_NO_CRACK  # noqa: E402
from inspection_service import InspectionService  # noqa: E402
from photo_import import PhotoImportWatcher  # noqa: E402
from tests import stubs  # noqa: E402


def _service(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    return InspectionService(db, str(tmp_path / "inspections"), weights="x"), db


# -- service-level dedup ---------------------------------------------

def test_same_capture_id_processed_twice_is_one_record(tmp_path):
    stubs.stub_one_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "cap.jpg", value=10)

    a = svc.analyze_file(str(src), source="import", capture_id="CAP_1")
    b = svc.analyze_file(str(src), source="import", capture_id="CAP_1")

    assert a.id == b.id                 # same record returned, not a new one
    assert db.count() == 1


def test_reprocessing_same_capture_id_does_not_add_history(tmp_path):
    stubs.stub_no_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "cap.jpg", value=11)
    svc.analyze_file(str(src), source="import", capture_id="CAP_X")
    for _ in range(5):                  # retry/reprocess the same event
        svc.analyze_file(str(src), source="import", capture_id="CAP_X")
    assert db.count() == 1


def test_different_capture_ids_create_separate_records(tmp_path):
    stubs.stub_no_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "cap.jpg", value=12)
    svc.analyze_file(str(src), source="import", capture_id="CAP_A")
    svc.analyze_file(str(src), source="import", capture_id="CAP_B")
    assert db.count() == 2


def test_manual_uploads_without_capture_id_are_not_deduped(tmp_path):
    # capture_id=None (manual) must never collapse into one row.
    stubs.stub_no_crack(inference_core)
    svc, db = _service(tmp_path)
    src = stubs.write_jpg(tmp_path / "cap.jpg", value=13)
    svc.analyze_file(str(src), source="upload")
    svc.analyze_file(str(src), source="upload")
    assert db.count() == 2


# -- DB-level uniqueness ---------------------------------------------

def test_db_rejects_duplicate_capture_id(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    db.add_inspection(timestamp="t", status=STATUS_NO_CRACK, source="import",
                      capture_id="DUP")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_inspection(timestamp="t", status=STATUS_NO_CRACK, source="import",
                          capture_id="DUP")
    assert db.count() == 1
    # Many NULL capture_ids are allowed (manual uploads).
    db.add_inspection(timestamp="t", status=STATUS_NO_CRACK, source="upload")
    db.add_inspection(timestamp="t", status=STATUS_NO_CRACK, source="upload")
    assert db.count() == 3


# -- cross-transport dedup (watch folder + HTTP share the identity) --

@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(appmod, "UPLOAD_TMP_DIR", tmp_path / "upload_tmp")
    monkeypatch.setattr(appmod, "IMPORT_TMP_DIR", tmp_path / "import_tmp")
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "inspections")
    monkeypatch.setattr(appmod, "IMPORT_DIR", tmp_path / "import")
    monkeypatch.setattr(appmod, "IMPORT_LEDGER_PATH", tmp_path / "import_http_state.json")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "inspections.db")
    for attr in ["UPLOAD_TMP_DIR", "IMPORT_TMP_DIR", "INSPECTIONS_DIR", "IMPORT_DIR"]:
        getattr(appmod, attr).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    monkeypatch.setattr(appmod, "_watcher", None)
    monkeypatch.setattr(appmod, "_import_ledger", None)
    monkeypatch.setattr(appmod, "bridge_status", BridgeStatus())
    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c, appmod


def test_same_photo_via_watch_folder_then_http_is_one_record(client):
    c, appmod = client
    stubs.stub_no_crack(inference_core)
    payload = stubs.jpg_bytes(value=200, size=64)

    # Transport A: the watch folder imports the file.
    imp_dir = appmod.IMPORT_DIR
    (imp_dir / "DJI_0001.jpg").write_bytes(payload)
    watcher = PhotoImportWatcher(str(imp_dir), appmod.get_service(), stable_polls=1)
    res = watcher.poll_once()
    assert any(r["action"] == "analyzed" for r in res)
    assert c.get("/api/inspections").get_json()["count"] == 1

    # Transport B: the SAME bytes arrive over HTTP. Same content hash =
    # same capture_id -> the DB guard prevents a second record.
    resp = c.post("/api/import", data=payload, content_type="image/jpeg",
                  headers={"X-Filename": "DJI_0001.jpg"})
    assert resp.status_code in (200, 201)
    assert c.get("/api/inspections").get_json()["count"] == 1
