"""
Tests for the automatic photo-transfer bridge endpoints in app.py:
/api/import (automatic ingest, NOT manual upload), /api/bridge/ping, and
/api/bridge/status. Uses a stubbed YOLO (inference_core.predict_masks).
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from tests import stubs  # noqa: E402


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
        yield c


def test_import_raw_bytes_is_analyzed_as_dji_import(client):
    stubs.stub_one_crack(inference_core)
    resp = client.post(
        "/api/import",
        data=stubs.jpg_bytes(value=100),
        content_type="image/jpeg",
        headers={"X-Filename": "DJI_0001.JPG"},
    )
    assert resp.status_code == 201
    j = resp.get_json()
    assert j["status"] == "CRACK DETECTED"
    assert j["source"] == "import"
    assert j["source_label"] == "DJI import"
    assert j["duplicate"] is False
    # Never leak prohibited model metrics via the automatic path either
    # (latitude/longitude are allowed - they appear in the inspection details).
    for banned in ("confidence", "num_instances", "fps"):
        assert banned not in j


def test_import_multipart_also_works(client):
    stubs.stub_no_crack(inference_core)
    resp = client.post(
        "/api/import",
        data={"image": (io.BytesIO(stubs.jpg_bytes(value=60)), "DJI_0009.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    assert resp.get_json()["status"] == "NO CRACK DETECTED"


def test_duplicate_photo_is_analyzed_only_once_across_retries(client):
    stubs.stub_no_crack(inference_core)
    payload = stubs.jpg_bytes(value=77, size=48)

    first = client.post("/api/import", data=payload, content_type="image/jpeg")
    assert first.status_code == 201
    assert first.get_json()["duplicate"] is False

    # Simulate the bridge retrying the SAME photo (e.g. after a flaky link).
    second = client.post("/api/import", data=payload, content_type="image/jpeg")
    assert second.status_code == 200
    assert second.get_json()["status"] == "duplicate"

    # Only one inspection exists despite two POSTs of identical bytes.
    assert client.get("/api/inspections").get_json()["count"] == 1


def test_import_rejects_non_image(client):
    resp = client.post("/api/import", data=b"this is not an image", content_type="image/jpeg")
    assert resp.status_code == 400
    assert client.get("/api/inspections").get_json()["count"] == 0


def test_import_rejects_empty_body(client):
    resp = client.post("/api/import", data=b"", content_type="image/jpeg")
    assert resp.status_code == 400


def test_bridge_status_offline_then_ready(client):
    # No contact yet -> OFFLINE.
    assert client.get("/api/bridge/status").get_json()["state"] == "OFFLINE"

    # Heartbeat -> online (WAITING FOR PHOTO, no photo yet).
    assert client.post("/api/bridge/ping").get_json()["ok"] is True
    s1 = client.get("/api/bridge/status").get_json()
    assert s1["online"] is True
    assert s1["state"] in ("WAITING FOR PHOTO", "READY")

    # A real photo -> READY with the last result recorded.
    stubs.stub_one_crack(inference_core)
    client.post("/api/import", data=stubs.jpg_bytes(value=33), content_type="image/jpeg")
    s2 = client.get("/api/bridge/status").get_json()
    assert s2["state"] == "READY"
    assert s2["last_result"] == "CRACK DETECTED"
    assert s2["photos_received"] == 1


def test_dashboard_is_clean_and_bridge_status_api_still_works(client):
    # The Live Dashboard is intentionally clean: it focuses on the live POV
    # and the Capture Photo button, and no longer surfaces the photo-bridge
    # badge. The bridge ingest + status API remain available (DJI import path).
    body = client.get("/").data.decode()
    assert "PHOTO BRIDGE" not in body
    assert 'id="capture-btn"' in body
    snap = client.get("/api/bridge/status").get_json()
    assert "state" in snap


def test_import_does_not_touch_manual_upload_source(client):
    # The automatic path must record source 'import', never 'upload'.
    stubs.stub_no_crack(inference_core)
    client.post("/api/import", data=stubs.jpg_bytes(value=42), content_type="image/jpeg")
    latest = client.get("/api/inspection/latest").get_json()
    assert latest["source"] == "import"
