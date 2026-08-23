"""
Tests for the telemetry endpoints in app.py: /api/telemetry ingest,
/api/telemetry/latest, capture->nearest-GPS association into an inspection,
missing-GPS behavior, and that the capture coordinates are now surfaced in
the inspection details (payload + result page). The map has been removed.
Segmentation is stubbed.
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from telemetry_store import TelemetryStore  # noqa: E402
from tests import stubs  # noqa: E402

_TS = "2026-08-18T07:26:18.420Z"  # capture + telemetry share this instant


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
    monkeypatch.setattr(appmod, "telemetry_store", TelemetryStore())

    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


# -- ingest -----------------------------------------------------------

def test_post_telemetry_valid(client):
    r = client.post("/api/telemetry", json={
        "latitude": 7.123456, "longitude": 125.654321, "altitude_m": 43.2,
        "timestamp": _TS, "source": "dji_flight_record"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    latest = client.get("/api/telemetry/latest").get_json()
    assert latest["available"] is True
    assert abs(latest["latest"]["latitude"] - 7.123456) < 1e-6


def test_post_telemetry_invalid_rejected(client):
    r = client.post("/api/telemetry", json={"latitude": 999, "longitude": 0})
    assert r.status_code == 400 and r.get_json()["ok"] is False
    assert client.get("/api/telemetry/latest").get_json()["available"] is False


# -- capture -> nearest-GPS association ------------------------------

def test_capture_is_geotagged_with_nearest_sample(client):
    stubs.stub_one_crack(inference_core)
    # aircraft sample at the capture instant
    client.post("/api/telemetry", json={
        "latitude": 7.123456, "longitude": 125.654321, "altitude_m": 43.2, "timestamp": _TS})
    # capture with the same timestamp
    j = client.post("/api/inspect", data={
        "image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg"),
        "captured_at": _TS,
    }, content_type="multipart/form-data").get_json()
    assert j["status"] == "CRACK DETECTED"
    # capture coordinates are now surfaced in the inspection details payload
    assert j["gps_available"] is True
    assert abs(j["latitude"] - 7.123456) < 1e-6
    assert abs(j["longitude"] - 125.654321) < 1e-6

    # ...and rendered on the inspection result page ("the description itself").
    body = client.get(f"/inspection/{j['id']}").data.decode()
    assert "GPS location" in body
    assert "7.123456, 125.654321" in body


def test_capture_without_gps_still_works_and_has_no_coords(client):
    stubs.stub_no_crack(inference_core)
    j = client.post("/api/inspect", data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
                    content_type="multipart/form-data").get_json()
    assert j["status"] == "NO CRACK DETECTED"       # analysis NOT blocked by missing GPS
    assert j["gps_available"] is False
    assert j["latitude"] is None and j["longitude"] is None
    # the result page shows a graceful placeholder, never a blank/broken value
    body = client.get(f"/inspection/{j['id']}").data.decode()
    assert "GPS location" in body and "Not recorded" in body


def test_stale_far_sample_not_attached(client):
    stubs.stub_no_crack(inference_core)
    # telemetry far in the past relative to capture (default match window 2 s)
    client.post("/api/telemetry", json={
        "latitude": 7.1, "longitude": 125.6, "timestamp": "2020-01-01T00:00:00Z"})
    j = client.post("/api/inspect", data={
        "image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg"), "captured_at": _TS,
    }, content_type="multipart/form-data").get_json()
    assert j["gps_available"] is False
    assert j["latitude"] is None and j["longitude"] is None


# -- map removed -----------------------------------------------------

def test_map_routes_are_gone(client):
    # The map feature (page, tiles, geo API) was removed entirely.
    assert client.get("/map").status_code == 404
    assert client.get("/api/inspections/geo").status_code == 404
    assert client.get("/maps/5/10/12.png").status_code == 404
