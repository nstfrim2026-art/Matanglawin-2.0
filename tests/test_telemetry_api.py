"""
Tests for the telemetry + map endpoints in app.py: /api/telemetry ingest,
/api/telemetry/latest, capture->nearest-GPS association into an inspection,
/api/inspections/geo marker generation, missing-GPS behavior, and that the
operator result view still hides coordinates. Segmentation is stubbed.
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
    # operator result view must NOT expose coordinates
    assert "latitude" not in j and "longitude" not in j

    geo = client.get("/api/inspections/geo").get_json()
    assert geo["count"] == 1
    pt = geo["points"][0]
    assert pt["id"] == j["id"]
    assert abs(pt["latitude"] - 7.123456) < 1e-6
    assert pt["gps_available"] is True
    assert pt["gps_time_delta_ms"] is not None
    assert pt["has_crack"] is True
    assert "highlighted" in pt["urls"]


def test_capture_without_gps_still_works_and_has_no_marker(client):
    stubs.stub_no_crack(inference_core)
    j = client.post("/api/inspect", data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
                    content_type="multipart/form-data").get_json()
    assert j["status"] == "NO CRACK DETECTED"       # analysis NOT blocked by missing GPS
    assert client.get("/api/inspections/geo").get_json()["count"] == 0


def test_stale_far_sample_not_attached(client):
    stubs.stub_no_crack(inference_core)
    # telemetry far in the past relative to capture (default match window 2 s)
    client.post("/api/telemetry", json={
        "latitude": 7.1, "longitude": 125.6, "timestamp": "2020-01-01T00:00:00Z"})
    client.post("/api/inspect", data={
        "image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg"), "captured_at": _TS,
    }, content_type="multipart/form-data")
    assert client.get("/api/inspections/geo").get_json()["count"] == 0


# -- map page --------------------------------------------------------

def test_map_page_renders(client):
    resp = client.get("/map")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Inspection Map" in body
    assert "leaflet" in body.lower()
    assert "/api/inspections/geo" in body
