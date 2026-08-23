"""
Tests for phone-GPS (Colota) ingest, configurable inspection radius, the
crack alert UI, and the dashboard cleanup.
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
import telemetry_store as ts  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from telemetry_store import TelemetryStore  # noqa: E402
from tests import stubs  # noqa: E402

_TS = "2026-08-21T07:40:18.420Z"


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


# -- Colota {lat, lon, timestamp} ingest + validation ----------------

def test_telemetry_accepts_colota_lat_lon(client):
    r = client.post("/api/telemetry", json={"lat": 7.123456, "lon": 125.654321, "timestamp": 1704067200})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    latest = client.get("/api/telemetry/latest").get_json()
    assert latest["available"] is True
    assert abs(latest["latest"]["latitude"] - 7.123456) < 1e-6


def test_telemetry_rejects_missing_timestamp(client):
    r = client.post("/api/telemetry", json={"lat": 7.1, "lon": 125.6})
    assert r.status_code == 400
    assert client.get("/api/telemetry/latest").get_json()["available"] is False


def test_telemetry_rejects_out_of_range(client):
    assert client.post("/api/telemetry", json={"lat": 999, "lon": 0, "timestamp": 1704067200}).status_code == 400
    assert client.post("/api/telemetry", json={"lat": 0, "lon": 500, "timestamp": 1704067200}).status_code == 400
    assert client.post("/api/telemetry", json={"lat": "x", "lon": "y", "timestamp": 1}).status_code == 400


# -- configurable radius on capture ---------------------------------

def test_capture_stores_configured_radius(client, monkeypatch):
    monkeypatch.setenv("PHONE_GPS_RADIUS_METERS", "100")
    monkeypatch.setattr(__import__("app"), "_service", None)  # rebuild service with new radius
    stubs.stub_one_crack(inference_core)
    client.post("/api/telemetry", json={"lat": 7.123456, "lon": 125.654321, "timestamp": _TS})
    j = client.post("/api/inspect", data={
        "image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg"), "captured_at": _TS,
    }, content_type="multipart/form-data").get_json()
    assert j["status"] == "CRACK DETECTED"
    # radius is NOT exposed in the operator result payload
    assert "radius_m" not in j
    geo = client.get("/api/inspections/geo").get_json()
    assert geo["count"] == 1
    pt = geo["points"][0]
    assert pt["radius_m"] == 100
    assert pt["gps_available"] is True
    # no prohibited fields leak onto the map either
    assert "accuracy" not in pt and "altitude_m" in pt  # altitude present but null/unused
    assert pt["altitude_m"] is None


def test_capture_without_gps_has_no_radius_and_no_marker(client):
    stubs.stub_no_crack(inference_core)
    j = client.post("/api/inspect", data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
                    content_type="multipart/form-data").get_json()
    assert j["status"] == "NO CRACK DETECTED"       # analysis still runs
    assert client.get("/api/inspections/geo").get_json()["count"] == 0


# -- dashboard cleanup + crack alert --------------------------------

def test_dashboard_cleanup_and_alert(client):
    body = client.get("/").data.decode()
    assert "Live Feed Connection" not in body           # panel removed
    assert "MediaMTX over WebRTC" not in body            # old sentence removed
    assert "No connection" in body                       # minimal offline text
    assert 'id="crack-alert"' in body                    # visible alert element


def test_dashboard_js_has_beep_and_alert():
    js = (Path(__file__).resolve().parent.parent / "static" / "dashboard.js").read_text()
    assert "AudioContext" in js and "function beep" in js
    assert "showCrackAlert" in js


def test_map_has_radius_circle(client):
    body = client.get("/map").data.decode()
    assert "L.circle" in body
    assert "Inspection Radius" in body


# -- radius config + persistence (unit) -----------------------------

def test_phone_gps_radius_env(monkeypatch):
    monkeypatch.delenv("PHONE_GPS_RADIUS_METERS", raising=False)
    assert ts.phone_gps_radius_m() == 50.0
    monkeypatch.setenv("PHONE_GPS_RADIUS_METERS", "200")
    assert ts.phone_gps_radius_m() == 200.0
    monkeypatch.setenv("PHONE_GPS_RADIUS_METERS", "bad")
    assert ts.phone_gps_radius_m() == 50.0


def test_latest_sample_persists_across_restart(tmp_path):
    p = tmp_path / "telemetry_latest.json"
    s1 = TelemetryStore(persist_path=str(p))
    s1.add(7.1, 125.6, timestamp=1704067200, source="phone_gps")
    assert p.exists()
    s2 = TelemetryStore(persist_path=str(p))  # simulate restart
    latest, _ = s2.latest(now_ms=1704067200 * 1000.0 + 100)
    assert latest is not None
    assert abs(latest.latitude - 7.1) < 1e-9 and latest.source == "phone_gps"
