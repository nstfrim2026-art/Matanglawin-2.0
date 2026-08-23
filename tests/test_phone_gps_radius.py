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

def test_telemetry_accepts_full_colota_payload(client):
    # The exact real Colota Google Play payload (tst + extra fields).
    r = client.post("/api/telemetry", json={
        "lat": 7.123456, "lon": 125.654321, "acc": 8, "alt": 42, "vel": 0,
        "batt": 85, "bs": 1, "tst": 1704067200, "bear": 180,
    })
    assert r.status_code == 200 and r.get_json()["ok"] is True
    latest = client.get("/api/telemetry/latest").get_json()
    # simple debug view returns ONLY lat/lon/timestamp
    assert abs(latest["lat"] - 7.123456) < 1e-6
    assert abs(latest["lon"] - 125.654321) < 1e-6
    assert latest["timestamp"] == 1704067200
    # extras are NOT stored/exposed
    assert "acc" not in latest and "accuracy" not in latest and "battery" not in latest
    assert "bear" not in latest and "vel" not in latest
    assert latest["latest"]["altitude_m"] is None   # alt=42 ignored


def test_telemetry_accepts_colota_lat_lon(client):
    r = client.post("/api/telemetry", json={"lat": 7.123456, "lon": 125.654321, "tst": 1704067200})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    latest = client.get("/api/telemetry/latest").get_json()
    assert latest["available"] is True
    assert abs(latest["latest"]["latitude"] - 7.123456) < 1e-6


def test_telemetry_accepts_get_query_params(client):
    r = client.get("/api/telemetry?lat=7.10&lon=125.10&tst=1704067200")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert abs(client.get("/api/telemetry/latest").get_json()["lat"] - 7.10) < 1e-6


def test_telemetry_missing_timestamp_is_accepted_using_now(client):
    # Colota always sends tst, but a missing timestamp must NOT be a 400 -
    # only missing/invalid LOCATION is rejected. Arrival time is used.
    r = client.post("/api/telemetry", json={"lat": 7.1, "lon": 125.6})
    assert r.status_code == 200
    assert client.get("/api/telemetry/latest").get_json()["available"] is True


def test_telemetry_rejects_out_of_range(client):
    assert client.post("/api/telemetry", json={"lat": 999, "lon": 0, "tst": 1704067200}).status_code == 400
    assert client.post("/api/telemetry", json={"lat": 0, "lon": 500, "tst": 1704067200}).status_code == 400
    assert client.post("/api/telemetry", json={"lat": "x", "lon": "y", "tst": 1}).status_code == 400
    # (0,0) null-island is treated as no fix
    assert client.post("/api/telemetry", json={"lat": 0, "lon": 0, "tst": 1704067200}).status_code == 400


# -- configurable radius on capture ---------------------------------

def test_automatic_capture_with_gps_surfaces_coords_and_stores_radius(client, monkeypatch):
    import app as appmod
    monkeypatch.setenv("PHONE_GPS_RADIUS_METERS", "100")
    monkeypatch.setattr(appmod, "_service", None)  # rebuild service with new radius
    stubs.stub_one_crack(inference_core)
    client.post("/api/telemetry", json={"lat": 7.123456, "lon": 125.654321, "timestamp": _TS})
    # AUTOMATIC capture (DJI import) is the path that receives GPS.
    j = client.post("/api/import", data=stubs.jpg_bytes(value=201),
                    content_type="image/jpeg",
                    headers={"X-Filename": "DJI_1.jpg", "X-Captured-At": _TS}).get_json()
    assert j["status"] == "CRACK DETECTED"
    # capture coordinates are surfaced in the inspection details
    assert j["gps_available"] is True
    assert abs(j["latitude"] - 7.123456) < 1e-6 and abs(j["longitude"] - 125.654321) < 1e-6
    # radius is stored on the record (metadata) while accuracy/altitude are never exposed
    assert "radius_m" not in j and "accuracy" not in j and "altitude_m" not in j
    rec = appmod.get_db().get_inspection(j["id"])
    assert rec.radius_m == 100
    assert rec.altitude_m is None


def test_automatic_capture_without_gps_has_no_coords_and_no_radius(client):
    import app as appmod
    stubs.stub_no_crack(inference_core)
    j = client.post("/api/import", data=stubs.jpg_bytes(value=202),
                    content_type="image/jpeg",
                    headers={"X-Filename": "DJI_2.jpg"}).get_json()
    assert j["status"] == "NO CRACK DETECTED"       # analysis still runs
    assert j["gps_available"] is False
    assert j["latitude"] is None and j["longitude"] is None
    rec = appmod.get_db().get_inspection(j["id"])
    assert rec.radius_m is None


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
