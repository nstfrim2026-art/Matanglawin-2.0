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


def _import(client, value, captured_at=None, filename="DJI_x.jpg"):
    """POST one automatic-import photo (unique bytes per `value`)."""
    headers = {"X-Filename": filename}
    if captured_at:
        headers["X-Captured-At"] = captured_at
    return client.post("/api/import", data=stubs.jpg_bytes(value=value),
                       content_type="image/jpeg", headers=headers)


# -- AUTOMATIC capture -> nearest fresh-GPS association --------------

def test_automatic_capture_is_geotagged_with_fresh_sample(client):
    stubs.stub_one_crack(inference_core)
    # aircraft/phone sample at the capture instant
    client.post("/api/telemetry", json={
        "latitude": 7.123456, "longitude": 125.654321, "altitude_m": 43.2, "timestamp": _TS})
    # automatic capture (DJI import) with the same timestamp -> fresh match
    j = _import(client, value=101, captured_at=_TS).get_json()
    assert j["status"] == "CRACK DETECTED"
    assert j["source"] == "import"
    assert j["gps_available"] is True
    assert abs(j["latitude"] - 7.123456) < 1e-6
    assert abs(j["longitude"] - 125.654321) < 1e-6

    # ...and rendered on the inspection result page in the labeled readout.
    body = client.get(f"/inspection/{j['id']}").data.decode()
    assert "Latitude" in body and "Longitude" in body
    assert "7.123456&deg; N" in body
    assert "125.654321&deg; E" in body


def test_automatic_capture_without_fresh_gps_has_none(client):
    stubs.stub_no_crack(inference_core)
    # analysis still runs with no telemetry at all
    j = _import(client, value=102).get_json()
    assert j["status"] == "NO CRACK DETECTED"       # analysis NOT blocked by missing GPS
    assert j["gps_available"] is False
    assert j["latitude"] is None and j["longitude"] is None
    body = client.get(f"/inspection/{j['id']}").data.decode()
    assert "Latitude" in body and "Not recorded" in body


def test_stale_gps_is_not_reused_after_colota_stops(client):
    """Colota sent a fix, then stopped: a later automatic capture must NOT
    reuse that old position (freshness window in match_for_capture)."""
    import app as appmod
    stubs.stub_no_crack(inference_core)
    # A fix arrives while Colota is on.
    client.post("/api/telemetry", json={
        "latitude": 7.1, "longitude": 125.6, "timestamp": "2020-01-01T00:00:00Z"})
    assert client.get("/api/telemetry/latest").get_json()["available"] is True
    # Colota is now OFF; a NEW capture happens much later (2026). The last
    # known 2020 fix is far outside the match window -> not reused.
    j = _import(client, value=103, captured_at=_TS).get_json()
    assert j["gps_available"] is False
    assert j["latitude"] is None and j["longitude"] is None
    rec = appmod.get_db().get_inspection(j["id"])
    assert rec.gps_source is None


# -- MANUAL upload is independent of the phone GPS -------------------

def test_manual_upload_never_gets_gps_even_with_active_colota(client):
    """The critical fix: a manual upload must NOT inherit the current/latest
    Colota position, even when a fresh sample exists at the same instant."""
    import app as appmod
    stubs.stub_one_crack(inference_core)
    # Colota is actively sending a valid fix right now.
    client.post("/api/telemetry", json={
        "latitude": 7.123456, "longitude": 125.654321, "timestamp": _TS})
    # Manual upload with a captured_at that exactly matches the fresh sample.
    j = client.post("/api/inspect", data={
        "image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg"), "captured_at": _TS,
    }, content_type="multipart/form-data").get_json()
    assert j["status"] == "CRACK DETECTED"          # analysis still works
    assert j["source"] == "upload"
    # No GPS whatsoever on a manual upload.
    assert j["gps_available"] is False
    assert j["latitude"] is None and j["longitude"] is None
    rec = appmod.get_db().get_inspection(j["id"])
    assert rec.gps_source is None
    assert rec.captured_at is None                  # not eligible for SRT backfill
    body = client.get(f"/inspection/{j['id']}").data.decode()
    assert "Not recorded" in body


# -- map removed entirely --------------------------------------------

def test_map_routes_are_gone(client):
    # The map feature (page, tiles, geo API) has been removed for production.
    assert client.get("/map").status_code == 404
    assert client.get("/api/inspections/geo").status_code == 404
    assert client.get("/maps/5/10/12.png").status_code == 404
