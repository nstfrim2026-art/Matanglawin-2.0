"""
Connection-failure resilience (finalization requirements #4, #17-#19, #21).

A temporary video/network failure must be a NORMAL condition:
  * the live area falls back to a clean "No connection" and auto-reconnects;
  * MediaMTX is never started/stopped/killed by the app;
  * inspection history, GPS records, and map markers all survive a drop;
  * the live view carries NO AI overlay.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_service  # noqa: E402
import inference_core  # noqa: E402
import network_config  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from telemetry_store import TelemetryStore  # noqa: E402
from tests import stubs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    monkeypatch.setattr(appmod, "PICTURES_DIR", tmp_path / "pictures")
    for attr in ["UPLOAD_TMP_DIR", "IMPORT_TMP_DIR", "INSPECTIONS_DIR", "IMPORT_DIR"]:
        getattr(appmod, attr).mkdir(parents=True, exist_ok=True)
    for attr in ["_db", "_service", "_watcher", "_import_ledger", "_capture_service"]:
        monkeypatch.setattr(appmod, attr, None)
    monkeypatch.setattr(appmod, "bridge_status", BridgeStatus())
    monkeypatch.setattr(appmod, "telemetry_store", TelemetryStore())
    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


def _good_grab(url, out):
    stubs.write_jpg(out, value=170)
    return True


# -- #3: viewer disconnect / reconnect --------------------------------

def test_dashboard_js_has_bounded_reconnect_and_clean_offline():
    js = (REPO_ROOT / "static" / "dashboard.js").read_text()
    # bounded/backoff reconnect logic exists
    assert "scheduleReconnect" in js
    assert "reconnectAttempts" in js
    assert "RECONNECT_MAX_MS" in js
    # clean, non-technical offline text (no stack traces)
    assert "No connection" in js
    # internal stream-health monitoring is tracked (debug only)
    assert "streamHealth" in js and "lastFrameTs" in js


def test_stream_status_never_crashes_when_mediamtx_unreachable(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("MediaMTX API down")
    monkeypatch.setattr(network_config, "get_stream_status", boom)
    resp = client.get("/api/stream/status")
    assert resp.status_code == 200
    assert resp.get_json()["pov_state"] == "UNKNOWN"      # clean, not an error page


# -- #4 / MediaMTX lifecycle -----------------------------------------

def test_app_never_starts_or_kills_mediamtx():
    # The app only OBSERVES MediaMTX (dynamic config + status probes); it must
    # never spawn/kill/reconfigure the MediaMTX process. A viewer dying (or a
    # capture failing) therefore cannot take MediaMTX down.
    for name in ("app.py", "network_config.py", "capture_service.py"):
        text = (REPO_ROOT / name).read_text()
        low = text.lower()
        for banned in ("taskkill", "pkill", "mediamtx.exe", "kill(", "terminate("):
            assert banned not in low, f"{name} must not control the MediaMTX process ({banned})"


def test_capture_failure_leaves_mediamtx_status_reachable(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", lambda url, out: False)
    assert client.post("/api/capture").status_code == 503
    # MediaMTX status endpoint still answers cleanly afterwards.
    assert client.get("/api/mediamtx/status").status_code == 200


# -- #17 / #21: inspection history + map markers survive a drop -------

def test_inspection_history_and_map_survive_stream_drop(client, monkeypatch):
    # Create one geolocated inspection.
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    client.post("/api/telemetry", json={"lat": 7.123456, "lon": 125.654321})
    made = client.post("/api/capture").get_json()
    assert made["ok"] is True

    # Now the live stream drops hard.
    monkeypatch.setattr(network_config, "get_stream_status",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    # Stream status degrades cleanly...
    assert client.get("/api/stream/status").get_json()["pov_state"] == "UNKNOWN"
    # ...but inspection history is intact...
    hist = client.get("/api/inspections").get_json()
    assert hist["count"] == 1
    # ...GPS records are intact...
    assert client.get("/api/telemetry/latest").get_json()["available"] is True
    # ...and the map markers are still served.
    geo = client.get("/api/inspections/geo").get_json()
    assert geo["count"] == 1
    assert abs(geo["points"][0]["latitude"] - 7.123456) < 1e-6


# -- #18: video and GPS are independent -------------------------------

def test_gps_continues_when_video_is_down(client, monkeypatch):
    monkeypatch.setattr(network_config, "get_stream_status",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    r = client.post("/api/telemetry", json={"lat": 7.2, "lon": 125.7})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert client.get("/api/telemetry/latest").get_json()["available"] is True


# -- #16: the live view has no AI overlay -----------------------------

def test_live_view_has_no_ai_overlay(client):
    body = client.get("/").data.decode().lower()
    for banned in ("bounding box", "confidence", "segmentation overlay",
                   "crack label", "mask overlay", "iou"):
        assert banned not in body
    # The POV is a raw MediaMTX iframe (no canvas/overlay drawing on it).
    assert '<canvas' not in body


# -- #13: repeat capture does not create duplicate beeps --------------

def test_dashboard_js_beeps_once_per_new_inspection_id():
    js = (REPO_ROOT / "static" / "dashboard.js").read_text()
    # A single once-per-new-id guard drives the beep/alert.
    assert "lastInspectionId" in js
    assert "rec.id !== lastInspectionId" in js
    # beep() is only called inside that guard (not on every poll).
    guard = js.split("rec.id !== lastInspectionId", 1)[1]
    assert "beep()" in guard
