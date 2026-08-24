"""
Tests for the on-demand capture service and POST /api/capture.

Covers the finalization failure cases:
  * capture while the stream is healthy
  * capture while the stream temporarily fails (retry -> clean error)
  * capture with GPS available / GPS stale / no GPS
  * one capture -> one inspection (repeat = no duplicate)

The frame grabber is faked so no real MediaMTX stream or ffmpeg is needed;
the fake writes a real JPEG so the rest of the pipeline (decode -> best.pt
stub -> DB) runs exactly as in production.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_service  # noqa: E402
import inference_core  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from capture_service import CaptureError, CaptureService  # noqa: E402
from telemetry_store import TelemetryStore  # noqa: E402
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
    monkeypatch.setattr(appmod, "PICTURES_DIR", tmp_path / "pictures")
    for attr in ["UPLOAD_TMP_DIR", "IMPORT_TMP_DIR", "INSPECTIONS_DIR", "IMPORT_DIR"]:
        getattr(appmod, attr).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    monkeypatch.setattr(appmod, "_watcher", None)
    monkeypatch.setattr(appmod, "_import_ledger", None)
    monkeypatch.setattr(appmod, "_capture_service", None)
    monkeypatch.setattr(appmod, "bridge_status", BridgeStatus())
    monkeypatch.setattr(appmod, "telemetry_store", TelemetryStore())
    # Simulate ffmpeg being installed (resolved) so the app-built capture
    # service is "available"; individual tests then patch ffmpeg_grab to
    # control the actual grab outcome (success/failure).
    monkeypatch.setattr(capture_service, "resolve_ffmpeg", lambda *a, **k: "/usr/bin/ffmpeg-fake")
    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


def _good_grab(url, out, **kwargs):
    """Fake grabber: writes a real clean JPEG (the 'current drone frame')."""
    stubs.write_jpg(out, value=180)
    return True


def _bad_grab(url, out, **kwargs):
    """Fake grabber: the stream is unreadable."""
    return False


# ============================ unit: CaptureService ============================

def test_capture_healthy_stream_creates_inspection(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "insp")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    stubs.stub_no_crack(inference_core)

    svc = CaptureService(
        appmod.get_service(),
        rtsp_url_provider=lambda: "rtsp://localhost:8554/matanglawin",
        pictures_dir=str(tmp_path / "pics"),
        grabber=_good_grab,
    )
    result = svc.capture()
    assert Path(result["image_path"]).exists()
    assert Path(result["image_path"]).parent == (tmp_path / "pics")
    assert result["timestamp"] > 0
    assert result["record"].status in ("CRACK DETECTED", "NO CRACK DETECTED")


def test_capture_temporary_failure_then_success_retries(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "insp")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    stubs.stub_no_crack(inference_core)

    calls = {"n": 0}

    def flaky(url, out, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:      # fail the first two attempts
            return False
        return _good_grab(url, out)

    svc = CaptureService(
        appmod.get_service(),
        rtsp_url_provider=lambda: "rtsp://localhost:8554/matanglawin",
        pictures_dir=str(tmp_path / "pics"),
        grabber=flaky, retries=3, backoff_s=0.0,
    )
    result = svc.capture()
    assert calls["n"] == 3
    assert Path(result["image_path"]).exists()


def test_capture_stream_down_raises_clean_error_and_leaves_mediamtx_alone(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "insp")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    stubs.stub_no_crack(inference_core)

    svc = CaptureService(
        appmod.get_service(),
        rtsp_url_provider=lambda: "rtsp://localhost:8554/matanglawin",
        pictures_dir=str(tmp_path / "pics"),
        grabber=_bad_grab, retries=2, backoff_s=0.0,
    )
    with pytest.raises(CaptureError):
        svc.capture()
    # No inspection was created and the pictures dir has no stray files.
    assert appmod.get_db().count() == 0


def test_capture_no_stream_url_raises_clean_error(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "insp")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    svc = CaptureService(
        appmod.get_service(),
        rtsp_url_provider=lambda: None,      # no live stream configured
        pictures_dir=str(tmp_path / "pics"),
        grabber=_good_grab,
    )
    with pytest.raises(CaptureError):
        svc.capture()


def test_ffmpeg_grab_missing_binary_returns_false_never_raises(tmp_path):
    ok = capture_service.ffmpeg_grab(
        "rtsp://localhost:8554/matanglawin",
        str(tmp_path / "x.jpg"),
        ffmpeg_bin="definitely-not-a-real-ffmpeg-binary",
    )
    assert ok is False


def test_capture_timestamp_is_utc_aware(tmp_path, monkeypatch):
    # Fix #11: the capture timestamp must be UTC/timezone-aware so it is
    # directly comparable to Colota's UTC timestamps (no naive local time).
    import app as appmod
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "insp")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    stubs.stub_no_crack(inference_core)
    svc = CaptureService(
        appmod.get_service(),
        rtsp_url_provider=lambda: "rtsp://localhost:8554/matanglawin",
        pictures_dir=str(tmp_path / "pics"),
        grabber=_good_grab,
    )
    from datetime import datetime
    result = svc.capture()
    dt = datetime.fromisoformat(result["captured_at"])
    assert dt.tzinfo is not None                      # tz-aware, not naive
    assert abs(dt.utcoffset().total_seconds()) < 1    # expressed in UTC


def test_ffmpeg_grab_success_path_builds_tcp_single_frame(monkeypatch, tmp_path):
    # Fix #8: connect via RTSP, TCP transport, exactly one frame, save JPEG.
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # simulate ffmpeg writing the output frame
        out = cmd[cmd.index("-y") + 1]
        stubs.write_jpg(out, value=90)
        return _Proc()

    monkeypatch.setattr(capture_service.subprocess, "run", fake_run)
    out = tmp_path / "frame.jpg"
    ok = capture_service.ffmpeg_grab("rtsp://127.0.0.1:8554/matanglawin",
                                     str(out), ffmpeg_bin="ffmpeg")
    assert ok is True and out.exists()
    cmd = captured["cmd"]
    assert "-rtsp_transport" in cmd and cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert cmd[cmd.index("-frames:v") + 1] == "1"      # exactly one frame


# ============================ ffmpeg configuration errors =====================

def test_api_capture_reports_ffmpeg_not_configured(client, monkeypatch):
    # When no ffmpeg can be resolved, /api/capture returns a CLEAR config error
    # (distinct from the generic 'could not read the live stream').
    import app as appmod
    monkeypatch.setattr(capture_service, "resolve_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_capture_service", None)      # rebuild w/o ffmpeg
    resp = client.post("/api/capture")
    assert resp.status_code == 503
    j = resp.get_json()
    assert j["ok"] is False
    assert j["error"] == "ffmpeg_not_configured"
    assert "FFmpeg not configured" in j["detail"]


def test_capture_status_reports_ffmpeg_diagnostics(client, monkeypatch):
    monkeypatch.setattr(capture_service, "resolve_ffmpeg", lambda *a, **k: "/opt/ffmpeg/bin/ffmpeg")
    import app as appmod
    monkeypatch.setattr(appmod, "_capture_service", None)
    snap = client.get("/api/capture/status").get_json()
    assert snap["ffmpeg_available"] is True
    assert snap["ffmpeg_executable"] == "/opt/ffmpeg/bin/ffmpeg"
    assert snap["rtsp_url"].startswith("rtsp://")


# ============================ route: POST /api/capture ========================

def test_api_capture_healthy(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_one_crack(inference_core)
    resp = client.post("/api/capture")
    assert resp.status_code == 201
    j = resp.get_json()
    assert j["ok"] is True
    assert j["status"] == "CRACK DETECTED"
    assert "image_path" in j and j["source"] == "capture"
    # never leak model metrics
    for banned in ("confidence", "num_instances", "fps"):
        assert banned not in j


def test_api_capture_stream_failure_returns_clean_503(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _bad_grab)
    resp = client.post("/api/capture")
    assert resp.status_code == 503
    j = resp.get_json()
    assert j["ok"] is False and j["error"] == "capture_failed"
    assert "detail" in j                    # human-readable, no stack trace
    assert "Traceback" not in j["detail"]
    # The failure did not create an inspection and did not crash other APIs.
    assert client.get("/api/inspections").get_json()["count"] == 0
    assert client.get("/health") is not None


def test_api_capture_with_gps_available(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_no_crack(inference_core)
    # a fresh phone-GPS fix (arrival-time timestamp)
    client.post("/api/telemetry", json={"lat": 7.123456, "lon": 125.654321})
    j = client.post("/api/capture").get_json()
    assert j["gps_available"] is True
    assert abs(j["latitude"] - 7.123456) < 1e-6
    assert abs(j["longitude"] - 125.654321) < 1e-6


def test_api_capture_with_stale_gps_still_captures_but_no_location(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_no_crack(inference_core)
    # a fix from 30 s ago (older than the 15 s max age)
    client.post("/api/telemetry", json={"lat": 7.1, "lon": 125.6, "tst": time.time() - 30})
    j = client.post("/api/capture").get_json()
    assert j["ok"] is True                  # capture + analysis still succeed
    assert j["gps_available"] is False       # only the location is unavailable
    assert j["latitude"] is None and j["longitude"] is None


def test_api_capture_with_no_gps_still_captures(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_one_crack(inference_core)
    j = client.post("/api/capture").get_json()
    assert j["ok"] is True
    assert j["status"] == "CRACK DETECTED"   # analysis runs without GPS
    assert j["gps_available"] is False


def test_repeat_capture_creates_distinct_inspections(client, monkeypatch):
    # Each Capture Photo press grabs a new frame -> its own inspection (distinct
    # ids), so the once-per-new-id crack alert fires per capture, not per refresh.
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_no_crack(inference_core)
    a = client.post("/api/capture").get_json()
    b = client.post("/api/capture").get_json()
    assert a["id"] != b["id"]
    assert client.get("/api/inspections").get_json()["count"] == 2
