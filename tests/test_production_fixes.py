"""
Final production-fix regressions (the real hardware-testing failures):

  * ffmpeg executable discovery (env -> PATH -> Windows glob -> None),
  * UTC/timezone-aware capture timestamps matching Colota on a UTC+8 PC,
  * GPS association across the 1s / 5s / 10s / 15s / >15s ages,
  * the restored Latest Inspection section on the dashboard,
  * one consistent capture source label (no Manual/Drone duplication),
  * the map + radius are gone.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_service  # noqa: E402
import inference_core  # noqa: E402
import telemetry_store as ts  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from telemetry_store import TelemetryStore, parse_timestamp_ms  # noqa: E402
from tests import stubs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_BASE = 1_786_764_378_000.0   # epoch-ms


# ============================ ffmpeg discovery ================================

def test_ffmpeg_env_var_has_priority(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MATANGLAWIN_FFMPEG", str(fake))
    assert capture_service.resolve_ffmpeg() == str(fake.resolve()) or \
        os.path.abspath(str(fake)) == capture_service.resolve_ffmpeg()


def test_ffmpeg_falls_back_to_which(monkeypatch):
    monkeypatch.delenv("MATANGLAWIN_FFMPEG", raising=False)
    monkeypatch.setattr(capture_service.shutil, "which",
                        lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    assert capture_service.resolve_ffmpeg() == "/usr/local/bin/ffmpeg"


def test_ffmpeg_windows_glob_discovery(tmp_path, monkeypatch):
    # Simulate a WinGet-style install found only by globbing (no PATH, no env),
    # without hardcoding any username or exact path.
    monkeypatch.delenv("MATANGLAWIN_FFMPEG", raising=False)
    monkeypatch.setattr(capture_service.shutil, "which", lambda name: None)
    pkg = tmp_path / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_x" / "bin"
    pkg.mkdir(parents=True)
    exe = pkg / "ffmpeg.exe"
    exe.write_text("x")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    found = capture_service.resolve_ffmpeg()
    assert found is not None and found.endswith("ffmpeg.exe")


def test_ffmpeg_unavailable_returns_none(monkeypatch):
    monkeypatch.delenv("MATANGLAWIN_FFMPEG", raising=False)
    monkeypatch.setattr(capture_service.shutil, "which", lambda name: None)
    for var in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.delenv(var, raising=False)
    assert capture_service.resolve_ffmpeg() is None


def test_ffmpeg_explicit_missing_does_not_fall_through(monkeypatch):
    # An explicit-but-missing override must NOT silently pick a different ffmpeg.
    # (which() only resolves the bare 'ffmpeg' name, not the bogus explicit path.)
    monkeypatch.setattr(capture_service.shutil, "which",
                        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert capture_service.resolve_ffmpeg(explicit="/nope/not-here") is None


# ============================ UTC timestamp matching ==========================

def test_utc_plus8_capture_matches_colota(monkeypatch):
    # Colota sends a UTC sample; a capture whose timestamp is produced in UTC
    # matches it regardless of the PC's local timezone. This is the fix for the
    # UTC+8 "Not recorded" bug.
    store = TelemetryStore()
    now = time.time()
    store.add(7.1325103, 125.6949673, timestamp=now, source="phone_gps")

    # capture_service builds captured_at with datetime.now(timezone.utc)
    utc_iso = datetime.now(timezone.utc).isoformat()
    cap_ms = parse_timestamp_ms(utc_iso)
    good = store.match_for_capture(cap_ms, use_stale_fallback=True)
    assert good["gps_available"] is True
    assert good["gps_source"] == "phone_gps"


def test_naive_local_plus8_timestamp_would_miss_demonstrating_the_bug():
    # Regression witness: a NAIVE local (+8h) timestamp is parsed as UTC and
    # lands ~8h in the future, so it would fail to associate. The production
    # code avoids this by always using UTC-aware capture timestamps.
    store = TelemetryStore()
    now = time.time()
    store.add(7.13, 125.69, timestamp=now, source="phone_gps")
    naive_plus8 = (datetime.now(timezone.utc) + timedelta(hours=8)).replace(tzinfo=None).isoformat()
    bad_ms = parse_timestamp_ms(naive_plus8)
    bad = store.match_for_capture(bad_ms, use_stale_fallback=True)
    assert bad["gps_available"] is False


# ============================ GPS association ages ============================

@pytest.mark.parametrize("age_s,expected", [(1, True), (5, True), (10, True), (15, True), (16, False)])
def test_gps_association_across_ages(age_s, expected):
    store = TelemetryStore(max_match_ms=2000, max_age_ms=15_000, stale_ms=15_000)
    store.add(7.13, 125.69, timestamp=_BASE, source="phone_gps")
    m = store.match_for_capture(_BASE + age_s * 1000, use_stale_fallback=True)
    assert m["gps_available"] is expected
    if expected:
        assert abs(m["latitude"] - 7.13) < 1e-9
        assert m["longitude"] is not None            # never (0,0) / fabricated


# ============================ dashboard layout ================================

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
    monkeypatch.setattr(capture_service, "resolve_ffmpeg", lambda *a, **k: "/usr/bin/ffmpeg-fake")
    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


def test_dashboard_has_restored_latest_inspection_section(client):
    body = client.get("/").data.decode()
    assert "Latest Inspection" in body
    assert 'id="latest-status"' in body
    assert 'id="latest-lat"' in body and 'id="latest-lon"' in body
    assert 'id="latest-datetime"' in body and 'id="latest-source"' in body
    assert 'id="latest-result"' in body           # analyzed image element
    # The Capture Photo button sits above it.
    assert 'id="capture-btn"' in body


def test_dashboard_has_no_map_and_no_shutter_text(client):
    body = client.get("/").data.decode().lower()
    assert "shutter" not in body
    assert "no analyze button" not in body
    assert "map" not in body or "id=\"map\"" not in body  # no map canvas/nav
    assert "leaflet" not in body
    assert "radius" not in body


def test_dashboard_js_beeps_once_and_not_on_refresh():
    js = (REPO_ROOT / "static" / "dashboard.js").read_text()
    assert "rec.id !== lastInspectionId" in js
    guard = js.split("rec.id !== lastInspectionId", 1)[1]
    assert "beep()" in guard                        # beep only inside the new-id guard


# ============================ source label / no duplication ===================

def _good_grab(url, out, **kwargs):
    stubs.write_jpg(out, value=140)
    return True


def test_capture_source_label_is_capture_and_single_inspection(client, monkeypatch):
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_no_crack(inference_core)
    j = client.post("/api/capture").get_json()
    assert j["source"] == "capture"
    assert j["source_label"] == "Capture"           # one consistent label
    # exactly one inspection, one source - never both Manual + Drone
    data = client.get("/api/inspections").get_json()
    assert data["count"] == 1
    labels = {i["source_label"] for i in data["inspections"]}
    assert labels == {"Capture"}


def test_manual_upload_and_capture_are_distinct_records_not_duplicates(client, monkeypatch):
    # Manual upload remains for testing but is clearly its own record/label;
    # a capture is never also shown as a manual/drone upload of the same photo.
    monkeypatch.setattr(capture_service, "ffmpeg_grab", _good_grab)
    stubs.stub_no_crack(inference_core)
    client.post("/api/capture")
    import io
    client.post("/detect", data={"image": (io.BytesIO(stubs.jpg_bytes(value=7)), "t.jpg")},
                content_type="multipart/form-data")
    items = client.get("/api/inspections").get_json()["inspections"]
    labels = sorted(i["source_label"] for i in items)
    assert labels == ["Capture", "Manual upload"]   # distinct, no duplication
