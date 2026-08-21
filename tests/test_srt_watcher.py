"""
Tests for srt_watcher.SrtWatcher - discovery, ingest into telemetry_store,
Mode A (live growth) vs Mode B (post-recording) detection, and backfilling a
pending inspection's coordinates by nearest-timestamp match. Offline; fixtures
only.
"""

import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import srt_watcher as sw  # noqa: E402
from telemetry_store import TelemetryStore  # noqa: E402
from inspection_db import InspectionDB, STATUS_CRACK  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _fixed_tz(monkeypatch):
    # Deterministic timezone for naive SRT datetimes regardless of host TZ.
    monkeypatch.setenv("MATANGLAWIN_SRT_TZ_OFFSET_MIN", "480")


def _watcher(tmp_path, store=None, db=None):
    srt_dir = tmp_path / "srt"
    srt_dir.mkdir(parents=True, exist_ok=True)
    return sw.SrtWatcher([str(srt_dir)], store or TelemetryStore(), db=db), srt_dir


def test_poll_ingests_srt_samples(tmp_path):
    store = TelemetryStore()
    w, srt_dir = _watcher(tmp_path, store=store)
    shutil.copy(FIXTURES / "dji_neo2_sample.srt", srt_dir / "DJI_0001.SRT")

    res = w.poll_once()
    assert res["ingested"] == 3
    assert store.stats()["buffered"] == 3
    latest, _ = store.latest(now_ms=store._latest.timestamp_ms)
    assert abs(latest.latitude - 7.123468) < 1e-6  # last frame


def test_unchanged_file_not_reingested(tmp_path):
    store = TelemetryStore()
    w, srt_dir = _watcher(tmp_path, store=store)
    shutil.copy(FIXTURES / "dji_neo2_sample.srt", srt_dir / "DJI_0001.SRT")
    assert w.poll_once()["ingested"] == 3
    assert w.poll_once()["ingested"] == 0  # nothing new


def test_backfill_updates_pending_inspection(tmp_path):
    store = TelemetryStore()
    db = InspectionDB(str(tmp_path / "insp.db"))
    # An inspection captured (no GPS yet) at 15:42:18.420 +08:00.
    iid = db.add_inspection(
        timestamp="2026-08-21T15:42:18", status=STATUS_CRACK, source="import",
        captured_at="2026-08-21T15:42:18.420+08:00", gps_available=False,
    )
    w, srt_dir = _watcher(tmp_path, store=store, db=db)
    shutil.copy(FIXTURES / "dji_neo2_sample.srt", srt_dir / "DJI_0001.SRT")

    res = w.poll_once()
    assert res["backfilled"] == 1
    rec = db.get_inspection(iid)
    assert rec.gps_available is True
    # nearest frame to .420 is .300 -> lat 7.123455, delta ~120 ms
    assert abs(rec.latitude - 7.123455) < 1e-6
    assert rec.gps_source == "dji_srt"
    assert abs(rec.gps_time_delta_ms - 120.0) < 5.0


def test_no_backfill_when_capture_far_from_srt(tmp_path):
    store = TelemetryStore(max_match_ms=2000)
    db = InspectionDB(str(tmp_path / "insp.db"))
    iid = db.add_inspection(
        timestamp="t", status=STATUS_CRACK, source="import",
        captured_at="2020-01-01T00:00:00+08:00", gps_available=False,
    )
    w, srt_dir = _watcher(tmp_path, store=store, db=db)
    shutil.copy(FIXTURES / "dji_neo2_sample.srt", srt_dir / "DJI_0001.SRT")
    w.poll_once()
    assert db.get_inspection(iid).gps_available is False  # too far in time


def test_mode_post_for_new_complete_file(tmp_path):
    w, srt_dir = _watcher(tmp_path)
    shutil.copy(FIXTURES / "dji_neo2_sample.srt", srt_dir / "DJI_0001.SRT")
    res = w.poll_once()
    assert res["mode"] == sw.MODE_POST  # brand-new complete file = post-recording


def test_mode_live_when_file_grows_recently(tmp_path):
    store = TelemetryStore()
    w, srt_dir = _watcher(tmp_path, store=store)
    f = srt_dir / "DJI_0001.SRT"
    block1 = ("1\n00:00:00,000 --> 00:00:00,033\n2026-08-21 15:42:18.300\n"
              "[latitude: 7.10] [longitude: 125.10]\n")
    f.write_text(block1)
    assert w.poll_once()["ingested"] == 1  # first sighting -> post
    # Append a new block and mark it just-modified -> live growth.
    with open(f, "a") as fh:
        fh.write("\n2\n00:00:00,033 --> 00:00:00,066\n2026-08-21 15:42:18.700\n"
                 "[latitude: 7.11] [longitude: 125.11]\n")
    now = time.time()
    import os
    os.utime(f, (now, now))
    res = w.poll_once(now_ms=now * 1000.0)
    assert res["ingested"] == 1            # only the new sample
    assert res["mode"] == sw.MODE_LIVE


def test_empty_dir_is_safe(tmp_path):
    w, _ = _watcher(tmp_path)
    assert w.poll_once()["ingested"] == 0


def test_status_shape(tmp_path):
    w, _ = _watcher(tmp_path)
    st = w.status()
    assert "mode" in st and "watched_dirs" in st and "samples_ingested" in st
