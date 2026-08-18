"""
Tests for telemetry_store.py - timestamp parsing, validation, nearest-in-time
matching, staleness, and capture association quality flags.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_store import TelemetryStore, parse_timestamp_ms  # noqa: E402


def test_parse_timestamp_iso_with_offset():
    # 15:26:18.420 +08:00 is the same instant as 07:26:18.420 UTC.
    a = parse_timestamp_ms("2026-08-18T15:26:18.420+08:00")
    b = parse_timestamp_ms("2026-08-18T07:26:18.420Z")
    assert a is not None and abs(a - b) < 1.0
    # and the fractional second is preserved (…​.420)
    assert abs((a % 1000) - 420.0) < 1.0


def test_parse_timestamp_z_and_epoch():
    assert parse_timestamp_ms("2026-08-18T07:26:18.420Z") is not None
    assert parse_timestamp_ms(1786764378.42) == 1786764378420.0     # seconds -> ms
    assert parse_timestamp_ms(1786764378420) == 1786764378420.0     # already ms
    assert parse_timestamp_ms("not-a-time") is None
    assert parse_timestamp_ms(None) is None


def test_add_rejects_bad_coordinates():
    s = TelemetryStore()
    assert s.add(999, 0, timestamp=1000) is None       # lat out of range
    assert s.add(0, 0, timestamp=1000) is None          # null island
    assert s.add("x", "y", timestamp=1000) is None      # non-numeric
    assert s.stats()["rejected"] == 3
    assert s.add(7.1, 125.6, timestamp=1000) is not None


# Realistic epoch-ms base (parse_timestamp_ms treats >1e11 as already-ms).
_BASE = 1_786_764_378_000.0


def test_nearest_sample_matching_picks_closest_in_time():
    s = TelemetryStore()
    base = _BASE
    s.add(1.0, 10.0, timestamp=base + 0)      # A
    s.add(2.0, 20.0, timestamp=base + 400)    # B
    s.add(3.0, 30.0, timestamp=base + 800)    # C
    # capture 520 ms -> nearest is B (@400, delta 120) not C (@800, delta 280)
    sample, delta = s.nearest(base + 520)
    assert sample.latitude == 2.0
    assert abs(delta - 120.0) < 1e-6


def test_match_for_capture_quality_flags():
    s = TelemetryStore(max_match_ms=200)
    base = _BASE
    s.add(2.0, 20.0, timestamp=base + 400, altitude_m=43.2, source="dji_flight_record")
    # within window
    m = s.match_for_capture(base + 520)
    assert m["gps_available"] is True
    assert m["latitude"] == 2.0 and m["altitude_m"] == 43.2
    assert m["gps_source"] == "dji_flight_record"
    assert abs(m["gps_time_delta_ms"] - 120.0) < 1e-6
    # outside window -> no fix (but never raises)
    m2 = s.match_for_capture(base + 5000)
    assert m2["gps_available"] is False and m2["latitude"] is None


def test_latest_and_staleness():
    s = TelemetryStore(stale_ms=1000)
    s.add(1.0, 10.0, timestamp=_BASE)
    latest, stale = s.latest(now_ms=_BASE + 500)
    assert latest.latitude == 1.0 and stale is False
    latest, stale = s.latest(now_ms=_BASE + 9000)
    assert stale is True


def test_empty_store_match_is_safe():
    s = TelemetryStore()
    m = s.match_for_capture(123456.0)
    assert m["gps_available"] is False
    latest, stale = s.latest()
    assert latest is None and stale is True
