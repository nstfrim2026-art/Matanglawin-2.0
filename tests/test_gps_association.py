"""
GPS association robustness (finalization requirements #6-#10).

These cover the specific failure cases called out for the demo:
  * GPS received normally
  * GPS unavailable
  * GPS sample 1 second old   (usable)
  * GPS sample 10 seconds old (usable, <= PHONE_GPS_MAX_AGE_SECONDS)
  * GPS sample stale          (older than the max age -> unusable)

The rule under test: nearest sample when close enough, otherwise the latest
valid fix within the configurable max age, otherwise no fix. Never exact-match
required, never a fabricated (0,0), never a wrong coordinate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import telemetry_store as ts  # noqa: E402
from telemetry_store import TelemetryStore  # noqa: E402

# Epoch-ms base (parse_timestamp_ms treats > 1e11 as already-ms).
_BASE = 1_786_764_378_000.0
_MAX_AGE = 15_000.0  # 15 s, the documented default


def _store():
    return TelemetryStore(max_match_ms=2000, max_age_ms=_MAX_AGE, stale_ms=_MAX_AGE)


# -- #6: GPS received normally / unavailable --------------------------

def test_gps_received_normally():
    s = _store()
    assert s.add(7.123456, 125.654321, timestamp=_BASE, source="phone_gps") is not None
    latest, stale = s.latest(now_ms=_BASE + 500)
    assert latest is not None and stale is False
    assert abs(latest.latitude - 7.123456) < 1e-6


def test_gps_unavailable_is_safe_and_not_fabricated():
    s = _store()
    m = s.match_for_capture(_BASE, use_stale_fallback=True)
    assert m["gps_available"] is False
    assert m["latitude"] is None and m["longitude"] is None
    assert m["gps_source"] is None


# -- #7 / #8: recent samples are usable (not "Not recorded") ---------

def test_gps_sample_1s_old_is_usable():
    s = _store()
    s.add(7.1, 125.6, timestamp=_BASE, source="phone_gps")
    m = s.match_for_capture(_BASE + 1_000, use_stale_fallback=True)
    assert m["gps_available"] is True
    assert abs(m["latitude"] - 7.1) < 1e-9
    assert m["gps_source"] == "phone_gps"


def test_gps_sample_10s_old_is_usable():
    s = _store()
    s.add(7.1, 125.6, timestamp=_BASE, source="phone_gps")
    m = s.match_for_capture(_BASE + 10_000, use_stale_fallback=True)
    assert m["gps_available"] is True
    assert m["gps_match"] == "latest"           # used the fallback, not exact
    assert abs(m["gps_time_delta_ms"] - 10_000.0) < 5.0


# -- #9: excessively stale sample is NOT reused ----------------------

def test_gps_sample_stale_beyond_max_age_is_unavailable():
    s = _store()
    s.add(7.1, 125.6, timestamp=_BASE, source="phone_gps")
    m = s.match_for_capture(_BASE + 20_000, use_stale_fallback=True)   # 20 s > 15 s
    assert m["gps_available"] is False
    assert m["latitude"] is None


def test_nearest_is_preferred_over_stale_fallback():
    s = _store()
    s.add(1.0, 10.0, timestamp=_BASE, source="phone_gps")          # older
    s.add(2.0, 20.0, timestamp=_BASE + 9_800, source="phone_gps")  # near the capture
    m = s.match_for_capture(_BASE + 10_000, use_stale_fallback=True)
    assert m["gps_available"] is True
    assert m["gps_match"] == "nearest"          # 200 ms away -> exact-window match
    assert m["latitude"] == 2.0


def test_future_sample_is_not_used_as_last_known_position():
    # A sample AFTER the capture time is not a "latest known position".
    s = _store()
    s.add(7.1, 125.6, timestamp=_BASE + 60_000, source="phone_gps")
    m = s.match_for_capture(_BASE, use_stale_fallback=True)
    assert m["gps_available"] is False


def test_max_age_is_configurable_via_env(monkeypatch):
    monkeypatch.delenv("PHONE_GPS_MAX_AGE_SECONDS", raising=False)
    assert ts.phone_gps_max_age_ms() == 15_000.0
    monkeypatch.setenv("PHONE_GPS_MAX_AGE_SECONDS", "5")
    assert ts.phone_gps_max_age_ms() == 5_000.0
    monkeypatch.setenv("PHONE_GPS_MAX_AGE_SECONDS", "bad")
    assert ts.phone_gps_max_age_ms() == 15_000.0


def test_srt_backfill_keeps_strict_matching_by_default():
    # The stale fallback is opt-in; without it (SRT backfill), an old sample is
    # never reused for a far-away capture time.
    s = _store()
    s.add(7.1, 125.6, timestamp=_BASE, source="dji_srt")
    strict = s.match_for_capture(_BASE + 10_000)          # use_stale_fallback defaults False
    assert strict["gps_available"] is False
