"""
Tests for srt_telemetry.py - DJI SRT parsing (bracketed + GPS() variants),
timestamp/timezone handling, and partial/malformed tolerance. Offline; only
fixture files (no drone).
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import srt_telemetry as srt  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TZ8 = 480  # +08:00 in minutes, for deterministic tests regardless of host TZ


def _expected_ms(y, mo, d, h, mi, s, ms):
    return datetime(y, mo, d, h, mi, s, ms * 1000,
                    tzinfo=timezone(timedelta(minutes=TZ8))).timestamp() * 1000.0


def test_parse_neo2_bracketed_fixture():
    samples = srt.parse_file(str(FIXTURES / "dji_neo2_sample.srt"), tz_offset_min=TZ8)
    assert len(samples) == 3
    assert abs(samples[0].latitude - 7.123455) < 1e-6
    assert abs(samples[0].longitude - 125.654320) < 1e-6
    # timestamps are absolute + ordered
    assert samples[0].timestamp_ms < samples[1].timestamp_ms < samples[2].timestamp_ms
    assert abs(samples[0].timestamp_ms - _expected_ms(2026, 8, 21, 15, 42, 18, 300)) < 2.0


def test_parse_gps_paren_variant():
    # DJI GPS(lon, lat, sats) order.
    text = (
        "1\n00:00:00,000 --> 00:00:00,033\n"
        "2026-08-21 15:42:18\nGPS(125.654320,7.123455,18) BAROMETER:45.6\n"
    )
    samples = srt.parse_srt(text, tz_offset_min=TZ8)
    assert len(samples) == 1
    assert abs(samples[0].latitude - 7.123455) < 1e-6
    assert abs(samples[0].longitude - 125.654320) < 1e-6


def test_parse_loose_variant():
    text = ("1\n00:00:01,000 --> 00:00:01,033\n"
            "2026-08-21 15:42:20\nlatitude: 7.5 longitude: 125.5\n")
    s = srt.parse_srt(text, tz_offset_min=TZ8)
    assert len(s) == 1 and abs(s[0].latitude - 7.5) < 1e-9


def test_malformed_and_partial_blocks_are_skipped():
    text = (
        "1\n00:00:00,000 --> 00:00:00,033\n2026-08-21 15:42:18\n"
        "[latitude: 7.10] [longitude: 125.10]\n\n"
        "2\n00:00:00,033 --> 00:00:00,066\ngarbage with no coordinates\n\n"
        "3\n00:00:00,066 --> 00:00:00,100\n[latitude: 999.0] [longitude: 0.0]\n\n"  # invalid
        "4\n00:00:00,100 -->\n2026-08-21 15:42:18\n[latitude: 7.20"  # half-written
    )
    s = srt.parse_srt(text, tz_offset_min=TZ8)
    # only the first block is a valid, complete GPS sample
    assert len(s) == 1
    assert abs(s[0].latitude - 7.10) < 1e-9


def test_relative_offset_uses_base_time():
    text = ("1\n00:00:02,500 --> 00:00:02,533\n[latitude: 7.1] [longitude: 125.1]\n")
    base = 1_700_000_000_000.0
    s = srt.parse_srt(text, base_time_ms=base)
    assert len(s) == 1
    assert abs(s[0].timestamp_ms - (base + 2500)) < 1.0


def test_empty_and_no_gps():
    assert srt.parse_srt("") == []
    assert srt.parse_srt("1\n00:00:00,000 --> 00:00:00,033\nno coords here\n") == []


def test_timezone_offset_matters():
    text = "1\n00:00:00,000 --> 00:00:00,033\n2026-08-21 15:42:18\n[latitude: 7.1] [longitude: 125.1]\n"
    a = srt.parse_srt(text, tz_offset_min=0)[0].timestamp_ms
    b = srt.parse_srt(text, tz_offset_min=480)[0].timestamp_ms
    assert abs((a - b) - 8 * 3600 * 1000) < 2.0  # +08:00 is 8h earlier in UTC epoch


def test_latest_sample():
    samples = srt.parse_file(str(FIXTURES / "dji_neo2_sample.srt"), tz_offset_min=TZ8)
    assert srt.latest_sample(samples) is samples[-1]


def test_missing_file_is_graceful():
    assert srt.parse_file(str(FIXTURES / "nope.srt")) == []
