"""
Tests for flightrecord_parser.py - offline CSV parsing, timestamp handling,
partial-file tolerance, and honest detection of the encrypted DJI container.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flightrecord_parser as frp  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_parse_csv_fixture_extracts_points():
    res = frp.parse_file(str(FIXTURES / "flight_sample.csv"))
    assert res.decodable is True and res.fmt == "csv"
    assert len(res.points) == 3
    p = res.points[0]
    assert abs(p.latitude - 7.1234) < 1e-6
    assert abs(p.longitude - 125.6543) < 1e-6
    assert p.altitude_m == 40.0
    # latest point is the newest timestamp (the third row)
    assert abs(res.latest.latitude - 7.1235) < 1e-6


def test_parse_csv_flexible_columns_and_partial_rows(tmp_path):
    # DJI-style column names + a garbled/short final row (partial write).
    p = tmp_path / "log.csv"
    p.write_text(
        "OSD.latitude,OSD.longitude,OSD.altitude [m],CUSTOM.updateTime\n"
        "7.10,125.60,30,2026-08-18T15:00:00Z\n"
        "7.11,125.61,31,2026-08-18T15:00:01Z\n"
        "7.12\n"  # partial line (no longitude) -> skipped, not a crash
    )
    res = frp.parse_file(str(p))
    assert res.decodable is True
    assert len(res.points) == 2


def test_parse_csv_relative_time_offsets_use_filename_base(tmp_path):
    # Time column is a relative offset in ms; base comes from the filename.
    name = "DJIFlightRecord_2026-08-18_[15-26-18].csv"
    p = tmp_path / name
    p.write_text("latitude,longitude,time(millisecond)\n7.1,125.6,0\n7.2,125.7,500\n")
    res = frp.parse_file(str(p))
    assert res.decodable and len(res.points) == 2
    base = frp.base_time_ms_from_filename(name)
    assert base is not None
    assert abs(res.points[1].timestamp_ms - (base + 500)) < 1e-6


def test_missing_latlon_columns_is_not_decodable(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("time,temperature\n1,20\n2,21\n")
    res = frp.parse_file(str(p))
    assert res.decodable is False
    assert "latitude" in (res.reason or "")


def test_encrypted_dji_container_detected_not_faked(tmp_path):
    # A binary DJIFlightRecord_*.txt must be reported as not-offline-decodable.
    name = "DJIFlightRecord_2026-08-18_[15-26-18].txt"
    p = tmp_path / name
    p.write_bytes(bytes([0x00, 0x01, 0x02, 0xFF] * 512))
    res = frp.parse_file(str(p))
    assert res.decodable is False
    assert res.fmt == "dji_encrypted"
    assert "offline" in (res.reason or "").lower()
    assert res.points == []


def test_unreadable_file_is_graceful(tmp_path):
    res = frp.parse_file(str(tmp_path / "does_not_exist.csv"))
    assert res.decodable is False
    assert "unreadable" in (res.reason or "")
