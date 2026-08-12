"""
Tests for gps_provider.py - GPS is always optional, never fabricated.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gps_provider  # noqa: E402


@pytest.fixture(autouse=True)
def reset_gps_source():
    gps_provider.set_gps_source(None)
    yield
    gps_provider.set_gps_source(None)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ["MATANGLAWIN_GPS_LAT", "MATANGLAWIN_GPS_LON", "MATANGLAWIN_GPS_ALT"]:
        monkeypatch.delenv(key, raising=False)
    yield


def test_no_source_and_no_env_is_unavailable():
    fix = gps_provider.get_current_fix()
    assert fix.available is False
    assert fix.latitude is None
    assert fix.longitude is None


def test_env_override_provides_a_fix(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_GPS_LAT", "14.5995")
    monkeypatch.setenv("MATANGLAWIN_GPS_LON", "120.9842")
    fix = gps_provider.get_current_fix()
    assert fix.available is True
    assert fix.latitude == 14.5995
    assert fix.longitude == 120.9842
    assert fix.altitude is None


def test_env_override_with_altitude(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_GPS_LAT", "14.5995")
    monkeypatch.setenv("MATANGLAWIN_GPS_LON", "120.9842")
    monkeypatch.setenv("MATANGLAWIN_GPS_ALT", "35.2")
    fix = gps_provider.get_current_fix()
    assert fix.altitude == 35.2


def test_env_with_invalid_coordinates_is_unavailable(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_GPS_LAT", "999")   # out of range
    monkeypatch.setenv("MATANGLAWIN_GPS_LON", "120.9842")
    fix = gps_provider.get_current_fix()
    assert fix.available is False


def test_env_with_non_numeric_coordinates_does_not_crash(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_GPS_LAT", "not-a-number")
    monkeypatch.setenv("MATANGLAWIN_GPS_LON", "also-not-a-number")
    fix = gps_provider.get_current_fix()
    assert fix.available is False


def test_pluggable_source_takes_priority_over_env(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_GPS_LAT", "1.0")
    monkeypatch.setenv("MATANGLAWIN_GPS_LON", "1.0")
    gps_provider.set_gps_source(
        lambda: gps_provider.GpsFix(latitude=14.6, longitude=121.0, altitude=12.0)
    )
    fix = gps_provider.get_current_fix()
    assert fix.latitude == 14.6
    assert fix.longitude == 121.0


def test_pluggable_source_raising_exception_falls_back_safely(monkeypatch):
    def broken_source():
        raise RuntimeError("sensor failure")

    gps_provider.set_gps_source(broken_source)
    # Should not raise, should fall back to env/unavailable instead.
    fix = gps_provider.get_current_fix()
    assert fix.available is False


def test_pluggable_source_returning_none_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_GPS_LAT", "10.0")
    monkeypatch.setenv("MATANGLAWIN_GPS_LON", "20.0")
    gps_provider.set_gps_source(lambda: None)
    fix = gps_provider.get_current_fix()
    assert fix.latitude == 10.0
    assert fix.longitude == 20.0


def test_clearing_source_reverts_to_env_or_unavailable():
    gps_provider.set_gps_source(
        lambda: gps_provider.GpsFix(latitude=1.0, longitude=2.0)
    )
    assert gps_provider.get_current_fix().latitude == 1.0
    gps_provider.set_gps_source(None)
    assert gps_provider.get_current_fix().available is False


def test_gps_fix_to_dict_shape():
    fix = gps_provider.GpsFix(latitude=1.0, longitude=2.0, altitude=3.0)
    d = fix.to_dict()
    assert d == {"latitude": 1.0, "longitude": 2.0, "altitude": 3.0, "available": True}


def test_unavailable_constant_is_unavailable():
    assert gps_provider.UNAVAILABLE.available is False
