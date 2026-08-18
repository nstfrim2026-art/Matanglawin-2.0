"""
Tests for dji_gps_collector.py - newest-file discovery, sending the latest
sample via an injected poster, dedup, encrypted-file skip, and PC-unreachable
retry. No network is used.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dji_gps_collector as col  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class RecordingPoster:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def __call__(self, server, endpoint, payload):
        self.calls.append(payload)
        return (self.ok, "200" if self.ok else "unreachable")


def _collector(tmp_path, poster):
    return col.TelemetryCollector(str(tmp_path), server="http://pc.example:5000",
                                  poster=poster, sleep=lambda _s: None)


def test_sends_latest_sample_from_csv(tmp_path):
    shutil.copy(FIXTURES / "flight_sample.csv", tmp_path / "flight_sample.csv")
    poster = RecordingPoster(ok=True)
    c = _collector(tmp_path, poster)
    action = c.scan_once()
    assert action["action"] == "sent"
    assert len(poster.calls) == 1
    sent = poster.calls[0]
    # latest row = 7.1235,125.6544
    assert abs(sent["latitude"] - 7.1235) < 1e-6
    assert abs(sent["longitude"] - 125.6544) < 1e-6
    assert sent["source"] == "dji_flight_record"


def test_does_not_resend_unchanged(tmp_path):
    shutil.copy(FIXTURES / "flight_sample.csv", tmp_path / "flight_sample.csv")
    poster = RecordingPoster(ok=True)
    c = _collector(tmp_path, poster)
    assert c.scan_once()["action"] == "sent"
    assert c.scan_once()["action"] == "unchanged"
    assert len(poster.calls) == 1


def test_retries_when_pc_unreachable(tmp_path):
    shutil.copy(FIXTURES / "flight_sample.csv", tmp_path / "flight_sample.csv")
    poster = RecordingPoster(ok=False)
    c = _collector(tmp_path, poster)
    assert c.scan_once()["action"] == "failed"
    assert c.last_sent_ms is None            # not marked sent
    poster.ok = True
    assert c.scan_once()["action"] == "sent"  # later retry succeeds


def test_skips_encrypted_dji_container(tmp_path):
    name = "DJIFlightRecord_2026-08-18_[15-26-18].txt"
    (tmp_path / name).write_bytes(bytes([0x00, 0x01, 0xFF] * 400))
    poster = RecordingPoster(ok=True)
    c = _collector(tmp_path, poster)
    action = c.scan_once()
    assert action["action"] == "skip" and action["fmt"] == "dji_encrypted"
    assert poster.calls == []


def test_no_file_is_safe(tmp_path):
    c = _collector(tmp_path, RecordingPoster())
    assert c.scan_once()["action"] == "no_file"


def test_discover_server_prefers_explicit():
    assert col.discover_server("http://x:5000") == "http://x:5000"
    assert col.discover_server("") is None  # no mDNS resolver available -> None
