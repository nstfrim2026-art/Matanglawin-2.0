"""
Tests for bridge_status.py - PHOTO BRIDGE liveness state machine.
Time is injected so transitions are deterministic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge_status import BridgeStatus, STATE_READY, STATE_WAITING, STATE_OFFLINE  # noqa: E402


def test_offline_before_any_contact():
    bs = BridgeStatus()
    snap = bs.snapshot(now=1000.0)
    assert snap["state"] == STATE_OFFLINE
    assert snap["online"] is False
    assert snap["photos_received"] == 0


def test_ready_right_after_a_photo():
    bs = BridgeStatus()
    bs.record_photo(status="CRACK DETECTED", now=1000.0)
    snap = bs.snapshot(now=1002.0)  # 2s later
    assert snap["state"] == STATE_READY
    assert snap["online"] is True
    assert snap["last_result"] == "CRACK DETECTED"
    assert snap["photos_received"] == 1


def test_waiting_when_connected_but_idle():
    bs = BridgeStatus()
    bs.record_contact(now=1000.0)  # heartbeat, no photo yet
    snap = bs.snapshot(now=1005.0)
    assert snap["state"] == STATE_WAITING
    assert snap["online"] is True


def test_offline_when_contact_is_stale():
    bs = BridgeStatus()
    bs.record_contact(now=1000.0)
    snap = bs.snapshot(now=1000.0 + 999.0)  # well past the online window
    assert snap["state"] == STATE_OFFLINE
    assert snap["online"] is False


def test_photo_after_idle_returns_to_waiting():
    bs = BridgeStatus()
    bs.record_photo(status="NO CRACK DETECTED", now=1000.0)
    # 30s later still online (heartbeats assumed) but photo is no longer recent
    bs.record_contact(now=1030.0)
    snap = bs.snapshot(now=1031.0)
    assert snap["state"] == STATE_WAITING
