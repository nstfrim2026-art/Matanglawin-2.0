"""
Tests for video_source.py - RTSP state machine (OFFLINE/CONNECTING/LIVE),
resilience to a missing MediaMTX instance, and behavior on disconnect.
"""

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_source  # noqa: E402


def test_initial_status_is_offline():
    src = video_source.RtspVideoSource("rtsp://localhost:8554/matanglawin")
    status = src.get_status()
    assert status.state == video_source.STATE_OFFLINE
    assert status.last_frame_at is None


def test_get_latest_frame_is_none_before_start():
    src = video_source.RtspVideoSource("rtsp://localhost:8554/matanglawin")
    assert src.get_latest_frame() is None


def test_no_mediamtx_running_settles_to_offline_without_crashing():
    """
    With no MediaMTX/RTSP server actually listening, the background
    thread must repeatedly fail to connect, report OFFLINE with a
    clear error, and never raise into the calling thread.
    """
    src = video_source.RtspVideoSource(
        "rtsp://localhost:8554/matanglawin", reconnect_interval=0.3
    )
    src.start()
    try:
        time.sleep(2.0)
        status = src.get_status()
        assert status.state == video_source.STATE_OFFLINE
        assert status.last_error is not None
    finally:
        src.stop()


def test_stop_before_start_does_not_crash():
    src = video_source.RtspVideoSource("rtsp://localhost:8554/matanglawin")
    src.stop()  # never started - must be a safe no-op
    assert src.get_status().state == video_source.STATE_OFFLINE


def test_double_start_is_idempotent():
    src = video_source.RtspVideoSource(
        "rtsp://localhost:8554/matanglawin", reconnect_interval=0.5
    )
    src.start()
    first_thread = src._thread
    src.start()  # calling start() again should not spawn a second thread
    try:
        assert src._thread is first_thread
    finally:
        src.stop()


@pytest.fixture()
def fake_cv2_live_then_disconnect(monkeypatch):
    """
    Install a fake cv2 module whose VideoCapture yields a few real
    frames (simulating a live drone feed) and then starts failing reads
    (simulating a disconnect), so we can verify the LIVE state
    transition and the reconnect behavior without a real RTSP server.
    """
    fake_cv2 = types.ModuleType("cv2")

    class FakeCap:
        def __init__(self, url):
            self.url = url
            self.opened = True
            self.reads = 0

        def isOpened(self):
            return self.opened

        def read(self):
            self.reads += 1
            if self.reads <= 5:
                return True, np.zeros((10, 10, 3), dtype=np.uint8)
            return False, None

        def release(self):
            self.opened = False

    fake_cv2.VideoCapture = FakeCap
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    yield FakeCap


def test_transitions_to_live_on_successful_frames(fake_cv2_live_then_disconnect):
    src = video_source.RtspVideoSource(
        "rtsp://localhost:8554/matanglawin", reconnect_interval=0.3
    )
    src.start()
    try:
        time.sleep(0.3)
        status = src.get_status()
        assert status.state == video_source.STATE_LIVE
        frame = src.get_latest_frame()
        assert frame is not None
        assert frame.shape == (10, 10, 3)
    finally:
        src.stop()


def test_status_to_dict_shape():
    src = video_source.RtspVideoSource("rtsp://localhost:8554/matanglawin")
    d = src.get_status().to_dict()
    for key in ["state", "rtsp_url", "last_frame_at", "last_error"]:
        assert key in d
