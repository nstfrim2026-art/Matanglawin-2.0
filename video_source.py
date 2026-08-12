"""
video_source.py - Background RTSP stream reader with LIVE/OFFLINE state.

Reads frames from the local MediaMTX RTSP output
(rtsp://localhost:8554/<stream_key> - see network_config.py, which is
the single source of truth for that URL) on a background thread, so the
Flask request/response cycle is never blocked waiting on network I/O.

State machine (matches the required dashboard states):

    OFFLINE     -> no publisher connected / stream not opened yet
    CONNECTING  -> we are attempting to open/reconnect to the RTSP URL
    LIVE        -> frames are actively being received

Transitions:

    start() -> CONNECTING
    frame successfully read -> LIVE
    N consecutive failed reads / stream closed -> OFFLINE -> retries -> CONNECTING

This module deliberately does NOT crash the process when the drone
disconnects or MediaMTX isn't running: every OpenCV failure is caught
and turned into an OFFLINE state, and the background thread just keeps
retrying at a fixed interval until frames start flowing again.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

STATE_OFFLINE = "OFFLINE"
STATE_CONNECTING = "CONNECTING"
STATE_LIVE = "LIVE"

RECONNECT_INTERVAL_SEC = 2.0
READ_FAILURE_THRESHOLD = 5  # consecutive failed reads before declaring OFFLINE


@dataclass
class StreamStatus:
    state: str
    rtsp_url: str
    last_frame_at: Optional[float]
    last_error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "rtsp_url": self.rtsp_url,
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
        }


class RtspVideoSource:
    """
    Background thread that keeps a `cv2.VideoCapture` on `rtsp_url` open,
    exposing the latest decoded frame and a LIVE/CONNECTING/OFFLINE
    status that the dashboard/API can poll cheaply (no I/O on the
    request path).
    """

    def __init__(self, rtsp_url: str, reconnect_interval: float = RECONNECT_INTERVAL_SEC):
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._state = STATE_OFFLINE
        self._last_frame_at: Optional[float] = None
        self._last_error: Optional[str] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._state = STATE_CONNECTING
        self._thread = threading.Thread(target=self._run, daemon=True, name="RtspVideoSource")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            self._state = STATE_OFFLINE
            self._latest_frame = None

    # -- public read API -------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_status(self) -> StreamStatus:
        with self._lock:
            return StreamStatus(
                state=self._state,
                rtsp_url=self.rtsp_url,
                last_frame_at=self._last_frame_at,
                last_error=self._last_error,
            )

    # -- internals -------------------------------------------------------

    def _set_state(self, state: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._state = state
            if error is not None:
                self._last_error = error

    def _run(self) -> None:
        # cv2 is imported lazily so importing this module doesn't require
        # opencv to be pre-loaded at process start, matching the
        # lazy-import pattern used for the heavy YOLO dependency.
        import cv2

        while not self._stop_event.is_set():
            self._set_state(STATE_CONNECTING)
            cap = None
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open RTSP stream: {self.rtsp_url}")

                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures >= READ_FAILURE_THRESHOLD:
                            raise RuntimeError("Too many consecutive failed frame reads")
                        time.sleep(0.1)
                        continue

                    consecutive_failures = 0
                    with self._lock:
                        self._latest_frame = frame
                        self._state = STATE_LIVE
                        self._last_frame_at = time.time()
                        self._last_error = None

            except Exception as exc:  # noqa: BLE001 - never let the thread die
                self._set_state(STATE_OFFLINE, error=str(exc))
            finally:
                if cap is not None:
                    cap.release()
                with self._lock:
                    self._latest_frame = None

            # Back off before retrying so we don't spin-loop against a
            # publisher that isn't there yet.
            self._stop_event.wait(self.reconnect_interval)
