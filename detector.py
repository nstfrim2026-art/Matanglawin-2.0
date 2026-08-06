"""
detector.py - Continuous live-video crack detection worker.

This module owns the always-running background loop that:
    1. Pulls frames from the currently selected VideoSource.
    2. Runs each frame through the YOLO11-seg inference pipeline
       (inference_core.annotate_frame) to highlight visible cracks.
    3. Keeps the latest annotated JPEG frame and a small status dict
       (camera connected? crack currently visible?) available for the
       Flask routes to read, without blocking the video loop itself.

Flask routes never touch the camera or the model directly - they only
read the latest snapshot this worker keeps in memory. That keeps the
video-source / inference concerns fully separate from the web layer, per
the requested clean architecture (video source, inference, Flask routes,
templates, static are all separate modules/files).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from capture_manager import CaptureManager
from inference_core import annotate_frame
from letsview_source import LetsViewSource
from video_source import VideoSource, placeholder_frame


class Detector:
    def __init__(
        self,
        registry: dict,
        weights: str,
        conf: float = 0.25,
        imgsz: int = 640,
        target_fps: float = 8.0,
        default_source_key: str = "webcam",
    ):
        self._registry = registry
        self._weights = weights
        self._conf = conf
        self._imgsz = imgsz
        self._min_frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        self._lock = threading.Lock()
        self._source_key = default_source_key
        self._video_source: Optional[VideoSource] = None
        self._pending_source_key: Optional[str] = default_source_key

        self._latest_jpeg: bytes = self._encode(placeholder_frame(message="STARTING..."))
        self._camera_connected = False
        self._crack_present = False
        self._crack_metadata: list = []
        self._last_frame_time: float = 0.0  # heartbeat: monotonic time of last frame

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Capture manager for auto-capture workflow
        self._capture_manager = CaptureManager(
            weights=weights,
            conf=conf,
            imgsz=imgsz,
        )

    # -- lifecycle ----------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    # -- public API used by Flask routes -------------------------------
    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> dict:
        with self._lock:
            return {
                "camera_connected": self._camera_connected,
                "crack_present": self._crack_present,
                "source": self._source_key,
                "cracks": self._crack_metadata,
                "last_frame_time": self._last_frame_time,
            }

    def set_source(self, source_key: str) -> bool:
        if source_key not in self._registry:
            return False
        with self._lock:
            self._pending_source_key = source_key
        return True

    def available_sources(self) -> list:
        return [
            {"key": s.key, "label": s.label} for s in self._registry.values()
        ]

    def get_capture_manager(self) -> CaptureManager:
        """Return the capture manager instance for use by Flask routes."""
        return self._capture_manager

    # -- internal worker loop -------------------------------------------
    def _run(self) -> None:
        reconnect_cooldown = 0.0  # monotonic time when next reconnect is allowed
        _failure_count = 0        # Track consecutive source failures
        _failure_window_start = time.monotonic()

        while not self._stop_event.is_set():
            try:
                self._maybe_switch_source()

                if self._video_source is None or not self._video_source.is_open:
                    self._set_disconnected_frame()

                    # Automatic reconnection: if no pending source change and we
                    # have a known source key, re-attempt opening after a cooldown.
                    with self._lock:
                        has_pending = self._pending_source_key is not None
                        current_key = self._source_key

                    if not has_pending and current_key in self._registry:
                        now = time.monotonic()
                        if now >= reconnect_cooldown:
                            spec = self._registry[current_key].spec
                            if spec == "letsview://":
                                vs = LetsViewSource()
                            else:
                                vs = VideoSource(spec)

                            if vs.open():
                                self._video_source = vs
                                _failure_count = 0
                                continue
                            else:
                                vs.release()

                            # Track failures and increase cooldown with backoff
                            _failure_count += 1
                            now2 = time.monotonic()
                            # Reset failure window every 30 seconds
                            if now2 - _failure_window_start > 30.0:
                                _failure_count = 1
                                _failure_window_start = now2

                            # If more than 5 failures in 30s, use 5s cooldown
                            if _failure_count > 5:
                                reconnect_cooldown = time.monotonic() + 5.0
                            else:
                                reconnect_cooldown = time.monotonic() + 1.0

                    time.sleep(0.5)
                    continue

                loop_start = time.monotonic()
                ok, frame = self._video_source.read()

                if not ok or frame is None:
                    self._set_disconnected_frame()
                    # Camera dropped - try to reopen next iteration.
                    self._video_source.release()
                    self._video_source = None
                    time.sleep(0.5)
                    continue

                try:
                    annotated, crack_present, crack_metadata, _masks = annotate_frame(
                        frame,
                        weights=self._weights,
                        conf=self._conf,
                        imgsz=self._imgsz,
                    )
                except Exception:
                    annotated, crack_present, crack_metadata = frame, False, []

                # Stream the RAW frame (no overlays) for a clean live feed.
                # The annotated frame is only used for capture analysis.
                jpeg_bytes = self._encode(frame)
                with self._lock:
                    self._latest_jpeg = jpeg_bytes
                    self._camera_connected = True
                    self._crack_present = crack_present
                    self._crack_metadata = crack_metadata
                    self._last_frame_time = time.monotonic()

                # Pass raw frame and metadata to the capture manager for
                # automatic capture workflow (non-blocking state machine check)
                try:
                    self._capture_manager.process_frame(frame, crack_metadata)
                except Exception:
                    pass  # Never let capture logic break the live feed

                elapsed = time.monotonic() - loop_start
                remaining = self._min_frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            except Exception:
                # Catch-all: log nothing to avoid noise, sleep briefly and continue
                time.sleep(1.0)
                continue

    def _maybe_switch_source(self) -> None:
        with self._lock:
            pending = self._pending_source_key
            self._pending_source_key = None

        if pending is None:
            return

        if self._video_source is not None:
            self._video_source.release()
            self._video_source = None

        spec = self._registry[pending].spec

        # Use LetsViewSource for the LetsView screen capture source;
        # otherwise use the standard OpenCV-based VideoSource.
        if spec == "letsview://":
            vs = LetsViewSource()
        else:
            vs = VideoSource(spec)

        opened = vs.open()

        with self._lock:
            self._source_key = pending
        self._video_source = vs if opened else None

        # Update the capture manager with the current camera source name
        self._capture_manager.set_camera_source(pending)

        if not opened:
            self._set_disconnected_frame()

    def _set_disconnected_frame(self) -> None:
        with self._lock:
            self._camera_connected = False
            self._crack_present = False
            self._crack_metadata = []
            self._latest_jpeg = self._encode(placeholder_frame(message="NO SIGNAL"))

    @staticmethod
    def _encode(frame) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            # Fall back to a tiny black frame rather than raising.
            ok, buf = cv2.imencode(".jpg", placeholder_frame())
        return buf.tobytes()
