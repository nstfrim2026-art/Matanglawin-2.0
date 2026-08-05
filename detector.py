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

from inference_core import annotate_frame
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

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

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

    # -- internal worker loop -------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._maybe_switch_source()

            if self._video_source is None or not self._video_source.is_open:
                self._set_disconnected_frame()
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
                annotated, crack_present, crack_metadata = annotate_frame(
                    frame,
                    weights=self._weights,
                    conf=self._conf,
                    imgsz=self._imgsz,
                )
            except Exception:
                annotated, crack_present, crack_metadata = frame, False, []

            jpeg_bytes = self._encode(annotated)
            with self._lock:
                self._latest_jpeg = jpeg_bytes
                self._camera_connected = True
                self._crack_present = crack_present
                self._crack_metadata = crack_metadata

            elapsed = time.monotonic() - loop_start
            remaining = self._min_frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

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
        vs = VideoSource(spec)
        opened = vs.open()

        with self._lock:
            self._source_key = pending
        self._video_source = vs if opened else None
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
