"""
live_pipeline.py - Wires the live drone pipeline together for app.py.

    RTSP (video_source.RtspVideoSource)
            |  latest frame
            v
    YOLO11-seg (detector.CrackDetector)
            |  DetectionResult
            v
    Auto-capture (capture_manager.CaptureManager)
            |
            v
    inspection_db.InspectionDB

This module owns a single background "processing" thread that pulls the
latest frame from the video source, runs detection, and hands the result
to the capture manager - decoupled from both the RTSP-reading thread
(video_source's own thread) and the Flask request threads.

Everything here is defensive: if the YOLO weights fail to load, or a
single frame fails to process, the pipeline logs the error into its
status and keeps running rather than taking the whole app down. A
disconnected/offline drone is an expected, normal state, not a fatal
error.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from capture_manager import CaptureManager
from detector import CrackDetector
from inspection_db import InspectionDB
from video_source import RtspVideoSource, STATE_LIVE

DETECTION_INTERVAL_SEC = 0.5  # run YOLO at most twice a second - plenty for a slow-moving inspection drone


@dataclass
class PipelineStatus:
    stream_state: str
    detector_ready: bool
    detector_error: Optional[str]
    last_detection_at: Optional[float]
    last_capture_id: Optional[int]
    frames_processed: int


class LivePipeline:
    def __init__(
        self,
        rtsp_url: str,
        weights: str,
        db: InspectionDB,
        capture_dir: str,
        detection_interval: float = DETECTION_INTERVAL_SEC,
    ):
        self.video_source = RtspVideoSource(rtsp_url)
        self.db = db
        self.capture_manager = CaptureManager(db, capture_dir)
        self.weights = weights
        self.detection_interval = detection_interval

        self._detector: Optional[CrackDetector] = None
        self._detector_error: Optional[str] = None

        self._lock = threading.Lock()
        self._last_detection_at: Optional[float] = None
        self._last_capture_id: Optional[int] = None
        self._frames_processed = 0
        self._latest_overlay = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        self.video_source.start()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LivePipeline")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.video_source.stop()

    # -- status ----------------------------------------------------------

    def get_status(self) -> PipelineStatus:
        stream_status = self.video_source.get_status()
        with self._lock:
            return PipelineStatus(
                stream_state=stream_status.state,
                detector_ready=self._detector is not None,
                detector_error=self._detector_error,
                last_detection_at=self._last_detection_at,
                last_capture_id=self._last_capture_id,
                frames_processed=self._frames_processed,
            )

    def get_latest_overlay_jpeg(self):
        """Return the most recent annotated frame (BGR numpy array) or None."""
        with self._lock:
            return None if self._latest_overlay is None else self._latest_overlay.copy()

    # -- internals -------------------------------------------------------

    def _ensure_detector(self) -> Optional[CrackDetector]:
        if self._detector is not None:
            return self._detector
        try:
            self._detector = CrackDetector(weights=self.weights)
            self._detector_error = None
        except Exception as exc:  # noqa: BLE001
            self._detector_error = str(exc)
            self._detector = None
        return self._detector

    def _run(self) -> None:
        while not self._stop_event.is_set():
            stream_status = self.video_source.get_status()
            if stream_status.state != STATE_LIVE:
                self._stop_event.wait(self.detection_interval)
                continue

            frame = self.video_source.get_latest_frame()
            if frame is None:
                self._stop_event.wait(self.detection_interval)
                continue

            detector = self._ensure_detector()
            if detector is None:
                # YOLO failed to load (e.g. missing weights). Keep the
                # video/stream status accurate, but don't spin retrying
                # model load on every loop iteration.
                self._stop_event.wait(max(self.detection_interval, 2.0))
                continue

            try:
                result = detector.process_frame(frame, draw_overlay=True)
                capture_result = self.capture_manager.maybe_capture(frame, result)

                with self._lock:
                    self._last_detection_at = time.time()
                    self._frames_processed += 1
                    if result.overlay_frame is not None:
                        self._latest_overlay = result.overlay_frame
                    if capture_result.saved:
                        self._last_capture_id = capture_result.inspection_id
            except Exception as exc:  # noqa: BLE001 - a single bad frame must not kill the loop
                with self._lock:
                    self._detector_error = f"frame processing error: {exc}"

            self._stop_event.wait(self.detection_interval)
