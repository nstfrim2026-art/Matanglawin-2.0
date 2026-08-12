"""
video_source.py - Configurable live video source abstraction.

The goal is that the detection pipeline (detector.py) never needs to know
*where* frames come from. It just asks a VideoSource for the next frame.
Swapping the physical camera - laptop webcam today, a phone running an
IP camera app tomorrow, a drone's video feed in the future - never requires
touching the inference/detection code, only the small mapping in
SOURCE_REGISTRY below (or the OpenCV-compatible index/URL passed in).

Supported source kinds:
    "webcam"  - a local camera attached to this machine, opened by
                numeric index (0, 1, 2, ...).
    "ip_camera" / "ip" - any camera exposed over the network as an MJPEG/
                RTSP/HTTP stream (IP cameras, and - later - a drone's
                video downlink all work the same way here).

Both are opened with cv2.VideoCapture, which already understands plain
integers (local camera index) and URL strings (network streams), so this
module is mostly a thin, named wrapper that keeps the "which source" and
"how to read frames" concerns in one small, swappable place.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Known source presets. `spec` is whatever cv2.VideoCapture accepts:
# an int (local device index) or a string (URL / path).
#
# IP Camera's default MJPEG URL looks like http://<phone-ip>:4747/video -
# override IP_CAMERA_URL via the environment for your own camera's IP.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    spec: object  # int (device index) or str (URL)


def build_registry(ip_camera_url: str, drone_url: str) -> dict:
    return {
        "webcam": SourceSpec("webcam", "Webcam", 0),
        "ip_camera": SourceSpec("ip_camera", "IP Camera", ip_camera_url),
        "drone": SourceSpec("drone", "Drone Camera", drone_url),
    }


class VideoSource:
    """
    Thin, swappable wrapper around cv2.VideoCapture.

    Usage:
        vs = VideoSource(spec=0)   # or spec="http://192.168.1.5:4747/video"
        vs.open()
        ok, frame = vs.read()
        vs.release()
    """

    def __init__(self, spec):
        self._spec = spec
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        with self._lock:
            self._release_locked()
            cap = cv2.VideoCapture(self._spec)
            if not cap.isOpened():
                cap.release()
                self._cap = None
                return False
            self._cap = cap
            return True

    def read(self):
        """Returns (success: bool, frame: Optional[np.ndarray])."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            ok, frame = self._cap.read()
            if not ok:
                return False, None
            return True, frame

    def release(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def placeholder_frame(width: int = 960, height: int = 540, message: str = "NO SIGNAL") -> np.ndarray:
    """A dark placeholder frame shown while no camera source is connected."""
    frame = np.full((height, width, 3), (26, 22, 20), dtype=np.uint8)  # dark charcoal, BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 2
    text_size, _ = cv2.getTextSize(message, font, scale, thickness)
    tx = (width - text_size[0]) // 2
    ty = (height + text_size[1]) // 2
    cv2.putText(frame, message, (tx, ty), font, scale, (90, 90, 90), thickness, cv2.LINE_AA)
    return frame
