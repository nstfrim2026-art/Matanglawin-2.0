"""
letsview_source.py - Screen capture source for LetsView mirroring window.

This module provides the LetsViewSource class, which captures frames directly
from the LetsView application window on Windows. It implements the same
interface as VideoSource (open, read, release, is_open) so the detector
pipeline can use it interchangeably.

On non-Windows platforms (e.g., Linux), open() returns False gracefully and
the detector shows a NO SIGNAL placeholder - this is expected since LetsView
window capture requires Windows-specific APIs.

Pipeline:
    DJI Drone -> DJI Controller -> Android Phone (DJI Fly) -> LetsView
    Screen Mirroring -> Windows Laptop -> Matanglawin Detection System
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Platform-specific imports - these may not be available on Linux.
_PLATFORM_OK = False
_gw = None
_mss_module = None

try:
    import mss as _mss_module_import

    _mss_module = _mss_module_import
except ImportError:
    _mss_module = None

try:
    import pygetwindow as _gw_import

    # pygetwindow imports but its Windows functions only work on Windows
    if platform.system() == "Windows":
        _gw = _gw_import
        _PLATFORM_OK = True
except (ImportError, NotImplementedError):
    _gw = None

# Window title patterns to match (case-insensitive partial match)
_LETSVIEW_TITLES = [
    "letsview",
    "letsview player",
    "letsview mirror",
    "letsview windows",
]


def _make_status_frame(
    message: str, width: int = 960, height: int = 540
) -> np.ndarray:
    """Create a dark status frame with centered text message."""
    frame = np.full((height, width, 3), (26, 22, 20), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(message, font, scale, thickness)
    tx = (width - text_size[0]) // 2
    ty = (height + text_size[1]) // 2
    cv2.putText(
        frame, message, (tx, ty), font, scale, (90, 90, 90), thickness, cv2.LINE_AA
    )
    return frame


def _find_letsview_window():
    """
    Search for a LetsView window using case-insensitive partial title matching.

    Returns the window object if found, None otherwise.
    """
    if _gw is None:
        return None

    try:
        all_windows = _gw.getAllWindows()
    except Exception:
        return None

    for win in all_windows:
        title_lower = (win.title or "").lower()
        for pattern in _LETSVIEW_TITLES:
            if pattern in title_lower:
                return win
    return None


class LetsViewSource:
    """
    Screen capture source that grabs frames from the LetsView window.

    Implements the same interface as VideoSource:
        open() -> bool
        read() -> (bool, Optional[ndarray])
        release() -> None
        is_open (property)

    On non-Windows platforms, open() returns False immediately (no LetsView
    window capture is possible without Windows APIs).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._window = None
        self._sct = None  # mss instance
        self._opened = False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened

    def open(self) -> bool:
        """
        Search for the LetsView window and prepare for capture.

        Returns True if the window was found and capture can begin.
        Returns False if the platform is unsupported or no window was found.
        """
        with self._lock:
            self._release_locked()

            # Platform check: only works on Windows
            if not _PLATFORM_OK:
                return False

            # mss is required for capture
            if _mss_module is None:
                return False

            window = _find_letsview_window()
            if window is None:
                return False

            self._window = window
            try:
                self._sct = _mss_module.mss()
            except Exception:
                self._window = None
                return False

            self._opened = True
            return True

    def read(self):
        """
        Capture the current frame from the LetsView window.

        Returns:
            (True, frame) - successful capture or status frame
            (False, None) - window disappeared, triggers reconnection
        """
        with self._lock:
            if not self._opened:
                return False, None

            # Re-check that the window still exists
            window = self._window
            if window is None:
                self._opened = False
                return False, None

            # Check if window is minimized
            try:
                if window.isMinimized:
                    return True, _make_status_frame("LetsView window minimized.")
            except Exception:
                # Window may have been closed
                self._release_locked()
                return False, None

            # Get current window geometry (handles resize automatically)
            try:
                left = window.left
                top = window.top
                width = window.width
                height = window.height
            except Exception:
                # Window no longer accessible
                self._release_locked()
                return False, None

            # Validate geometry
            if width <= 0 or height <= 0:
                return True, _make_status_frame("LetsView window minimized.")

            # Capture the window region using mss
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }

            try:
                screenshot = self._sct.grab(monitor)
            except Exception:
                # Capture failed - window may have closed
                self._release_locked()
                return False, None

            # Convert BGRA -> BGR numpy array
            frame = np.array(screenshot, dtype=np.uint8)
            # mss returns BGRA format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            return True, frame

    def release(self) -> None:
        """Release capture resources."""
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        """Internal release without acquiring lock (caller must hold lock)."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        self._window = None
        self._opened = False
