"""
letsview_source.py - Screen capture source for LetsView mirroring window.

This module provides the LetsViewSource class, which captures frames directly
from the LetsView application window on Windows using the pywin32 PrintWindow
API (PW_RENDERFULLCONTENT=2). This approach correctly captures the window
using its own rendering pipeline, so it works even when the window is covered
by other applications or is partially off-screen.

On non-Windows platforms or if pywin32 is not available, open() returns False
gracefully and the detector shows a NO SIGNAL placeholder.

Pipeline:
    DJI Drone -> DJI Controller -> Android Phone (DJI Fly) -> LetsView
    Screen Mirroring -> Windows Laptop -> Matanglawin Detection System
"""

from __future__ import annotations

import ctypes
import platform
import threading
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Platform-specific imports for pywin32
# ---------------------------------------------------------------------------
_PLATFORM_OK = False
_pywin32_available = False

if platform.system() == "Windows":
    try:
        import win32gui
        import win32ui
        import win32con

        _pywin32_available = True
        _PLATFORM_OK = True
    except ImportError:
        _pywin32_available = False
        _PLATFORM_OK = False
else:
    _pywin32_available = False

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


def _find_letsview_hwnd():
    """
    Search for a LetsView window by enumerating top-level windows using
    win32gui on Windows. Returns the HWND (int) if found, None otherwise.
    """
    if not _PLATFORM_OK or not _pywin32_available:
        return None

    try:
        found_hwnd = [None]

        def enum_callback(hwnd, lparam):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd).lower()
            for pattern in _LETSVIEW_TITLES:
                if pattern in title:
                    found_hwnd[0] = hwnd
                    return False  # Stop enumeration
            return True

        win32gui.EnumWindows(enum_callback, None)
        return found_hwnd[0]
    except Exception:
        return None


def _capture_window_printwindow(hwnd: int) -> Optional[np.ndarray]:
    """
    Capture a window's content using PrintWindow with PW_RENDERFULLCONTENT=2.

    This flag instructs Windows to render the window using its own rendering
    pipeline (including Direct2D/Direct3D content), which works correctly
    even when the window is covered by other windows or is partially off-screen.

    Returns the captured frame as a BGR numpy array, or None on failure.
    """
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top

        if w <= 0 or h <= 0:
            return None

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(save_bitmap)

        # PW_RENDERFULLCONTENT = 2: captures even when window is covered or uses
        # hardware-accelerated rendering (Direct2D, Direct3D, etc.)
        result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

        if not result:
            # Fallback: try with flag=0 (basic GDI capture)
            ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)

        bmp_info = save_bitmap.GetInfo()
        bmp_str = save_bitmap.GetBitmapBits(True)

        img = np.frombuffer(bmp_str, dtype=np.uint8).reshape(
            bmp_info["bmHeight"], bmp_info["bmWidth"], 4
        )
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Cleanup GDI resources
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return frame
    except Exception:
        return None


class LetsViewSource:
    """
    Screen capture source that grabs frames from the LetsView window
    using the pywin32 PrintWindow API (PW_RENDERFULLCONTENT=2).

    Implements the same interface as VideoSource:
        open() -> bool
        read() -> (bool, Optional[ndarray])
        release() -> None
        is_open (property)

    On non-Windows platforms or when pywin32 is unavailable, open() returns
    False immediately (no LetsView window capture is possible).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._opened = False
        self._hwnd = None
        self._latest_frame: Optional[np.ndarray] = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened

    def open(self) -> bool:
        """
        Search for the LetsView window and prepare for capture using
        the pywin32 PrintWindow API.

        Returns True if the window was found and capture can begin.
        Returns False if the platform is unsupported, pywin32 is unavailable,
        or no LetsView window was found.
        """
        with self._lock:
            self._release_locked()

            # Platform and dependency check
            if not _PLATFORM_OK or not _pywin32_available:
                return False

            # Find the LetsView window by HWND
            hwnd = _find_letsview_hwnd()
            if hwnd is None:
                return False

            self._hwnd = hwnd
            self._opened = True
            return True

    def read(self):
        """
        Capture the current frame from the LetsView window using
        PrintWindow with PW_RENDERFULLCONTENT=2.

        This captures directly from the window's own rendering pipeline,
        so it continues to work even if the window is covered by other
        applications or is partially off-screen.

        Returns:
            (True, frame) - successful capture
            (False, None) - window lost or capture failed, triggers reconnection
        """
        with self._lock:
            if not self._opened or self._hwnd is None:
                return False, None

            # Verify window still exists
            try:
                if not win32gui.IsWindow(self._hwnd):
                    self._release_locked()
                    return False, None
            except Exception:
                self._release_locked()
                return False, None

            frame = _capture_window_printwindow(self._hwnd)

            if frame is None:
                # If we got nothing but had a previous frame, return that
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy()
                return True, _make_status_frame("Waiting for LetsView frame...")

            self._latest_frame = frame
            return True, frame.copy()

    def release(self) -> None:
        """Release capture resources."""
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        """Internal release without acquiring lock (caller must hold lock)."""
        self._hwnd = None
        self._latest_frame = None
        self._opened = False
