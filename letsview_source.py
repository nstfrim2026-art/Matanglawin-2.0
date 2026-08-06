"""
letsview_source.py - Screen capture source for LetsView mirroring window.

This module provides the LetsViewSource class, which captures frames directly
from the LetsView application window on Windows using the Windows Graphics
Capture API (via winsdk). It implements the same interface as VideoSource
(open, read, release, is_open) so the detector pipeline can use it
interchangeably.

On non-Windows platforms or if winsdk is not available, open() returns False
gracefully and the detector shows a NO SIGNAL placeholder.

Pipeline:
    DJI Drone -> DJI Controller -> Android Phone (DJI Fly) -> LetsView
    Screen Mirroring -> Windows Laptop -> Matanglawin Detection System
"""

from __future__ import annotations

import platform
import threading
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Platform-specific imports for Windows Graphics Capture API
# ---------------------------------------------------------------------------
_PLATFORM_OK = False
_winsdk_available = False

if platform.system() == "Windows":
    try:
        import ctypes
        import ctypes.wintypes

        # Attempt to import winsdk for Windows Graphics Capture
        from winsdk.windows.graphics.capture import (
            Direct3D11CaptureFramePool,
            GraphicsCaptureItem,
            GraphicsCaptureSession,
        )
        from winsdk.windows.graphics.directx import DirectXPixelFormat
        from winsdk.windows.graphics.directx.direct3d11 import (
            IDirect3DDevice,
            IDirect3DSurface,
        )
        import winsdk.windows.graphics.capture as wgc

        _winsdk_available = True
        _PLATFORM_OK = True
    except (ImportError, OSError, AttributeError):
        _winsdk_available = False
        _PLATFORM_OK = False
else:
    _winsdk_available = False

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
    ctypes on Windows. Returns the HWND (int) if found, None otherwise.
    """
    if platform.system() != "Windows":
        return None

    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        GetWindowTextW = user32.GetWindowTextW
        GetWindowTextLengthW = user32.GetWindowTextLengthW
        IsWindowVisible = user32.IsWindowVisible

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL,
            ctypes.wintypes.HWND,
            ctypes.wintypes.LPARAM,
        )

        found_hwnd = [None]

        def enum_callback(hwnd, lparam):
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.lower()
            for pattern in _LETSVIEW_TITLES:
                if pattern in title:
                    found_hwnd[0] = hwnd
                    return False  # Stop enumeration
            return True

        EnumWindows(WNDENUMPROC(enum_callback), 0)
        return found_hwnd[0]
    except Exception:
        return None


class LetsViewSource:
    """
    Screen capture source that grabs frames from the LetsView window
    using the Windows Graphics Capture API (winsdk).

    Implements the same interface as VideoSource:
        open() -> bool
        read() -> (bool, Optional[ndarray])
        release() -> None
        is_open (property)

    On non-Windows platforms or when winsdk is unavailable, open() returns
    False immediately (no LetsView window capture is possible).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._opened = False
        self._session = None
        self._frame_pool = None
        self._hwnd = None
        self._latest_frame: Optional[np.ndarray] = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened

    def open(self) -> bool:
        """
        Search for the LetsView window and prepare for capture using
        the Windows Graphics Capture API.

        Returns True if the window was found and capture can begin.
        Returns False if the platform is unsupported, winsdk is unavailable,
        or no LetsView window was found.
        """
        with self._lock:
            self._release_locked()

            # Platform and dependency check
            if not _PLATFORM_OK or not _winsdk_available:
                return False

            # Find the LetsView window by HWND
            hwnd = _find_letsview_hwnd()
            if hwnd is None:
                return False

            try:
                # Create GraphicsCaptureItem from HWND
                import winsdk.windows.graphics.capture as wgc

                interop = wgc.GraphicsCaptureItem
                item = interop.create_from_window_handle(hwnd)
                if item is None:
                    return False

                # Create D3D11 device and frame pool
                from winsdk.windows.graphics.directx import DirectXPixelFormat
                from winsdk.windows.graphics.directx.direct3d11 import (
                    IDirect3DDevice,
                )

                device = IDirect3DDevice.create()
                size = item.size
                pool = Direct3D11CaptureFramePool.create_free_threaded(
                    device,
                    DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED,
                    2,  # buffer count
                    size,
                )

                session = pool.create_capture_session(item)
                session.start_capture()

                self._hwnd = hwnd
                self._session = session
                self._frame_pool = pool
                self._opened = True
                return True
            except Exception:
                self._release_locked()
                return False

    def read(self):
        """
        Capture the current frame from the LetsView window using the
        Windows Graphics Capture API.

        This captures directly from the window HWND, so it continues to
        work even if the window is covered by other applications or is
        partially offscreen.

        Returns:
            (True, frame) - successful capture
            (False, None) - capture session lost, triggers reconnection
        """
        with self._lock:
            if not self._opened:
                return False, None

            if self._frame_pool is None:
                self._opened = False
                return False, None

            try:
                frame = self._frame_pool.try_get_next_frame()
                if frame is None:
                    # No new frame available yet; return last known frame
                    if self._latest_frame is not None:
                        return True, self._latest_frame.copy()
                    return True, _make_status_frame("Waiting for LetsView frame...")

                # Convert the captured surface to numpy array
                surface = frame.surface
                # Access the underlying bitmap data
                import winsdk.windows.graphics.imaging as imaging

                soft_bitmap = imaging.SoftwareBitmap.create_copy_from_surface_async(
                    surface,
                    imaging.BitmapPixelFormat.BGRA8,
                ).get()

                buf = soft_bitmap.lock_buffer(imaging.BitmapBufferAccessMode.READ)
                ref = buf.create_reference()
                data = np.frombuffer(ref, dtype=np.uint8)
                h = soft_bitmap.pixel_height
                w = soft_bitmap.pixel_width
                bgra = data.reshape((h, w, 4))
                bgr = bgra[:, :, :3].copy()

                self._latest_frame = bgr
                frame.close()
                return True, bgr

            except Exception:
                # Capture failed - session may be invalid
                self._release_locked()
                return False, None

    def release(self) -> None:
        """Release capture resources."""
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        """Internal release without acquiring lock (caller must hold lock)."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        if self._frame_pool is not None:
            try:
                self._frame_pool.close()
            except Exception:
                pass
            self._frame_pool = None
        self._hwnd = None
        self._latest_frame = None
        self._opened = False
