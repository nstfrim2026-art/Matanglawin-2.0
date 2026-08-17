"""
bridge_status.py - Liveness state for the automatic DJI photo-transfer
bridge, so the dashboard can show a small "PHOTO BRIDGE: READY / WAITING
FOR PHOTO / OFFLINE" indicator.

The companion uploader (dji_photo_bridge.py) sends a lightweight heartbeat
to /api/bridge/ping and posts photos to /api/import. Every such contact is
recorded here. The dashboard polls /api/bridge/status, which asks for a
snapshot: if the bridge has checked in recently the indicator is READY (or
WAITING FOR PHOTO while idle); otherwise OFFLINE.

Deliberately time-injectable (``now`` is passed in) so it can be unit
tested deterministically. Thread-safe: the Flask request threads update it
concurrently. It stores only timestamps and the last result STATUS string
("CRACK DETECTED" / "NO CRACK DETECTED") - never any confidence, metric,
or image data.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# How long after the last contact the bridge is still considered online.
# The companion uploader heartbeats well within this window.
DEFAULT_ONLINE_WINDOW = 45.0
# After being idle this long (but still online), show "WAITING FOR PHOTO".
DEFAULT_RECENT_PHOTO_WINDOW = 12.0

STATE_READY = "READY"
STATE_WAITING = "WAITING FOR PHOTO"
STATE_OFFLINE = "OFFLINE"


class BridgeStatus:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_contact: Optional[float] = None
        self._last_photo: Optional[float] = None
        self._last_result: Optional[str] = None
        self._photos_received = 0

    def record_contact(self, now: Optional[float] = None) -> None:
        """A heartbeat (or any bridge request) was received."""
        now = time.time() if now is None else now
        with self._lock:
            self._last_contact = now

    def record_photo(self, status: Optional[str] = None, now: Optional[float] = None) -> None:
        """A photo was received from the bridge and analyzed."""
        now = time.time() if now is None else now
        with self._lock:
            self._last_contact = now
            self._last_photo = now
            self._last_result = status
            self._photos_received += 1

    def snapshot(
        self,
        now: Optional[float] = None,
        online_window: float = DEFAULT_ONLINE_WINDOW,
        recent_photo_window: float = DEFAULT_RECENT_PHOTO_WINDOW,
    ) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            last_contact = self._last_contact
            last_photo = self._last_photo
            last_result = self._last_result
            photos = self._photos_received

        online = last_contact is not None and (now - last_contact) <= online_window
        if not online:
            state = STATE_OFFLINE
        elif last_photo is not None and (now - last_photo) <= recent_photo_window:
            state = STATE_READY
        else:
            state = STATE_WAITING

        return {
            "state": state,
            "online": online,
            "photos_received": photos,
            "last_result": last_result,
            "seconds_since_contact": None if last_contact is None else round(now - last_contact, 1),
            "seconds_since_photo": None if last_photo is None else round(now - last_photo, 1),
        }
