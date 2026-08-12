"""
gps_provider.py - Optional GPS/location association for inspection events.

Matanglawin does NOT require a dedicated GPS module to function. GPS is
purely supplementary metadata attached to an inspection record (see
inspection_service.py / inspection_db.py) - detection keeps working with
or without it.

We NEVER fabricate coordinates. If no real source is available, every
function here returns `None` values and callers must render/store that
as "GPS: Unavailable".

Supported sources, in priority order:

1. A pluggable in-process provider registered via `set_gps_source()`.
   This is the extension point for a *future* real source - e.g. a
   background thread parsing DJI SRT/telemetry sidecar data, a serial
   GPS receiver, or any other feed - without changing any other module.
2. Manual/host-provided coordinates via environment variables
   (`MATANGLAWIN_GPS_LAT` / `MATANGLAWIN_GPS_LON` /
   `MATANGLAWIN_GPS_ALT`), useful for a fixed inspection site or for
   testing the GPS-present code path without hardware.
3. Otherwise: unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GpsFix:
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "available": self.available,
        }


UNAVAILABLE = GpsFix(latitude=None, longitude=None, altitude=None)

# A pluggable source: a zero-arg callable returning a GpsFix (or None).
# Registered at runtime by whatever component owns a real GPS feed.
_gps_source: Optional[Callable[[], Optional[GpsFix]]] = None


def set_gps_source(source: Optional[Callable[[], Optional[GpsFix]]]) -> None:
    """
    Register (or clear, with None) a callable that returns the current
    GpsFix on demand. Intended for a future real GPS/telemetry feed to
    plug into without touching inspection_service.py or inspection_db.py.
    """
    global _gps_source
    _gps_source = source


def _valid_coord(lat, lon) -> bool:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def _from_env() -> Optional[GpsFix]:
    lat = os.environ.get("MATANGLAWIN_GPS_LAT", "").strip()
    lon = os.environ.get("MATANGLAWIN_GPS_LON", "").strip()
    if not lat or not lon:
        return None
    if not _valid_coord(lat, lon):
        return None
    alt_raw = os.environ.get("MATANGLAWIN_GPS_ALT", "").strip()
    try:
        alt = float(alt_raw) if alt_raw else None
    except ValueError:
        alt = None
    return GpsFix(latitude=float(lat), longitude=float(lon), altitude=alt)


def get_current_fix() -> GpsFix:
    """
    Return the best currently-available GPS fix, or UNAVAILABLE.

    Never raises, never invents coordinates. Safe to call on every
    detection/capture event.
    """
    if _gps_source is not None:
        try:
            fix = _gps_source()
        except Exception:
            fix = None
        if fix is not None and fix.available:
            return fix

    env_fix = _from_env()
    if env_fix is not None:
        return env_fix

    return UNAVAILABLE
