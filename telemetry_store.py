"""
telemetry_store.py - In-memory aircraft-telemetry buffer with nearest-in-time
matching, for the offline DJI GPS geotagging feature.

Telemetry samples (aircraft latitude/longitude/altitude + a timestamp) arrive
over the LAN at POST /api/telemetry. This store:

    * validates + normalizes each sample (rejects impossible coordinates and
      "null island" 0,0; parses ISO-8601 / epoch timestamps to epoch-ms),
    * keeps a bounded, time-ordered buffer plus the latest valid sample,
    * at capture time returns the sample **closest to the capture timestamp**
      (not merely the newest) with a `gps_time_delta_ms` quality figure,
    * reports staleness so a capture can record `gps_available=false` rather
      than attaching a wrong/old fix.

It is deliberately source-agnostic: it does not care whether samples came
from a DJI FlightRecord export, an external GPS, or a fixture - it only does
time association. Thread-safe (Flask request threads write; captures read).
No third-party dependencies.
"""

from __future__ import annotations

import bisect
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# How close (ms) a telemetry sample must be to the capture time to be trusted.
DEFAULT_MAX_MATCH_MS = 2000.0
# How old (ms) the newest sample may be before the source is "stale".
DEFAULT_STALE_MS = 5000.0
DEFAULT_BUFFER = 6000  # ~10 min at 10 Hz


def parse_timestamp_ms(value) -> Optional[float]:
    """
    Normalize a timestamp to epoch milliseconds (float), or None if invalid.

    Accepts:
      * ISO-8601 strings with or without timezone (e.g.
        "2026-08-18T15:26:18.420+08:00", trailing "Z" allowed),
      * epoch seconds or milliseconds as int/float/numeric-string.
    Naive datetimes are assumed UTC.
    """
    if value is None:
        return None
    # numeric epoch (seconds or ms)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        return v if v > 1e11 else v * 1000.0  # >~2001 in ms => already ms
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # numeric string?
        try:
            v = float(s)
            return v if v > 1e11 else v * 1000.0
        except ValueError:
            pass
        if s.endswith("Z") or s.endswith("z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000.0
    return None


def _valid_latlon(lat, lon) -> bool:
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return False
    if abs(lat) < 1e-7 and abs(lon) < 1e-7:
        return False  # (0,0) null island -> treat as no fix
    return True


@dataclass
class TelemetrySample:
    latitude: float
    longitude: float
    altitude_m: Optional[float]
    timestamp_ms: float          # epoch ms (normalized)
    source: str = "dji_flight_record"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Also expose an ISO string for convenience/readability.
        d["timestamp"] = datetime.fromtimestamp(
            self.timestamp_ms / 1000.0, tz=timezone.utc
        ).isoformat()
        return d


class TelemetryStore:
    def __init__(
        self,
        max_samples: int = DEFAULT_BUFFER,
        max_match_ms: float = DEFAULT_MAX_MATCH_MS,
        stale_ms: float = DEFAULT_STALE_MS,
    ):
        self.max_samples = max_samples
        self.max_match_ms = max_match_ms
        self.stale_ms = stale_ms
        self._lock = threading.Lock()
        self._times: List[float] = []                 # sorted epoch-ms
        self._samples: List[TelemetrySample] = []      # parallel to _times
        self._latest: Optional[TelemetrySample] = None
        self._received = 0
        self._rejected = 0

    # -- ingest --------------------------------------------------------

    def add(
        self,
        latitude,
        longitude,
        altitude_m=None,
        timestamp=None,
        source: str = "dji_flight_record",
    ) -> Optional[TelemetrySample]:
        """
        Validate and insert one sample. Returns the stored TelemetrySample,
        or None if it was rejected (bad coords / unparseable time). Never
        raises on bad input.
        """
        if not _valid_latlon(latitude, longitude):
            with self._lock:
                self._rejected += 1
            return None
        ts = parse_timestamp_ms(timestamp)
        if ts is None:
            ts = time.time() * 1000.0  # fall back to arrival time
        alt = None
        try:
            alt = None if altitude_m is None else float(altitude_m)
        except (TypeError, ValueError):
            alt = None

        sample = TelemetrySample(float(latitude), float(longitude), alt, ts, str(source))
        with self._lock:
            idx = bisect.bisect_left(self._times, ts)
            self._times.insert(idx, ts)
            self._samples.insert(idx, sample)
            if len(self._times) > self.max_samples:
                # drop the oldest
                self._times.pop(0)
                self._samples.pop(0)
            if self._latest is None or ts >= self._latest.timestamp_ms:
                self._latest = sample
            self._received += 1
        return sample

    # -- query ---------------------------------------------------------

    def latest(self, now_ms: Optional[float] = None) -> Tuple[Optional[TelemetrySample], bool]:
        """Return (latest_sample, is_stale). Stale if older than stale_ms."""
        now_ms = time.time() * 1000.0 if now_ms is None else now_ms
        with self._lock:
            latest = self._latest
        if latest is None:
            return None, True
        return latest, (now_ms - latest.timestamp_ms) > self.stale_ms

    def nearest(self, capture_ms: float) -> Tuple[Optional[TelemetrySample], Optional[float]]:
        """
        Return (sample, delta_ms) for the buffered sample closest in time to
        `capture_ms`, or (None, None) if the buffer is empty. `delta_ms` is
        the absolute time difference.
        """
        with self._lock:
            if not self._times:
                return None, None
            times = self._times
            samples = self._samples
            i = bisect.bisect_left(times, capture_ms)
            best_idx = i
            if i == 0:
                best_idx = 0
            elif i >= len(times):
                best_idx = len(times) - 1
            else:
                before, after = times[i - 1], times[i]
                best_idx = i if (after - capture_ms) < (capture_ms - before) else i - 1
            sample = samples[best_idx]
        return sample, abs(sample.timestamp_ms - capture_ms)

    def match_for_capture(self, capture_ms: Optional[float] = None) -> dict:
        """
        Association result for a capture: the nearest sample if it is within
        `max_match_ms`, else no fix. Never raises.

        Returns a dict with:
            gps_available, gps_time_delta_ms, latitude, longitude,
            altitude_m, gps_source, captured_ms
        """
        capture_ms = time.time() * 1000.0 if capture_ms is None else capture_ms
        sample, delta = self.nearest(capture_ms)
        out = {
            "gps_available": False,
            "gps_time_delta_ms": None,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "gps_source": None,
            "captured_ms": capture_ms,
        }
        if sample is not None and delta is not None and delta <= self.max_match_ms:
            out.update(
                gps_available=True,
                gps_time_delta_ms=round(delta, 1),
                latitude=sample.latitude,
                longitude=sample.longitude,
                altitude_m=sample.altitude_m,
                gps_source=sample.source,
            )
        return out

    def stats(self) -> dict:
        with self._lock:
            latest = self._latest
            n = len(self._times)
            received, rejected = self._received, self._rejected
        return {
            "buffered": n,
            "received": received,
            "rejected": rejected,
            "latest": latest.to_dict() if latest else None,
        }

    def clear(self) -> None:
        with self._lock:
            self._times.clear()
            self._samples.clear()
            self._latest = None
