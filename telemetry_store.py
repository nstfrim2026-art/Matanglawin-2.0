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

# How close (ms) a telemetry sample must be to the capture time to be trusted
# as an EXACT match (the highest-quality association).
DEFAULT_MAX_MATCH_MS = 2000.0
# Maximum age (ms) of the newest valid sample for it to still be usable as a
# fallback ("latest known position"). Configurable via PHONE_GPS_MAX_AGE_SECONDS.
# This is the window that decides gps_usable at capture time (requirement:
# "PHONE_GPS_MAX_AGE_SECONDS=15"): if the freshest fix is <= this old, GPS is
# usable; older than this, GPS is unavailable (but capture/analysis still run).
DEFAULT_MAX_AGE_MS = 15000.0
# How old (ms) the newest sample may be before the source is "stale". Kept in
# lockstep with the max-age window so the live "current drone position" dot and
# the capture-association fallback agree on what "fresh" means.
DEFAULT_STALE_MS = DEFAULT_MAX_AGE_MS
DEFAULT_BUFFER = 6000  # ~10 min at 10 Hz


def phone_gps_max_age_ms() -> float:
    """
    Maximum age (milliseconds) a phone-GPS fix may have and still be attached
    to a capture, from PHONE_GPS_MAX_AGE_SECONDS (default 15 s). Single source
    of truth so the staleness window is never hardcoded in multiple files.
    """
    import os
    try:
        secs = float(os.environ.get("PHONE_GPS_MAX_AGE_SECONDS", DEFAULT_MAX_AGE_MS / 1000.0))
        return (secs * 1000.0) if secs > 0 else DEFAULT_MAX_AGE_MS
    except (TypeError, ValueError):
        return DEFAULT_MAX_AGE_MS


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
        max_age_ms: float = DEFAULT_MAX_AGE_MS,
        persist_path: Optional[str] = None,
    ):
        self.max_samples = max_samples
        self.max_match_ms = max_match_ms
        self.stale_ms = stale_ms
        # Fallback window: how old the freshest fix may be and still be used as
        # a capture's "latest known position" when no sample sits inside the
        # tight max_match_ms window. This is what makes association robust
        # instead of requiring near-exact timestamp equality.
        self.max_age_ms = max_age_ms
        self.persist_path = persist_path
        self._lock = threading.Lock()
        self._times: List[float] = []                 # sorted epoch-ms
        self._samples: List[TelemetrySample] = []      # parallel to _times
        self._latest: Optional[TelemetrySample] = None
        self._received = 0
        self._rejected = 0
        self._load_latest()

    # -- local persistence of the latest sample (offline, best-effort) --

    def _load_latest(self) -> None:
        if not self.persist_path:
            return
        try:
            import json
            with open(self.persist_path, encoding="utf-8") as f:
                d = json.load(f)
            self._latest = TelemetrySample(
                float(d["latitude"]), float(d["longitude"]),
                d.get("altitude_m"), float(d["timestamp_ms"]),
                d.get("source", "phone_gps"),
            )
        except Exception:  # noqa: BLE001 - missing/corrupt file is fine
            pass

    def _save_latest(self) -> None:
        if not self.persist_path or self._latest is None:
            return
        try:
            import json
            from pathlib import Path as _P
            _P(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({
                    "latitude": self._latest.latitude,
                    "longitude": self._latest.longitude,
                    "altitude_m": self._latest.altitude_m,
                    "timestamp_ms": self._latest.timestamp_ms,
                    "source": self._latest.source,
                }, f)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

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
            new_latest = self._latest is None or ts >= self._latest.timestamp_ms
            if new_latest:
                self._latest = sample
            self._received += 1
        if new_latest:
            self._save_latest()
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

    def match_for_capture(
        self,
        capture_ms: Optional[float] = None,
        use_stale_fallback: bool = False,
        max_age_ms: Optional[float] = None,
    ) -> dict:
        """
        Robust association result for a capture. Never raises, never fabricates
        coordinates, and never defaults to (0,0).

        Association strategy (requirement: "robust association, not exact
        timestamp matching"):

          1. Take the sample NEAREST in time to ``capture_ms``. If it is within
             ``max_match_ms`` of the capture time, use it (highest quality,
             ``gps_match="nearest"``).
          2. Otherwise, if ``use_stale_fallback`` is set, fall back to the
             LATEST valid sample as long as it is at/just-before the capture
             time and no older than ``max_age_ms`` (``gps_match="latest"``).
             This is the "latest usable phone-GPS position" used by a live
             capture, so a slightly-out-of-sync clock never yields
             "Not recorded".
          3. Otherwise there is genuinely no usable fix -> ``gps_available``
             is False (the image is still captured and still analyzed).

        The stale fallback is deliberately opt-in: the SRT geotagging backfill
        keeps strict nearest-only matching (it must not attach a fix that is
        far from the frame's own timestamp), while the live phone-GPS capture
        path enables it.

        Returns a dict with:
            gps_available, gps_time_delta_ms, latitude, longitude,
            altitude_m, gps_source, captured_ms, gps_match
        """
        capture_ms = time.time() * 1000.0 if capture_ms is None else capture_ms
        max_age = self.max_age_ms if max_age_ms is None else max_age_ms
        out = {
            "gps_available": False,
            "gps_time_delta_ms": None,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "gps_source": None,
            "captured_ms": capture_ms,
            "gps_match": "none",
        }

        # 1) Nearest sample inside the tight match window (best quality).
        sample, delta = self.nearest(capture_ms)
        if sample is not None and delta is not None and delta <= self.max_match_ms:
            out.update(
                gps_available=True,
                gps_time_delta_ms=round(delta, 1),
                latitude=sample.latitude,
                longitude=sample.longitude,
                altitude_m=sample.altitude_m,
                gps_source=sample.source,
                gps_match="nearest",
            )
            return out

        # 2) Fallback: latest valid, non-excessively-stale position.
        if use_stale_fallback:
            with self._lock:
                latest = self._latest
            if latest is not None:
                age = capture_ms - latest.timestamp_ms
                # Sample must be at/before the capture (a future sample is not a
                # "last known position") and within the configured max age.
                if 0 <= age <= max_age:
                    out.update(
                        gps_available=True,
                        gps_time_delta_ms=round(age, 1),
                        latitude=latest.latitude,
                        longitude=latest.longitude,
                        altitude_m=latest.altitude_m,
                        gps_source=latest.source,
                        gps_match="latest",
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
