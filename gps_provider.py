"""
gps_provider.py - GPS provider abstraction for inspection geolocation.

The DJI Neo 2 does not expose GPS telemetry via RTMP stream. This module
provides a clean abstraction that:
  - Returns None values for live GPS (since real-time GPS is unavailable)
  - Supports importing GPS logs after a flight
  - Matches GPS points to inspection timestamps using nearest-neighbor search

GPS data is primarily a report/export feature, not a live dashboard feature.
"""

from __future__ import annotations

import bisect
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Default tolerance in seconds for timestamp-based GPS matching
DEFAULT_GPS_TOLERANCE_SECONDS = 5.0


class GPSProvider:
    """
    GPS provider abstraction for crack inspection geolocation.

    Since the DJI Neo 2 does not expose GPS via RTMP, live GPS is unavailable.
    GPS data can be imported after a flight and matched to inspections by timestamp.
    """

    def __init__(self, tolerance_seconds: float = DEFAULT_GPS_TOLERANCE_SECONDS):
        """
        Initialize the GPS provider.

        Args:
            tolerance_seconds: Maximum time difference (in seconds) for
                timestamp-based GPS matching. Points further than this
                threshold will not be matched.
        """
        self._tolerance_seconds = tolerance_seconds
        self._gps_log: List[Dict[str, Any]] = []
        self._timestamps: List[float] = []  # sorted epoch timestamps for bisect

    def get_current_position(self) -> Dict[str, Any]:
        """
        Get the current GPS position.

        Since real-time GPS is not available from DJI Neo 2 via RTMP,
        this always returns None values. Never returns fake coordinates.

        Returns:
            Dict with latitude, longitude, altitude, timestamp - all None.
        """
        return {
            "latitude": None,
            "longitude": None,
            "altitude": None,
            "timestamp": None,
        }

    def import_gps_log(self, log_data: List[Dict[str, Any]]) -> int:
        """
        Import GPS telemetry data from a flight log.

        Each entry must have at minimum:
          - timestamp: ISO 8601 string or datetime object
          - latitude: float
          - longitude: float

        Optional fields:
          - altitude: float (meters)

        Args:
            log_data: List of GPS point dicts.

        Returns:
            Number of points successfully imported.
        """
        imported = 0
        points = []

        for entry in log_data:
            ts = entry.get("timestamp")
            lat = entry.get("latitude")
            lon = entry.get("longitude")
            alt = entry.get("altitude")

            if ts is None or lat is None or lon is None:
                continue

            # Parse timestamp to epoch seconds
            epoch = self._parse_timestamp(ts)
            if epoch is None:
                continue

            points.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude": float(alt) if alt is not None else None,
                "timestamp": ts if isinstance(ts, str) else ts.isoformat(),
                "_epoch": epoch,
            })
            imported += 1

        # Sort by epoch timestamp
        points.sort(key=lambda p: p["_epoch"])

        # Store sorted log and epoch list for bisect
        self._gps_log = points
        self._timestamps = [p["_epoch"] for p in points]

        return imported

    def get_position_at_timestamp(self, timestamp) -> Dict[str, Any]:
        """
        Find the nearest GPS point to a given timestamp.

        Uses binary search (bisect) for efficient lookup. Returns the nearest
        point only if it is within the configured tolerance.

        Args:
            timestamp: ISO 8601 string, datetime object, or epoch float.

        Returns:
            Dict with latitude, longitude, altitude, timestamp if a match
            is found within tolerance. Otherwise returns None values.
        """
        if not self._gps_log:
            return self.get_current_position()

        epoch = self._parse_timestamp(timestamp)
        if epoch is None:
            return self.get_current_position()

        # Binary search for nearest point
        idx = bisect.bisect_left(self._timestamps, epoch)

        # Check candidates: idx-1 and idx
        best_point = None
        best_diff = float("inf")

        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(self._timestamps):
                diff = abs(self._timestamps[candidate_idx] - epoch)
                if diff < best_diff:
                    best_diff = diff
                    best_point = self._gps_log[candidate_idx]

        # Check tolerance
        if best_point is not None and best_diff <= self._tolerance_seconds:
            return {
                "latitude": best_point["latitude"],
                "longitude": best_point["longitude"],
                "altitude": best_point["altitude"],
                "timestamp": best_point["timestamp"],
            }

        # No match within tolerance
        return self.get_current_position()

    @property
    def tolerance_seconds(self) -> float:
        """Get the current GPS matching tolerance in seconds."""
        return self._tolerance_seconds

    @tolerance_seconds.setter
    def tolerance_seconds(self, value: float) -> None:
        """Set the GPS matching tolerance in seconds."""
        self._tolerance_seconds = max(0.0, float(value))

    @property
    def log_size(self) -> int:
        """Return the number of GPS points currently loaded."""
        return len(self._gps_log)

    def _parse_timestamp(self, ts) -> Optional[float]:
        """
        Parse a timestamp into epoch seconds (float).

        Accepts:
          - float/int (treated as epoch seconds)
          - datetime object (timezone-aware treated as-is; naive treated as UTC)
          - ISO 8601 string (with or without timezone; naive assumed UTC)

        Returns:
            Epoch seconds as float, or None if parsing fails.
        """
        if isinstance(ts, (int, float)):
            return float(ts)

        if isinstance(ts, datetime):
            # If naive (no tzinfo), assume UTC to match capture timestamps
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp()

        if isinstance(ts, str):
            # Try with timezone info using fromisoformat first (Python 3.7+)
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                # fromisoformat preserves timezone; if parsed as naive, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except (ValueError, AttributeError):
                pass

            # Fall back to strptime for common ISO formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(ts, fmt)
                    # strptime produces naive datetimes; assume UTC
                    dt = dt.replace(tzinfo=timezone.utc)
                    return dt.timestamp()
                except ValueError:
                    continue

        return None
