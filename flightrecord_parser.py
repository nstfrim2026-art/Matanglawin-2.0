"""
flightrecord_parser.py - Pluggable parser for aircraft-position telemetry.

See .kiro/specs/dji-gps-geotagging/investigation.md for why this is
pluggable: the raw DJI Fly FlightRecord (`DJIFlightRecord_*.txt`) for the
Neo 2 is AES-encrypted (log v13+) and CANNOT be decoded offline (its
keychain must be fetched from DJI's cloud). So this module:

    * parses an OFFLINE, plaintext **CSV** telemetry export (the format any
      offline decrypter / external GPS logger can produce), and
    * DETECTS the encrypted DJI `.txt` container and returns an honest
      "not decodable offline" result instead of fabricating coordinates.

A "telemetry source" is anything that yields TelemetryPoint samples; the
collector (dji_gps_collector.py) uses whatever parser applies to the file.

Pure standard library (csv, re) + telemetry_store for timestamp parsing.
Robust to partial/garbled lines (a half-written file yields the good rows).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from telemetry_store import parse_timestamp_ms

# Column-name fragments we accept (case-insensitive substring match).
_LAT_KEYS = ("latitude", "gps:lat", "gps_lat", "osd.latitude", "lat")
_LON_KEYS = ("longitude", "gps:long", "gps_long", "gps:lng", "osd.longitude", "lon", "lng")
_ALT_KEYS = ("altitude", "osd.altitude", "height", "alt")
_TIME_KEYS = ("timestamp", "datetime", "updatetime", "time(millisecond)", "time_ms", "time")

DJI_FILENAME_RE = re.compile(r"DJIFlightRecord_.*\.txt$", re.IGNORECASE)
FILENAME_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_ ]\[?(\d{2})-(\d{2})-(\d{2})\]?")


@dataclass
class TelemetryPoint:
    latitude: float
    longitude: float
    altitude_m: Optional[float]
    timestamp_ms: float


@dataclass
class ParseResult:
    decodable: bool
    fmt: str                                  # "csv" | "dji_encrypted" | "unknown"
    filename: str = ""
    points: List[TelemetryPoint] = field(default_factory=list)
    reason: Optional[str] = None

    @property
    def latest(self) -> Optional[TelemetryPoint]:
        return max(self.points, key=lambda p: p.timestamp_ms) if self.points else None


def base_time_ms_from_filename(name: str) -> Optional[float]:
    """DJIFlightRecord_2026-08-18_[15-26-18].txt -> epoch ms of that instant."""
    m = FILENAME_TIME_RE.search(name)
    if not m:
        return None
    date, hh, mm, ss = m.group(1), m.group(2), m.group(3), m.group(4)
    return parse_timestamp_ms(f"{date}T{hh}:{mm}:{ss}")


def _absolute_ts(raw: str) -> Optional[float]:
    """
    Return epoch-ms if `raw` is an ABSOLUTE timestamp (an ISO/date string or
    a plausible epoch >= 1e9), else None (meaning it is a small relative
    offset to be added to the flight-start base time).
    """
    if not raw:
        return None
    try:
        v = float(raw)
        return parse_timestamp_ms(raw) if v >= 1e9 else None
    except ValueError:
        return parse_timestamp_ms(raw)  # non-numeric -> try as datetime string


def _find_col(header: List[str], keys) -> Optional[int]:
    low = [h.strip().lower() for h in header]
    # exact-ish first, then substring
    for i, h in enumerate(low):
        if h in keys:
            return i
    for i, h in enumerate(low):
        if any(k in h for k in keys):
            return i
    return None


def parse_csv(text: str, filename: str = "") -> ParseResult:
    """
    Parse a plaintext CSV telemetry export into TelemetryPoints. Detects
    latitude/longitude/altitude/time columns by flexible name matching.
    The time column may be absolute (datetime/epoch) or a relative offset
    (seconds or ms) that is added to the base time parsed from the filename.
    Malformed rows are skipped (partial-write safe).
    """
    res = ParseResult(decodable=True, fmt="csv", filename=filename)
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        res.reason = "empty file"
        return res

    i_lat = _find_col(header, _LAT_KEYS)
    i_lon = _find_col(header, _LON_KEYS)
    if i_lat is None or i_lon is None:
        return ParseResult(decodable=False, fmt="unknown", filename=filename,
                           reason="no latitude/longitude columns found")
    i_alt = _find_col(header, _ALT_KEYS)
    i_time = _find_col(header, _TIME_KEYS)
    time_header = header[i_time].lower() if i_time is not None else ""
    time_is_ms = ("milli" in time_header) or time_header.endswith("_ms") or "(ms" in time_header
    base_ms = base_time_ms_from_filename(filename) or 0.0

    for row in reader:
        try:
            lat = float(row[i_lat])
            lon = float(row[i_lon])
        except (IndexError, ValueError):
            continue  # skip garbled / partial row (no usable lat/lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        alt = None
        if i_alt is not None:
            try:
                alt = float(row[i_alt])
            except (IndexError, ValueError):
                alt = None
        ts_ms = None
        if i_time is not None:
            try:
                raw = row[i_time].strip()
            except IndexError:
                raw = ""
            abs_ms = _absolute_ts(raw)
            if abs_ms is not None:
                ts_ms = abs_ms
            elif raw:
                # small numeric => relative offset from the flight-start base.
                try:
                    off = float(raw)
                    ts_ms = base_ms + (off if time_is_ms else off * 1000.0)
                except ValueError:
                    ts_ms = None
        if ts_ms is None:
            ts_ms = base_ms + len(res.points) * 100.0  # assume ~10 Hz ordering
        res.points.append(TelemetryPoint(lat, lon, alt, ts_ms))

    if not res.points:
        res.reason = "no valid rows"
    return res


def _looks_binary(raw: bytes) -> bool:
    if b"\x00" in raw[:4096]:
        return True
    # high proportion of non-text bytes?
    sample = raw[:4096]
    if not sample:
        return False
    text_bytes = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127)
    return (text_bytes / len(sample)) < 0.85


def parse_bytes(raw: bytes, filename: str = "") -> ParseResult:
    """Dispatch on content/filename. Never raises."""
    # Encrypted DJI FlightRecord container -> not decodable offline (honest).
    if DJI_FILENAME_RE.search(filename) and _looks_binary(raw):
        return ParseResult(
            decodable=False, fmt="dji_encrypted", filename=filename,
            reason=("DJI Fly FlightRecord is AES-encrypted (log v13+); its keychain "
                    "must be fetched from DJI's cloud API, so it cannot be decoded "
                    "offline. Provide a decrypted CSV export instead."),
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:  # noqa: BLE001
            return ParseResult(decodable=False, fmt="unknown", filename=filename,
                               reason="not text; unrecognized binary format")
    return parse_csv(text, filename=filename)


def parse_file(path: str) -> ParseResult:
    """Read + parse a telemetry file. Tolerates locked/partial/missing files."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return ParseResult(decodable=False, fmt="unknown", filename=p.name,
                           reason=f"unreadable: {exc}")
    return parse_bytes(raw, filename=p.name)
