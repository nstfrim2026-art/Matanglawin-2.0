"""
srt_telemetry.py - Parse DJI Neo 2 SRT (Video Subtitles) telemetry, OFFLINE.

DJI Fly's "Video Subtitles" feature writes a plaintext .SRT alongside the
recorded video, with a timestamped telemetry line per frame. Unlike the
encrypted FlightRecord, the SRT is plaintext, so it can be parsed fully
offline - which is why it is the authoritative aircraft-position source for
MatanglaWIN's geotagging (see .kiro/specs/dji-gps-geotagging).

We extract ONLY what geotagging needs: latitude, longitude, timestamp.
Altitude/heading/speed are intentionally ignored.

DJI's SRT payload has varied across models/firmware; this parser handles the
common variants robustly:

    * bracketed:  ... [latitude: 7.123456] [longitude: 125.654321] ...
    * loose:      latitude: 7.123456  longitude: 125.654321
    * GPS():      GPS(125.654321,7.123456,18)     # DJI order = (lon,lat,sats)

Each SRT block also carries a subtitle time range (00:00:00,033) and usually
an absolute wall-clock datetime (2026-08-21 15:42:18.300). We prefer the
absolute datetime; otherwise we use a base time (from the filename or file
mtime) plus the block's relative offset.

Robustness: a malformed/partial/half-written block is skipped, never fatal.

References (implementation guidance): jetervaz/dji-telemetry, bhanurp/
dji-flight, and the optional `dji-telemetry` PyPI package. This module is a
compact, dependency-free equivalent so the app stays offline and testable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# --- regexes (case-insensitive where it matters) ---------------------------
_TIME_RANGE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
_LAT = re.compile(r"latitude\s*[:=]\s*([+-]?\d{1,3}\.\d+)", re.IGNORECASE)
_LON = re.compile(r"longitude\s*[:=]\s*([+-]?\d{1,3}\.\d+)", re.IGNORECASE)
# DJI GPS(lon, lat, sats) - longitude FIRST.
_GPS = re.compile(r"GPS\s*\(?\s*([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,3}\.\d+)", re.IGNORECASE)
_DT = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?")
_FNAME_DT = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_ ]?(\d{2})?(\d{2})?(\d{2})?")


@dataclass
class SrtSample:
    timestamp_ms: float
    latitude: float
    longitude: float


def _valid_latlon(lat: float, lon: float) -> bool:
    return (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
            and not (abs(lat) < 1e-7 and abs(lon) < 1e-7))


def srt_tz_offset_min() -> Optional[int]:
    """
    Optional fixed timezone for naive SRT datetimes, in minutes east of UTC,
    from MATANGLAWIN_SRT_TZ_OFFSET_MIN. If unset, naive SRT times are
    interpreted in the machine's LOCAL timezone (drone + laptop share it in
    the field). Handles the "timezone ambiguity" requirement explicitly.
    """
    raw = os.environ.get("MATANGLAWIN_SRT_TZ_OFFSET_MIN")
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _dt_to_ms(match, tz_offset_min: Optional[int]) -> Optional[float]:
    try:
        y, mo, d, h, mi, s = (int(match.group(i)) for i in range(1, 7))
        frac = match.group(7)
        micro = 0
        if frac:
            micro = int((frac + "000000")[:6])
        naive = datetime(y, mo, d, h, mi, s, micro)
    except (ValueError, TypeError):
        return None
    if tz_offset_min is None:
        # Interpret as local wall-clock time.
        return naive.timestamp() * 1000.0
    aware = naive.replace(tzinfo=timezone(timedelta(minutes=tz_offset_min)))
    return aware.timestamp() * 1000.0


def base_time_ms_from_filename(name: str) -> Optional[float]:
    """Derive a recording start time from a DJI filename, else None."""
    m = _FNAME_DT.search(name)
    if not m:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        h = int(m.group(4) or 0); mi = int(m.group(5) or 0); s = int(m.group(6) or 0)
        return datetime(y, mo, d, h, mi, s).timestamp() * 1000.0
    except (ValueError, TypeError):
        return None


def _extract_latlon(text: str):
    """Return (lat, lon) from an SRT block payload, or None."""
    mlat, mlon = _LAT.search(text), _LON.search(text)
    if mlat and mlon:
        lat, lon = float(mlat.group(1)), float(mlon.group(1))
        if _valid_latlon(lat, lon):
            return lat, lon
    mg = _GPS.search(text)
    if mg:
        a, b = float(mg.group(1)), float(mg.group(2))
        # DJI GPS() is (lon, lat); accept, and swap if that fails validation.
        if _valid_latlon(b, a):
            return b, a
        if _valid_latlon(a, b):
            return a, b
    return None


def parse_srt(text: str, base_time_ms: Optional[float] = None,
              tz_offset_min: Optional[int] = None) -> List[SrtSample]:
    """
    Parse SRT text into time-ordered SrtSamples (lat/lon/timestamp only).
    Malformed or GPS-less blocks are skipped; never raises.
    """
    samples: List[SrtSample] = []
    # Split into blocks on blank lines (tolerant of \r\n and extra spaces).
    blocks = re.split(r"\r?\n[ \t]*\r?\n", text)
    idx = 0
    for block in blocks:
        if not block.strip():
            continue
        idx += 1
        try:
            latlon = _extract_latlon(block)
            if latlon is None:
                continue
            lat, lon = latlon
            ts_ms = None
            mdt = _DT.search(block)
            if mdt:
                ts_ms = _dt_to_ms(mdt, tz_offset_min)
            if ts_ms is None:
                mtr = _TIME_RANGE.search(block)
                if mtr:
                    off = (int(mtr.group(1)) * 3600 + int(mtr.group(2)) * 60
                           + int(mtr.group(3))) * 1000 + int(mtr.group(4).ljust(3, "0"))
                    ts_ms = (base_time_ms or 0.0) + off
            if ts_ms is None:
                ts_ms = (base_time_ms or 0.0) + idx * 33.0  # ~30 fps fallback
            samples.append(SrtSample(ts_ms, lat, lon))
        except Exception:  # noqa: BLE001 - one bad block never breaks the file
            continue
    samples.sort(key=lambda s: s.timestamp_ms)
    return samples


def parse_file(path: str, tz_offset_min: Optional[int] = None) -> List[SrtSample]:
    """Read + parse an SRT file. Tolerates locked/partial/missing files."""
    p = Path(path)
    if tz_offset_min is None:
        tz_offset_min = srt_tz_offset_min()
    try:
        raw = p.read_bytes()
    except OSError:
        return []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    base = base_time_ms_from_filename(p.name)
    if base is None:
        try:
            base = p.stat().st_mtime * 1000.0
        except OSError:
            base = None
    return parse_srt(text, base_time_ms=base, tz_offset_min=tz_offset_min)


def latest_sample(samples: List[SrtSample]) -> Optional[SrtSample]:
    return samples[-1] if samples else None
