"""
exif_metadata.py - Read drone photo geolocation + orientation metadata.

Part of the Drone Crack Geotagging module. Extracts, from a still photo:
    * GPS latitude / longitude / altitude (EXIF), converting DMS -> decimal
      degrees (S and W negative),
    * DJI gimbal yaw (heading) / pitch and relative/absolute altitude from
      the embedded XMP packet,
    * capture timestamp (ISO-8601),
    * image pixel dimensions and focal length when present.

Design notes:
    * Pure helpers (dms_to_decimal) never need any third-party library.
    * GPS reading tries `exifread` first, then Pillow; if neither is
      installed the module still imports and returns has_gps=False.
    * XMP is parsed straight from the file bytes with small regexes, so
      DJI gimbal/altitude fields need no extra dependency.
    * Nothing here ever raises on missing/garbled metadata - callers get a
      structured `ImageGeoMetadata` with has_gps=False and can degrade.

This module is used only by the offline/Colab geotagging path; it does not
touch the live web UI, the POV pipeline, or the import watcher.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pure helper: degrees-minutes-seconds -> signed decimal degrees.
# ---------------------------------------------------------------------------
def dms_to_decimal(degrees: float, minutes: float, seconds: float, ref: Optional[str]) -> float:
    """
    Convert DMS to signed decimal degrees. `ref` is the hemisphere letter
    ('N'/'S' for latitude, 'E'/'W' for longitude); 'S' and 'W' yield a
    negative result. Pure Python - always available.
    """
    dec = abs(float(degrees)) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if ref and str(ref).strip().upper() in ("S", "W"):
        dec = -dec
    return dec


@dataclass
class ImageGeoMetadata:
    """Structured geolocation/orientation metadata for one photo."""
    image_name: str = ""
    has_gps: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None          # EXIF GPS altitude (AMSL, if present)
    rel_altitude_m: Optional[float] = None       # DJI RelativeAltitude (above take-off)
    abs_altitude_m: Optional[float] = None       # DJI AbsoluteAltitude
    gimbal_yaw_deg: Optional[float] = None       # heading of image "up"
    gimbal_pitch_deg: Optional[float] = None     # ~ -90 for nadir
    flight_yaw_deg: Optional[float] = None
    timestamp: Optional[str] = None              # ISO-8601
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None

    def best_altitude(self) -> Optional[float]:
        """Preferred altitude for GSD: relative (above ground) if available."""
        for a in (self.rel_altitude_m, self.altitude_m, self.abs_altitude_m):
            if a is not None:
                return a
        return None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# GPS via exifread (preferred) then Pillow.
# ---------------------------------------------------------------------------
def _ratio_to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        # exifread Ratio has .num/.den
        num = getattr(x, "num", None)
        den = getattr(x, "den", None)
        if num is not None and den:
            return float(num) / float(den)
        raise


def _gps_from_exifread(path: Path, meta: ImageGeoMetadata) -> bool:
    try:
        import exifread  # type: ignore
    except Exception:  # noqa: BLE001 - not installed
        return False
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:  # noqa: BLE001
        return False
    if not tags:
        return False

    def dms(tag):
        vals = tags[tag].values
        return [_ratio_to_float(v) for v in vals]

    try:
        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
            la = dms("GPS GPSLatitude")
            lo = dms("GPS GPSLongitude")
            la_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
            lo_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
            meta.latitude = dms_to_decimal(la[0], la[1], la[2], la_ref)
            meta.longitude = dms_to_decimal(lo[0], lo[1], lo[2], lo_ref)
            meta.has_gps = True
    except Exception:  # noqa: BLE001
        return meta.has_gps

    try:
        if "GPS GPSAltitude" in tags:
            alt = _ratio_to_float(tags["GPS GPSAltitude"].values[0])
            ref = tags.get("GPS GPSAltitudeRef")
            if ref is not None and str(ref.values) in ("1", "[1]"):
                alt = -alt  # below sea level
            meta.altitude_m = alt
    except Exception:  # noqa: BLE001
        pass

    try:
        dto = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if dto is not None:
            meta.timestamp = _exif_datetime_to_iso(str(dto))
    except Exception:  # noqa: BLE001
        pass

    try:
        fl = tags.get("EXIF FocalLength")
        if fl is not None:
            meta.focal_length_mm = _ratio_to_float(fl.values[0])
    except Exception:  # noqa: BLE001
        pass

    return meta.has_gps


def _gps_from_pillow(path: Path, meta: ImageGeoMetadata) -> bool:
    try:
        from PIL import Image, ExifTags  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        img = Image.open(path)
        meta.image_width, meta.image_height = img.size
        exif = img.getexif()
    except Exception:  # noqa: BLE001
        return False
    if not exif:
        return meta.has_gps

    tag_by_name = {v: k for k, v in ExifTags.TAGS.items()}
    gps_ifd_tag = tag_by_name.get("GPSInfo")

    # timestamp / focal length from the main IFD
    try:
        dto = exif.get(tag_by_name.get("DateTimeOriginal")) or exif.get(tag_by_name.get("DateTime"))
        if dto:
            meta.timestamp = _exif_datetime_to_iso(str(dto))
        fl = exif.get(tag_by_name.get("FocalLength"))
        if fl is not None:
            meta.focal_length_mm = float(fl)
    except Exception:  # noqa: BLE001
        pass

    try:
        gps = exif.get_ifd(gps_ifd_tag) if gps_ifd_tag else None
    except Exception:  # noqa: BLE001
        gps = None
    if not gps:
        return meta.has_gps

    gtags = ExifTags.GPSTAGS
    g = {gtags.get(k, k): v for k, v in gps.items()}
    try:
        if "GPSLatitude" in g and "GPSLongitude" in g:
            la = [float(v) for v in g["GPSLatitude"]]
            lo = [float(v) for v in g["GPSLongitude"]]
            meta.latitude = dms_to_decimal(la[0], la[1], la[2], g.get("GPSLatitudeRef", "N"))
            meta.longitude = dms_to_decimal(lo[0], lo[1], lo[2], g.get("GPSLongitudeRef", "E"))
            meta.has_gps = True
        if "GPSAltitude" in g:
            alt = float(g["GPSAltitude"])
            if str(g.get("GPSAltitudeRef", 0)) in ("1", "b'\\x01'"):
                alt = -alt
            meta.altitude_m = alt
    except Exception:  # noqa: BLE001
        pass
    return meta.has_gps


# ---------------------------------------------------------------------------
# DJI XMP (gimbal + altitude) parsed straight from the file bytes.
# ---------------------------------------------------------------------------
_XMP_FIELDS = {
    "gimbal_yaw_deg": r"GimbalYawDegree",
    "gimbal_pitch_deg": r"GimbalPitchDegree",
    "flight_yaw_deg": r"FlightYawDegree",
    "rel_altitude_m": r"RelativeAltitude",
    "abs_altitude_m": r"AbsoluteAltitude",
}


def parse_xmp(raw: bytes, meta: ImageGeoMetadata) -> ImageGeoMetadata:
    """
    Extract DJI drone fields from an XMP packet (raw file or XMP bytes).
    Values may be attributes (`drone-dji:GimbalYawDegree="+1.2"`) or
    elements (`<drone-dji:GimbalYawDegree>+1.2</...>`). Robust to either.
    """
    try:
        text = raw.decode("latin-1", errors="ignore")
    except Exception:  # noqa: BLE001
        return meta
    num = r'([+-]?\d+(?:\.\d+)?)'
    for field, name in _XMP_FIELDS.items():
        m = re.search(name + r'\s*=\s*"' + num + r'"', text)
        if not m:
            m = re.search(r'<[\w:.\-]*' + name + r'\s*>\s*' + num, text)
        if m:
            try:
                setattr(meta, field, float(m.group(1)))
            except (TypeError, ValueError):
                pass
    return meta


def _exif_datetime_to_iso(s: str) -> Optional[str]:
    s = s.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    return s or None


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def read_metadata(image_path: str) -> ImageGeoMetadata:
    """
    Read all available geolocation/orientation metadata for one image.
    Never raises: on any problem the returned object simply has has_gps=False
    and None fields.
    """
    path = Path(image_path)
    meta = ImageGeoMetadata(image_name=path.name)

    # GPS: exifread preferred, Pillow fallback.
    got = False
    try:
        got = _gps_from_exifread(path, meta)
    except Exception:  # noqa: BLE001
        got = False
    if not got:
        try:
            _gps_from_pillow(path, meta)
        except Exception:  # noqa: BLE001
            pass

    # Pixel dimensions (if Pillow didn't already set them).
    if meta.image_width is None:
        try:
            import cv2  # type: ignore
            img = cv2.imread(str(path))
            if img is not None:
                meta.image_height, meta.image_width = img.shape[:2]
        except Exception:  # noqa: BLE001
            pass

    # DJI XMP gimbal/altitude from raw bytes.
    try:
        raw = path.read_bytes()
        parse_xmp(raw, meta)
    except Exception:  # noqa: BLE001
        pass

    return meta
