"""
geotag_engine.py - Turn pixel-space crack segmentations into WGS84 GeoJSON.

This is the backend of the Drone Crack Geotagging & Spatial Mapping module.
It ties together:
    * segmentation      - reused from inference_core.predict_masks (best.pt),
                          so there is NO second model implementation;
    * metadata          - exif_metadata.read_metadata (GPS + gimbal + alt);
    * georeferencing     - either an orthomosaic GeoTIFF geo-transform
                          (rasterio, most accurate) or a Ground-Sample-
                          Distance transform from flight altitude + camera
                          specs (nadir/flat-ground approximation);
    * export             - standard GeoJSON FeatureCollection with per-crack
                          area / length / confidence / image / timestamp.

It is headless and offline (Colab / scripts). It does not import Flask and
does not touch the live web UI, the display-only POV pipeline, or the
photo-import watcher.

Dependency policy: numpy + OpenCV + ultralytics (via inference_core) are the
only hard requirements. `pyproj` (geodesic area / reprojection) and
`rasterio` (orthomosaic) are optional and imported lazily; the pure-math
helpers below work with none of them installed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import exif_metadata

# Metres per degree of latitude (WGS84 mean); longitude scaled by cos(lat).
_M_PER_DEG = 111320.0
DEFAULT_WEIGHTS = "best.pt"


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
@dataclass
class CameraModel:
    """
    Physical camera parameters needed for the GSD transform. Defaults are a
    generic small-sensor drone placeholder - override with your aircraft's
    real sensor width and focal length for accurate scaling.
    """
    sensor_width_mm: Optional[float] = 6.4
    focal_length_mm: Optional[float] = 4.7
    name: str = "generic-drone"


# A rough DJI Neo 2-class default; adjust to the real datasheet values.
DEFAULT_CAMERA = CameraModel(sensor_width_mm=6.4, focal_length_mm=4.7, name="dji-neo2-default")


# ---------------------------------------------------------------------------
# GSD geometry (pure math; nadir + flat-ground approximation)
# ---------------------------------------------------------------------------
def ground_sample_distance(
    sensor_width_mm: float, focal_length_mm: float, image_width_px: int, altitude_m: float
) -> float:
    """Metres of ground covered per image pixel."""
    if focal_length_mm <= 0 or image_width_px <= 0:
        raise ValueError("focal_length_mm and image_width_px must be > 0")
    return (float(sensor_width_mm) * float(altitude_m)) / (
        float(focal_length_mm) * float(image_width_px)
    )


def pixel_offset_to_ground(
    dx_px: float, dy_px: float, gsd: float, yaw_deg: float = 0.0
) -> Tuple[float, float]:
    """
    Convert a pixel offset from the image centre into a ground offset
    (east_m, north_m), rotating by the gimbal yaw (heading of image "up").
    Image row increases downward, so "up" in the image is +north before the
    yaw rotation.
    """
    east0 = dx_px * gsd
    north0 = -dy_px * gsd
    theta = math.radians(yaw_deg or 0.0)
    east = east0 * math.cos(theta) + north0 * math.sin(theta)
    north = -east0 * math.sin(theta) + north0 * math.cos(theta)
    return east, north


def offset_to_latlon(lat: float, lon: float, east_m: float, north_m: float) -> Tuple[float, float]:
    """Local equirectangular approximation: metres -> WGS84 lat/lon."""
    dlat = north_m / _M_PER_DEG
    dlon = east_m / (_M_PER_DEG * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def pixel_polygon_to_lonlat(
    poly_px: Sequence[Sequence[float]],
    center_px: Tuple[float, float],
    gsd: float,
    yaw_deg: float,
    lat: float,
    lon: float,
) -> List[List[float]]:
    """
    Convert a pixel polygon (ring of (col,row)) to a closed WGS84 ring in
    GeoJSON order [[lon,lat], ...] (first point repeated as last).
    """
    cx, cy = center_px
    ring: List[List[float]] = []
    for (col, row) in poly_px:
        east, north = pixel_offset_to_ground(col - cx, row - cy, gsd, yaw_deg)
        plat, plon = offset_to_latlon(lat, lon, east, north)
        ring.append([plon, plat])
    if ring and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return ring


# ---------------------------------------------------------------------------
# Mask / polygon geometry
# ---------------------------------------------------------------------------
def mask_to_polygons(mask: np.ndarray, min_area_px: int = 40, epsilon_frac: float = 0.002) -> List[np.ndarray]:
    """
    Extract simplified pixel polygons (Nx2 int arrays of (col,row)) from a
    binary mask via OpenCV contours. Drops sub-threshold specks and smooths
    with approxPolyDP.
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append(approx.astype(float))
    return polys


def polygon_pixel_area(poly_px: Sequence[Sequence[float]]) -> float:
    return float(abs(cv2.contourArea(np.asarray(poly_px, dtype=np.float32))))


def polygon_length_m(poly_px: Sequence[Sequence[float]], gsd: float) -> float:
    """Estimate crack length as the min-area-rect long side (px) x GSD."""
    pts = np.asarray(poly_px, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(max(w, h) * gsd)


def polygon_area_m2(ring_lonlat: Sequence[Sequence[float]], fallback_px_area: float = 0.0,
                    gsd: float = 0.0) -> Tuple[float, str]:
    """
    Geodesic area of a WGS84 ring via pyproj when available; otherwise fall
    back to pixel_area * GSD^2. Returns (area_m2, method).
    """
    try:
        from pyproj import Geod  # type: ignore
        lons = [p[0] for p in ring_lonlat]
        lats = [p[1] for p in ring_lonlat]
        area, _ = Geod(ellps="WGS84").polygon_area_perimeter(lons, lats)
        return abs(float(area)), "geodesic"
    except Exception:  # noqa: BLE001 - pyproj absent or degenerate ring
        return float(fallback_px_area) * float(gsd) ** 2, "gsd_pixel"


# ---------------------------------------------------------------------------
# Orthomosaic (optional rasterio)
# ---------------------------------------------------------------------------
class OrthomosaicReferencer:
    """
    Map pixel (col,row) of an orthomosaic GeoTIFF to WGS84 lon/lat using the
    file's own affine transform (and CRS reprojection if needed). Requires
    `rasterio`; raises a clear error if it is missing.
    """

    def __init__(self, geotiff_path: str):
        try:
            import rasterio  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Orthomosaic georeferencing needs the optional 'rasterio' package. "
                "Install it (pip install rasterio) or use the GSD/EXIF method instead."
            ) from exc
        self._rasterio = rasterio
        self.path = geotiff_path
        self._ds = rasterio.open(geotiff_path)
        self.transform = self._ds.transform
        self.crs = self._ds.crs
        self._to_wgs84 = None
        try:
            if self.crs and self.crs.to_epsg() != 4326:
                from pyproj import Transformer  # type: ignore
                self._to_wgs84 = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        except Exception:  # noqa: BLE001
            self._to_wgs84 = None

    def pixel_to_lonlat(self, col: float, row: float) -> Tuple[float, float]:
        x, y = self.transform * (col, row)  # map coords in the file CRS
        if self._to_wgs84 is not None:
            lon, lat = self._to_wgs84.transform(x, y)
            return float(lon), float(lat)
        return float(x), float(y)  # already lon/lat (EPSG:4326)

    def ring_to_lonlat(self, poly_px: Sequence[Sequence[float]]) -> List[List[float]]:
        ring = [list(self.pixel_to_lonlat(col, row)) for (col, row) in poly_px]
        if ring and ring[0] != ring[-1]:
            ring.append(list(ring[0]))
        return ring


# ---------------------------------------------------------------------------
# GeoJSON builders
# ---------------------------------------------------------------------------
def feature(ring_lonlat: Optional[List[List[float]]], properties: dict) -> dict:
    geometry = None
    if ring_lonlat:
        geometry = {"type": "Polygon", "coordinates": [ring_lonlat]}
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def feature_collection(features: List[dict], meta: Optional[dict] = None) -> dict:
    fc = {"type": "FeatureCollection", "features": list(features)}
    if meta:
        fc["properties"] = meta
    return fc


def merge_collections(collections: Sequence[dict]) -> dict:
    feats: List[dict] = []
    images: List[str] = []
    for fc in collections:
        feats.extend(fc.get("features", []))
        name = (fc.get("properties") or {}).get("image_name")
        if name:
            images.append(name)
    return feature_collection(feats, {"images": images, "num_cracks": len(feats),
                                      "crs": "EPSG:4326"})


def write_geojson(fc: dict, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Detection helper (reuses the shared inference - no duplicate model code)
# ---------------------------------------------------------------------------
def _detect_polygons(image_path: str, weights: str, conf: float, imgsz: int):
    """Return (list_of_pixel_polygons, list_of_confidences, (h, w))."""
    import inference_core  # local import so this module imports without torch

    result = inference_core.predict_masks(image_path, weights=weights, conf=conf, imgsz=imgsz)
    h, w = result.orig_img.shape[:2]
    polys: List[np.ndarray] = []
    confs: List[float] = []
    if result.masks is None or len(result.masks) == 0:
        return polys, confs, (h, w)

    xy = result.masks.xy  # list of Nx2 float arrays (pixel coords)
    conf_arr = None
    if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
        conf_arr = result.boxes.conf.cpu().numpy()
    for i, poly in enumerate(xy):
        poly = np.asarray(poly, dtype=float)
        if poly.ndim != 2 or len(poly) < 3:
            continue
        polys.append(poly)
        confs.append(float(conf_arr[i]) if conf_arr is not None and i < len(conf_arr) else None)
    return polys, confs, (h, w)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def geotag_image(
    image_path: str,
    weights: str = DEFAULT_WEIGHTS,
    camera: CameraModel = DEFAULT_CAMERA,
    altitude_m: Optional[float] = None,
    orthomosaic: Optional[str] = None,
    conf: float = 0.20,
    imgsz: int = 1280,
    min_area_px: int = 40,
) -> dict:
    """
    Detect cracks in one image and return a GeoJSON FeatureCollection.

    Georeferencing method is chosen automatically:
      1. orthomosaic GeoTIFF (if supplied)  -> most accurate,
      2. GSD + EXIF (GPS + altitude + camera specs),
      3. non-georeferenced (polygons kept in pixel space, flagged).
    Never raises on a per-image problem that can be represented as
    "not georeferenced".
    """
    name = Path(image_path).name
    polys_px, confs, (h, w) = _detect_polygons(image_path, weights, conf, imgsz)

    method = "none"
    georeferenced = False
    reason = None
    gsd = None
    referencer = None
    meta = exif_metadata.read_metadata(image_path)

    if orthomosaic:
        try:
            referencer = OrthomosaicReferencer(orthomosaic)
            method, georeferenced = "orthomosaic", True
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
    if not georeferenced:
        alt = altitude_m if altitude_m is not None else meta.best_altitude()
        focal = camera.focal_length_mm or meta.focal_length_mm
        sensor = camera.sensor_width_mm or meta.sensor_width_mm
        if meta.has_gps and alt and focal and sensor and meta.image_width:
            gsd = ground_sample_distance(sensor, focal, meta.image_width or w, alt)
            method, georeferenced = "gsd", True
        elif orthomosaic is None:
            reason = reason or _why_not_georeferenced(meta, alt, focal, sensor)

    yaw = (meta.gimbal_yaw_deg if meta.gimbal_yaw_deg is not None else meta.flight_yaw_deg) or 0.0
    center = (w / 2.0, h / 2.0)

    features: List[dict] = []
    for poly, cf in zip(polys_px, confs):
        px_area = polygon_pixel_area(poly)
        if px_area < min_area_px:
            continue
        props: Dict[str, object] = {
            "image_name": name,
            "timestamp": meta.timestamp,
            "confidence": None if cf is None else round(cf, 4),
        }
        if method == "orthomosaic":
            ring = referencer.ring_to_lonlat(poly)
            area_m2, area_method = polygon_area_m2(ring)
            props["length_m"] = round(_ring_length_m(ring), 3)
        elif method == "gsd":
            ring = pixel_polygon_to_lonlat(poly, center, gsd, yaw, meta.latitude, meta.longitude)
            area_m2, area_method = polygon_area_m2(ring, fallback_px_area=px_area, gsd=gsd)
            props["length_m"] = round(polygon_length_m(poly, gsd), 3)
        else:
            ring = None
            area_m2, area_method = 0.0, "none"
            props["pixel_polygon"] = poly.astype(int).tolist()
            props["length_m"] = None
        props["crack_area_m2"] = round(area_m2, 4) if ring else None
        props["area_method"] = area_method
        features.append(feature(ring, props))

    fc_meta = {
        "image_name": name,
        "timestamp": meta.timestamp,
        "georeferenced": georeferenced,
        "method": method,
        "crs": "EPSG:4326",
        "weights": Path(weights).name,
        "gsd_m_per_px": None if gsd is None else round(gsd, 6),
        "num_cracks": len(features),
        "latitude": meta.latitude,
        "longitude": meta.longitude,
    }
    if reason:
        fc_meta["reason"] = reason
    return feature_collection(features, fc_meta)


def geotag_batch(image_paths: Sequence[str], **kwargs) -> dict:
    """
    Geotag many images and merge into one FeatureCollection. A failure on
    one image is captured in `errors` and never aborts the batch.
    """
    collections: List[dict] = []
    errors: List[dict] = []
    for p in image_paths:
        try:
            collections.append(geotag_image(p, **kwargs))
        except Exception as exc:  # noqa: BLE001
            errors.append({"image": Path(p).name, "error": str(exc)})
    merged = merge_collections(collections)
    merged["properties"]["errors"] = errors
    merged["properties"]["images_processed"] = len(collections)
    return merged


def _ring_length_m(ring_lonlat: Sequence[Sequence[float]]) -> float:
    """Perimeter-based length proxy (geodesic if pyproj present, else planar-ish)."""
    try:
        from pyproj import Geod  # type: ignore
        lons = [p[0] for p in ring_lonlat]
        lats = [p[1] for p in ring_lonlat]
        _, perim = Geod(ellps="WGS84").polygon_area_perimeter(lons, lats)
        return abs(float(perim)) / 2.0  # half-perimeter ~ crack length for a thin ring
    except Exception:  # noqa: BLE001
        return 0.0


def _why_not_georeferenced(meta, alt, focal, sensor) -> str:
    missing = []
    if not meta.has_gps:
        missing.append("GPS")
    if not alt:
        missing.append("altitude")
    if not focal:
        missing.append("focal_length_mm")
    if not sensor:
        missing.append("sensor_width_mm")
    return "missing: " + ", ".join(missing) if missing else "unknown"
