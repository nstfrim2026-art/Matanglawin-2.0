# Design — Drone Crack Geotagging & Spatial Mapping Module

## 1. Overview

A backend/offline geospatial layer on top of the existing segmentation
stack. It turns pixel-space crack polygons into WGS84 GeoJSON using either
(a) drone EXIF/XMP + Ground Sample Distance, or (b) an orthomosaic GeoTIFF's
geo-transform. It reuses `inference_core` for detection and adds two new,
dependency-light modules plus a Colab demo. Nothing in the live Flask app,
POV pipeline, or import watcher changes.

```
                    ┌─────────────────────────────────────────────┐
 drone photo  ───▶  │ inference_core.predict_masks (best.pt)       │
 (.jpg + EXIF)      │   → polygons (masks.xy) + confidences        │
                    └───────────────┬─────────────────────────────┘
                                    │ pixel polygons + conf
 exif_metadata.py ─ GPS/yaw/alt ───▶│
 (EXIF/XMP → DD)                    ▼
                    ┌─────────────────────────────────────────────┐
                    │ geotag_engine.py                            │
                    │  • choose transform: orthomosaic | GSD | none│
                    │  • pixel (col,row) → WGS84 (lon,lat)         │
                    │  • area/length in meters, attach properties │
                    │  • build GeoJSON FeatureCollection          │
                    └───────────────┬─────────────────────────────┘
                                    ▼
                    GeoJSON  ──▶  Folium/Leaflet map (Colab)  +  any GIS
```

## 2. Modules

### 2.1 `exif_metadata.py`
Pure-Python + optional readers. Public surface:

- `dms_to_decimal(degrees, minutes, seconds, ref) -> float` — pure, always
  available; `S`/`W` → negative.
- `@dataclass ImageGeoMetadata`: `has_gps`, `latitude`, `longitude`,
  `altitude_m`, `rel_altitude_m`, `gimbal_yaw_deg`, `gimbal_pitch_deg`,
  `timestamp` (ISO-8601), `image_width`, `image_height`, plus optional
  `focal_length_mm`, `sensor_width_mm` when derivable; `to_dict()`.
- `read_metadata(image_path) -> ImageGeoMetadata` — tries `exifread`, then
  Pillow `ExifTags`, for GPS; parses DJI XMP (`GimbalYawDegree`,
  `GimbalPitchDegree`, `RelativeAltitude`, `AbsoluteAltitude`) from the raw
  file bytes with a small regex (no extra dependency). Never raises on
  missing/garbled data → returns `has_gps=False`.

Reader precedence and lazy imports keep import-time cost and hard deps out.

### 2.2 `geotag_engine.py`
Depends on NumPy/OpenCV (already present) + `inference_core`; optional
`pyproj`, `rasterio`. Public surface:

- Camera/flight params: `@dataclass CameraModel(sensor_width_mm,
  focal_length_mm)` with presets (e.g. a DJI Neo 2 default) and override.
- GSD math (pure):
  - `ground_sample_distance(sensor_width_mm, focal_length_mm,
    image_width_px, altitude_m) -> m_per_px`.
  - `pixel_offset_to_ground(dx_px, dy_px, gsd, yaw_deg) -> (east_m, north_m)`
    — rotate the image-plane offset by yaw; image row increases downward, so
    north = -dy·gsd before rotation.
  - `offset_to_latlon(lat, lon, east_m, north_m) -> (lat, lon)` —
    equirectangular local approx.
  - `pixel_polygon_to_lonlat(poly_px, center_px, gsd, yaw, lat, lon) ->
    [[lon,lat],...]` (closed ring).
- Orthomosaic path (optional): `OrthomosaicReferencer(geotiff_path)` wrapping
  `rasterio` — `pixel_to_lonlat(col, row)` via affine `transform`, reproject
  to EPSG:4326 with `rasterio.warp.transform`/`pyproj` if needed. Raises a
  clear error if `rasterio` missing.
- Geometry helpers: `mask_to_polygons(binary_mask) -> [poly_px,...]` (OpenCV
  `findContours`, drop specks, simplify with `approxPolyDP`);
  `polygon_area_m2(ring_lonlat)` via `pyproj.Geod` when available else
  `pixel_area·gsd²`; `polygon_length_m(...)` via min-area-rect long side ×
  gsd (a stable proxy for crack length).
- GeoJSON builders: `feature(ring_lonlat, props)`,
  `feature_collection(features, meta)`, `merge_collections([...])`,
  `write_geojson(fc, path)`.
- Orchestration:
  - `geotag_image(image_path, weights=WEIGHTS, camera=..., altitude_m=...,
    orthomosaic=None, conf=..., imgsz=...) -> dict (FeatureCollection)`.
  - `geotag_batch(image_paths, ...) -> dict (merged FeatureCollection)` with
    per-image error capture.

### 2.3 `geotagging_colab_demo.ipynb`
Cells: (1) pip install optional extras; (2) `drive.mount`; (3) config paths +
camera params; (4) `from geotag_engine import geotag_batch`; run; (5) build a
Folium map from the GeoJSON with polygon popups; (6) save GeoJSON to Drive.

## 3. Data flow / method selection

`geotag_image` decides the referencing method in this order:
1. **orthomosaic** GeoTIFF supplied → `OrthomosaicReferencer` (most accurate).
2. else **GSD+EXIF**: read metadata; if `has_gps` and altitude + camera
   params known → GSD transform.
3. else **non-georeferenced**: still return polygons (pixel space) with
   `georeferenced=False` and a `reason`, so the notebook can show them.

## 4. GSD geometry (nadir assumption)

```
GSD (m/px)      = sensor_width_mm · altitude_m / (focal_length_mm · image_w_px)
center          = (image_w/2, image_h/2)
dx, dy (px)     = col - cx, row - cy         # dy positive downward
east0  (m)      = dx · GSD
north0 (m)      = -dy · GSD                   # up in image = north before yaw
# rotate by gimbal yaw θ (clockwise-from-north heading):
east  =  east0·cosθ + north0·sinθ
north = -east0·sinθ + north0·cosθ
lat'  = lat + north / 111320
lon'  = lon + east  / (111320 · cos(lat))
```
Documented limitation: assumes nadir view over flat ground; oblique pitch,
terrain relief, and lens distortion introduce error. For survey-grade output
use an orthomosaic (Requirement 3).

## 5. GeoJSON schema

```json
{
  "type": "FeatureCollection",
  "properties": {
    "image_name": "DJI_0001.JPG",
    "timestamp": "2026-08-17T14:45:18",
    "georeferenced": true,
    "method": "gsd",
    "crs": "EPSG:4326",
    "weights": "best.pt",
    "gsd_m_per_px": 0.0123,
    "num_cracks": 2
  },
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [[[lon,lat], ...]] },
      "properties": {
        "crack_area_m2": 0.42, "length_m": 3.1, "confidence": 0.87,
        "image_name": "DJI_0001.JPG", "timestamp": "2026-08-17T14:45:18",
        "area_method": "geodesic"
      }
    }
  ]
}
```
CRS is WGS84 (EPSG:4326), the GeoJSON default; no `crs` member is emitted in
the geometry per RFC 7946, but it is noted in `properties`.

## 6. Dependencies

| Package | Role | Required? |
|---|---|---|
| numpy, opencv-python(-headless) | polygons, math | already present |
| ultralytics | segmentation (via inference_core) | already present |
| exifread (or Pillow) | EXIF GPS | optional (parser degrades) |
| pyproj | geodesic area + reprojection | optional (falls back) |
| rasterio | orthomosaic geo-transform | optional (only for that path) |
| folium | interactive map | optional (notebook only) |

All optional imports are lazy and guarded; `import geotag_engine` and the
pure-math helpers work with none of them installed.

## 7. Testing strategy

- `tests/test_exif_metadata.py`: DMS→DD signs/values; graceful `has_gps=False`
  on a non-image; XMP field extraction from a synthetic XMP byte blob.
- `tests/test_geotag_engine.py`: GSD value; `pixel_offset_to_ground` with
  yaw=0/90; `offset_to_latlon` deltas; closed-ring order `[lon,lat]`;
  `mask_to_polygons` on a synthetic mask; GeoJSON validity + required
  properties + empty collection + merge; area fallback vs geodesic (skip
  geodesic assert if `pyproj` absent).
- No network, no real drone image, no optional geo libs required. Existing
  suite stays green.

## 8. Non-goals / future work

Full SfM/photogrammetry, DEM-based terrain correction, oblique-view
rectification, and live-video geotagging are out of scope. The orthomosaic
path is the recommended route to survey-grade accuracy today.
