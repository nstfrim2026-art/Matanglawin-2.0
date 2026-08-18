# Requirements — Drone Crack Geotagging & Spatial Mapping Module

## Introduction

MatanglaWIN already detects cracks in drone still photos with an instance
segmentation model (YOLO11-seg, `best.pt`), producing per-instance polygon
masks in **pixel** coordinates. This feature adds a **geospatial layer**:
it reads each photo's drone GPS/orientation metadata, converts the crack
segmentation polygons from pixel space into **real-world geographic
coordinates (EPSG:4326 / WGS84)**, and exports them as standard **GeoJSON**
so cracks can be mapped, measured, and reviewed on a web map.

The module is a **separate backend/offline analysis path** (used from
Colab and scripts). It reuses the existing shared inference
(`inference_core`) and does **not** change the live web UI, the live POV
pipeline, or the automatic photo-import workflow. Confidence scores and
metrics are allowed **inside the GeoJSON export** (a GIS artifact for
engineers), even though they remain hidden in the operator-facing web UI.

### Scope

- In scope: EXIF/XMP metadata parsing, DMS→Decimal-Degrees conversion,
  GSD-based pixel→GPS transform, optional orthomosaic GeoTIFF geo-transform
  via `rasterio`, GeoJSON FeatureCollection export, a reusable
  `geotag_engine.py`, and a Colab + Folium demo notebook.
- Out of scope: retraining the model, changing the Flask web UI, real-time
  video geotagging, and full photogrammetry/SfM (documented as future work).

### Assumptions & constraints

- Primary geolocation assumes an approximately **nadir** (straight-down)
  camera over locally **flat** terrain; deviations are surfaced as reduced
  accuracy, not hard failures.
- Core logic must run with the existing stack (Python, NumPy, OpenCV,
  Ultralytics). Geospatial extras (`exifread`/Pillow, `pyproj`, `rasterio`,
  `folium`) are **optional** and imported lazily; the module must import and
  its pure-math functions must be testable without them installed.
- Runs in Google Colab (mount Drive) and as a plain CLI/importable module.

---

## Requirement 1 — Drone image metadata parser

**User story:** As a drone inspection engineer, I want the system to read
each photo's GPS position and camera orientation from its metadata, so that
detections can be placed on a map without manual surveying.

### Acceptance criteria

1. WHEN a JPEG with GPS EXIF is provided THEN the parser SHALL return
   latitude, longitude and altitude as **decimal degrees / meters**.
2. WHEN GPS coordinates are stored as **degrees-minutes-seconds (DMS)** with
   a hemisphere reference (N/S/E/W) THEN the parser SHALL convert them to
   signed **decimal degrees** (S and W negative).
3. WHERE drone XMP/maker fields are present THEN the parser SHALL extract
   **gimbal yaw (heading), gimbal pitch**, and relative/absolute **altitude**
   when available (e.g. DJI `GimbalYawDegree`, `RelativeAltitude`).
4. WHEN a capture **timestamp** is present THEN the parser SHALL return it in
   ISO-8601 form.
5. WHEN metadata is missing or unreadable THEN the parser SHALL return a
   structured result with `has_gps=False` (and `None` fields) rather than
   raising, so callers can degrade gracefully.
6. WHERE neither `exifread` nor Pillow is installed THEN importing the module
   SHALL still succeed, and the DMS→DD conversion helper SHALL remain
   callable (pure Python).

---

## Requirement 2 — Pixel-to-GPS transformation (GSD method)

**User story:** As an engineer, I want each crack's pixel polygon converted
to real-world coordinates using flight altitude and camera specs, so cracks
land in the correct location and size on a map.

### Acceptance criteria

1. WHEN sensor width, focal length, image width and flight altitude are
   provided THEN the module SHALL compute the **Ground Sample Distance
   (GSD)** in meters/pixel as `GSD = (sensor_width_mm * altitude_m) /
   (focal_length_mm * image_width_px)`.
2. WHEN a pixel `(col, row)` and the image center are given THEN the module
   SHALL compute its ground offset **east/north in meters**, applying the
   **gimbal yaw** rotation so image "up" maps to the drone heading.
3. WHEN a ground offset in meters and the drone's lat/lon are given THEN the
   module SHALL convert to WGS84 lat/lon using a local equirectangular
   approximation (`dlat = north/111320`, `dlon = east/(111320·cos(lat))`).
4. WHEN a full pixel polygon (ring of `(col,row)`) is provided THEN the
   module SHALL return a **closed lon/lat ring** (first point repeated last)
   in GeoJSON coordinate order `[lon, lat]`.
5. WHERE altitude, focal length, or sensor width are unknown THEN the module
   SHALL skip GSD georeferencing for that image and record why (rather than
   emitting wrong coordinates).
6. The module SHALL document that the GSD method assumes nadir capture over
   flat ground and is an approximation.

---

## Requirement 3 — Orthomosaic (GeoTIFF) geo-transform

**User story:** As a GIS user with a stitched orthomosaic, I want detections
referenced through the GeoTIFF's own geo-transform, so coordinates are as
accurate as the orthomosaic itself.

### Acceptance criteria

1. WHEN a georeferenced **GeoTIFF** and `rasterio` are available THEN the
   module SHALL map pixel `(col,row)` to map coordinates using the dataset's
   affine `transform`.
2. WHERE the orthomosaic CRS is not EPSG:4326 THEN the module SHALL reproject
   the resulting coordinates to **EPSG:4326** before export.
3. WHEN `rasterio` is not installed THEN calling the orthomosaic path SHALL
   raise a clear, actionable error naming the missing dependency, while the
   rest of the module remains usable.
4. WHEN an orthomosaic transform is used THEN GSD/EXIF flight parameters
   SHALL NOT be required.

---

## Requirement 4 — GeoJSON spatial output

**User story:** As an engineer, I want detected cracks exported as standard
GeoJSON with useful attributes, so I can open them in any GIS tool or web
map.

### Acceptance criteria

1. THE module SHALL export a valid **GeoJSON `FeatureCollection`** with one
   `Feature` per detected crack, geometry type `Polygon`, coordinates in
   `[lon, lat]` WGS84 order with closed rings.
2. EACH feature's `properties` SHALL include: `crack_area_m2`,
   `length_m` (estimated), `confidence`, `image_name`, and
   `timestamp` (flight capture time when known).
3. WHERE geodesic tools (`pyproj`) are available THEN `crack_area_m2` SHALL
   be computed on the WGS84 ring; OTHERWISE it SHALL fall back to
   `pixel_area · GSD²`, and the chosen method SHALL be recorded.
4. THE `FeatureCollection` SHALL carry top-level `properties`/metadata:
   source image name, capture timestamp, georeferencing method
   (`gsd` | `orthomosaic`), CRS `EPSG:4326`, and model weights name.
5. WHEN no cracks are detected THEN the module SHALL still emit a valid,
   empty `FeatureCollection`.
6. THE module SHALL be able to **merge** many per-image FeatureCollections
   into one project-level FeatureCollection.
7. THE written file SHALL be UTF-8 and re-parseable by `json.load`.

---

## Requirement 5 — `geotag_engine.py` backend

**User story:** As a developer, I want a single importable engine that ties
segmentation + metadata + transform + export together, so Colab, scripts,
and any future service reuse one implementation.

### Acceptance criteria

1. THE repo SHALL contain a modular `geotag_engine.py` exposing a high-level
   entry point, e.g. `geotag_image(image_path, ...) -> FeatureCollection`,
   and `geotag_batch([...]) -> merged FeatureCollection`.
2. THE engine SHALL obtain crack polygons + confidences by reusing the
   existing shared inference (`inference_core.predict_masks`) — it SHALL NOT
   introduce a second/duplicate model-loading implementation.
3. THE engine SHALL select the georeferencing method automatically:
   orthomosaic transform when a GeoTIFF is supplied, else GSD+EXIF, else a
   non-georeferenced result flagged `georeferenced=False`.
4. THE engine SHALL never crash the batch on one bad image: per-image errors
   SHALL be captured and reported, and processing SHALL continue.
5. THE engine SHALL be usable headless (no Flask, no display) and SHALL not
   modify the live web UI, POV pipeline, or import-watcher behavior.

---

## Requirement 6 — Colab & interactive web-map integration

**User story:** As a researcher working in Google Colab, I want a notebook
that runs segmentation + geotagging and shows the cracks on an interactive
map, so I can validate results visually.

### Acceptance criteria

1. THE repo SHALL contain `geotagging_colab_demo.ipynb` that: installs
   optional deps, mounts Google Drive, loads `best.pt`, runs segmentation +
   geotagging on sample images, and writes GeoJSON.
2. THE notebook SHALL render an **interactive Leaflet/Folium map** inline
   with the crack polygons overlaid and popups showing their properties.
3. THE notebook SHALL degrade gracefully: if an image has no GPS, it SHALL
   report this and still show the pixel-space result.
4. THE notebook SHALL clearly separate "mount Drive → configure → run →
   map" steps and reference `geotag_engine.py` rather than duplicating logic.

---

## Requirement 7 — Dependencies, packaging, and tests

**User story:** As a maintainer, I want the geospatial extras isolated and
the core logic tested, so the existing app keeps installing cleanly and the
math is trustworthy.

### Acceptance criteria

1. Geospatial extras (`exifread`, `pyproj`, `rasterio`, `folium`) SHALL be
   documented as an **optional** install group; the base app requirements
   SHALL remain unchanged for normal operation.
2. THE core math (DMS→DD, GSD, pixel→lat/lon, polygon extraction, GeoJSON
   assembly) SHALL have unit tests that run with only NumPy/OpenCV present
   (no network, no real drone image, no optional geo libs required).
3. THE existing test suite SHALL continue to pass unchanged.
4. Known LAN-IP / no-hardcoded-IP and no-live-video-inference audits SHALL
   remain green (the new module adds neither).
