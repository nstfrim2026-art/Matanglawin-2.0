# Tasks — Drone Crack Geotagging & Spatial Mapping Module

- [ ] 1. Metadata parser (`exif_metadata.py`)
  - [ ] 1.1 Pure `dms_to_decimal()` + `ImageGeoMetadata` dataclass with `to_dict()`.
  - [ ] 1.2 `read_metadata()` via `exifread` → Pillow fallback for GPS lat/lon/alt.
  - [ ] 1.3 Parse DJI XMP (gimbal yaw/pitch, relative/absolute altitude) from raw bytes (regex, no new dep).
  - [ ] 1.4 Extract capture timestamp → ISO-8601; graceful `has_gps=False` on missing/garbled.
  - _Requirements: 1.1–1.6_

- [ ] 2. GSD pixel→GPS core (`geotag_engine.py`, pure math)
  - [ ] 2.1 `ground_sample_distance()`.
  - [ ] 2.2 `pixel_offset_to_ground()` with gimbal-yaw rotation.
  - [ ] 2.3 `offset_to_latlon()` equirectangular local approximation.
  - [ ] 2.4 `pixel_polygon_to_lonlat()` returning a closed `[lon,lat]` ring.
  - [ ] 2.5 `CameraModel` presets + override; skip-with-reason when params unknown.
  - _Requirements: 2.1–2.6_

- [ ] 3. Geometry + measurement helpers
  - [ ] 3.1 `mask_to_polygons()` (OpenCV contours, speck drop, `approxPolyDP`).
  - [ ] 3.2 `polygon_area_m2()` (geodesic via `pyproj` if present, else `px·GSD²`).
  - [ ] 3.3 `polygon_length_m()` (min-area-rect long side × GSD proxy).
  - _Requirements: 4.2, 4.3_

- [ ] 4. Orthomosaic path (optional `rasterio`)
  - [ ] 4.1 `OrthomosaicReferencer` using affine `transform`.
  - [ ] 4.2 Reproject to EPSG:4326 when source CRS differs.
  - [ ] 4.3 Clear error if `rasterio` missing; GSD/EXIF not required on this path.
  - _Requirements: 3.1–3.4_

- [ ] 5. GeoJSON export
  - [ ] 5.1 `feature()` / `feature_collection()` with required properties + top-level metadata.
  - [ ] 5.2 `merge_collections()` and `write_geojson()` (UTF-8, `json`-parseable).
  - [ ] 5.3 Valid empty `FeatureCollection` when no cracks.
  - _Requirements: 4.1–4.7_

- [ ] 6. Orchestration
  - [ ] 6.1 `geotag_image()` reusing `inference_core.predict_masks` for polygons + confidence.
  - [ ] 6.2 Auto method selection: orthomosaic → GSD+EXIF → non-georeferenced (flagged).
  - [ ] 6.3 `geotag_batch()` with per-image error capture (never crash the batch).
  - _Requirements: 5.1–5.5_

- [ ] 7. Colab + Folium demo (`geotagging_colab_demo.ipynb`)
  - [ ] 7.1 Install extras, mount Drive, configure paths + camera params.
  - [ ] 7.2 Run `geotag_batch`, write GeoJSON, render inline Folium map with popups.
  - [ ] 7.3 Graceful handling when an image has no GPS.
  - _Requirements: 6.1–6.4_

- [ ] 8. Packaging + docs
  - [ ] 8.1 `requirements-geo.txt` (optional extras); keep base `requirements.txt` unchanged.
  - [ ] 8.2 README section: how geotagging works, GSD vs orthomosaic, limitations.
  - _Requirements: 7.1_

- [ ] 9. Tests + verification
  - [ ] 9.1 `tests/test_exif_metadata.py` and `tests/test_geotag_engine.py` (no optional deps needed).
  - [ ] 9.2 Run full suite; confirm existing tests + no-hardcoded-IP / no-live-video audits stay green.
  - _Requirements: 7.2–7.4_

- [ ] 10. Commit, push branch, open PR.
