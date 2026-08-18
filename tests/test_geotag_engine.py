"""
Tests for geotag_engine.py - GSD math, pixel->WGS84, polygon geometry, and
GeoJSON assembly. Runs with only numpy/opencv (pyproj/rasterio not required);
segmentation is stubbed so no model/torch is loaded.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exif_metadata as em  # noqa: E402
import geotag_engine as ge  # noqa: E402


# ------------------------------------------------------------- GSD math ----

def test_ground_sample_distance_value():
    gsd = ge.ground_sample_distance(sensor_width_mm=6.4, focal_length_mm=4.7,
                                    image_width_px=4000, altitude_m=50.0)
    assert abs(gsd - (6.4 * 50.0) / (4.7 * 4000)) < 1e-9


def test_gsd_rejects_bad_params():
    for bad in ({"focal_length_mm": 0}, {"image_width_px": 0}):
        kw = {"sensor_width_mm": 6.4, "focal_length_mm": 4.7, "image_width_px": 4000, "altitude_m": 50}
        kw.update(bad)
        try:
            ge.ground_sample_distance(**kw)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_pixel_offset_to_ground_no_yaw():
    e, n = ge.pixel_offset_to_ground(dx_px=100, dy_px=0, gsd=0.01, yaw_deg=0)
    assert abs(e - 1.0) < 1e-9 and abs(n) < 1e-9
    e, n = ge.pixel_offset_to_ground(dx_px=0, dy_px=100, gsd=0.01, yaw_deg=0)
    assert abs(e) < 1e-9 and abs(n + 1.0) < 1e-9  # row down = south


def test_pixel_offset_to_ground_yaw_90():
    # A point "up" in the image (north before rotation) with yaw 90 deg
    # should map toward +east (heading rotated clockwise).
    e, n = ge.pixel_offset_to_ground(dx_px=0, dy_px=-100, gsd=0.01, yaw_deg=90)
    assert abs(e - 1.0) < 1e-6 and abs(n) < 1e-6


def test_offset_to_latlon_deltas():
    lat, lon = ge.offset_to_latlon(0.0, 0.0, east_m=ge._M_PER_DEG, north_m=ge._M_PER_DEG)
    assert abs(lat - 1.0) < 1e-6           # ~111320 m north = 1 deg lat
    assert abs(lon - 1.0) < 1e-6           # at equator cos(0)=1


def test_pixel_polygon_to_lonlat_closed_ring_and_order():
    poly = [(90, 100), (110, 100), (110, 120), (90, 120)]  # square in px
    ring = ge.pixel_polygon_to_lonlat(poly, center_px=(100, 100), gsd=0.01,
                                      yaw_deg=0, lat=14.6, lon=120.98)
    assert ring[0] == ring[-1]             # closed
    # GeoJSON order [lon, lat]: lon near 120.98, lat near 14.6
    assert abs(ring[0][0] - 120.98) < 0.01 and abs(ring[0][1] - 14.6) < 0.01


# --------------------------------------------------- polygon geometry ------

def test_mask_to_polygons_and_area():
    mask = np.zeros((200, 200), np.uint8)
    mask[40:160, 90:110] = 1               # a tall thin rectangle
    polys = ge.mask_to_polygons(mask, min_area_px=50)
    assert len(polys) == 1
    area = ge.polygon_pixel_area(polys[0])
    assert area > 1500                      # ~ 120*20 = 2400
    gsd = 0.01
    length = ge.polygon_length_m(polys[0], gsd)
    assert abs(length - 120 * gsd) < 0.2    # long side ~120 px


def test_polygon_area_m2_method_label():
    ring = [[120.0, 14.0], [120.0001, 14.0], [120.0001, 14.0001], [120.0, 14.0001], [120.0, 14.0]]
    area, method = ge.polygon_area_m2(ring, fallback_px_area=2400, gsd=0.01)
    assert area > 0
    assert method in ("geodesic", "gsd_pixel")


# ------------------------------------------------------------- GeoJSON -----

def test_feature_collection_and_empty():
    fc = ge.feature_collection([], {"image_name": "a.jpg", "num_cracks": 0})
    assert fc["type"] == "FeatureCollection"
    assert fc["features"] == []
    assert fc["properties"]["num_cracks"] == 0


def test_feature_null_geometry_when_not_georeferenced():
    f = ge.feature(None, {"image_name": "a.jpg"})
    assert f["type"] == "Feature" and f["geometry"] is None


def test_write_and_reparse_geojson(tmp_path):
    ring = [[120.0, 14.0], [120.001, 14.0], [120.001, 14.001], [120.0, 14.0]]
    fc = ge.feature_collection([ge.feature(ring, {"confidence": 0.9})], {"crs": "EPSG:4326"})
    out = tmp_path / "cracks.geojson"
    ge.write_geojson(fc, str(out))
    reparsed = json.load(open(out, encoding="utf-8"))
    assert reparsed["type"] == "FeatureCollection"
    assert reparsed["features"][0]["geometry"]["type"] == "Polygon"


def test_merge_collections():
    a = ge.feature_collection([ge.feature([[0, 0], [0, 1], [1, 1], [0, 0]], {})], {"image_name": "a"})
    b = ge.feature_collection([ge.feature([[2, 2], [2, 3], [3, 3], [2, 2]], {})], {"image_name": "b"})
    merged = ge.merge_collections([a, b])
    assert merged["properties"]["num_cracks"] == 2
    assert set(merged["properties"]["images"]) == {"a", "b"}


# ------------------------------------------------ orchestration (stubbed) --

class _FakeMasks:
    def __init__(self, xy):
        self.xy = xy

    def __len__(self):
        return len(self.xy)


class _FakeConf:
    def __init__(self, v):
        self._v = np.asarray(v, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._v


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _FakeConf(confs)


class _FakeResult:
    def __init__(self, img, xy, confs):
        self.orig_img = img
        self.masks = _FakeMasks(xy) if xy else None
        self.boxes = _FakeBoxes(confs)


def test_geotag_image_gsd_path(tmp_path, monkeypatch):
    # Real (tiny) image on disk so Path()/dims work; content is irrelevant.
    img = np.full((400, 600, 3), 200, np.uint8)
    img_path = tmp_path / "DJI_0001.JPG"
    cv2.imwrite(str(img_path), img)

    square = np.array([[290, 190], [310, 190], [310, 210], [290, 210]], dtype=float)
    monkeypatch.setattr(
        ge, "_detect_polygons",
        lambda *a, **k: ([square], [0.87], (400, 600)),
    )
    # Force georeferencing inputs (has_gps + altitude + dims).
    monkeypatch.setattr(em, "read_metadata", lambda p: em.ImageGeoMetadata(
        image_name="DJI_0001.JPG", has_gps=True, latitude=14.6, longitude=120.98,
        rel_altitude_m=50.0, gimbal_yaw_deg=0.0, timestamp="2026-08-17T14:45:18",
        image_width=600, image_height=400,
    ))

    fc = ge.geotag_image(str(img_path), camera=ge.CameraModel(6.4, 4.7))
    assert fc["properties"]["georeferenced"] is True
    assert fc["properties"]["method"] == "gsd"
    assert fc["properties"]["num_cracks"] == 1
    feat = fc["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]                      # closed
    # coordinates near the drone position
    assert abs(ring[0][0] - 120.98) < 0.01 and abs(ring[0][1] - 14.6) < 0.01
    props = feat["properties"]
    assert props["confidence"] == 0.87
    assert props["crack_area_m2"] is not None and props["length_m"] is not None
    assert props["image_name"] == "DJI_0001.JPG"
    assert props["timestamp"] == "2026-08-17T14:45:18"


def test_geotag_image_non_georeferenced_when_no_gps(tmp_path, monkeypatch):
    img = np.full((400, 600, 3), 200, np.uint8)
    img_path = tmp_path / "nogps.jpg"
    cv2.imwrite(str(img_path), img)
    square = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=float)
    monkeypatch.setattr(ge, "_detect_polygons", lambda *a, **k: ([square], [0.5], (400, 600)))
    monkeypatch.setattr(em, "read_metadata", lambda p: em.ImageGeoMetadata(
        image_name="nogps.jpg", has_gps=False, image_width=600, image_height=400))

    fc = ge.geotag_image(str(img_path))
    assert fc["properties"]["georeferenced"] is False
    assert "missing" in fc["properties"].get("reason", "")
    feat = fc["features"][0]
    assert feat["geometry"] is None                 # no map geometry
    assert "pixel_polygon" in feat["properties"]    # pixel-space kept


def test_geotag_batch_captures_errors(tmp_path, monkeypatch):
    good = np.full((100, 100, 3), 200, np.uint8)
    gp = tmp_path / "ok.jpg"
    cv2.imwrite(str(gp), good)

    def fake_detect(image_path, *a, **k):
        if "boom" in str(image_path):
            raise RuntimeError("kaboom")
        return ([np.array([[10, 10], [40, 10], [40, 40], [10, 40]], float)], [0.6], (100, 100))

    monkeypatch.setattr(ge, "_detect_polygons", fake_detect)
    monkeypatch.setattr(em, "read_metadata", lambda p: em.ImageGeoMetadata(has_gps=False,
                                                                           image_width=100, image_height=100))
    merged = ge.geotag_batch([str(gp), str(tmp_path / "boom.jpg")])
    assert merged["properties"]["images_processed"] == 1
    assert merged["properties"]["errors"] and merged["properties"]["errors"][0]["error"] == "kaboom"
