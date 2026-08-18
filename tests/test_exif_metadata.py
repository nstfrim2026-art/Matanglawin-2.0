"""
Tests for exif_metadata.py - DMS->DD, XMP field parsing, graceful failure.
Runs with only the standard library + numpy/opencv (no exifread/Pillow needed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exif_metadata as em  # noqa: E402


def test_dms_to_decimal_north_east_positive():
    assert abs(em.dms_to_decimal(14, 35, 58.2, "N") - 14.59950) < 1e-4
    assert abs(em.dms_to_decimal(120, 59, 3.12, "E") - 120.98420) < 1e-4


def test_dms_to_decimal_south_west_negative():
    assert em.dms_to_decimal(33, 51, 54.0, "S") < 0
    assert em.dms_to_decimal(151, 12, 36.0, "W") < 0


def test_dms_zero_and_ref_case_insensitive():
    assert em.dms_to_decimal(0, 0, 0, "N") == 0.0
    assert em.dms_to_decimal(10, 0, 0, "s") == -10.0


def test_metadata_missing_is_graceful(tmp_path):
    # A non-image file must not raise and must report has_gps=False.
    p = tmp_path / "not_an_image.txt"
    p.write_text("hello")
    meta = em.read_metadata(str(p))
    assert meta.has_gps is False
    assert meta.latitude is None and meta.longitude is None
    assert meta.image_name == "not_an_image.txt"


def test_parse_xmp_dji_fields():
    xmp = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
        b'drone-dji:GimbalYawDegree="+45.30" '
        b'drone-dji:GimbalPitchDegree="-90.00" '
        b'drone-dji:RelativeAltitude="+52.10" '
        b'drone-dji:AbsoluteAltitude="+123.40" '
        b'/></rdf:RDF></x:xmpmeta>'
    )
    meta = em.ImageGeoMetadata(image_name="x.jpg")
    em.parse_xmp(xmp, meta)
    assert abs(meta.gimbal_yaw_deg - 45.30) < 1e-6
    assert abs(meta.gimbal_pitch_deg + 90.0) < 1e-6
    assert abs(meta.rel_altitude_m - 52.10) < 1e-6
    assert abs(meta.abs_altitude_m - 123.40) < 1e-6


def test_parse_xmp_element_form():
    xmp = b"<drone-dji:GimbalYawDegree>12.5</drone-dji:GimbalYawDegree>"
    meta = em.ImageGeoMetadata()
    em.parse_xmp(xmp, meta)
    assert abs(meta.gimbal_yaw_deg - 12.5) < 1e-6


def test_best_altitude_prefers_relative():
    meta = em.ImageGeoMetadata(altitude_m=100.0, rel_altitude_m=40.0, abs_altitude_m=140.0)
    assert meta.best_altitude() == 40.0
    meta2 = em.ImageGeoMetadata(altitude_m=100.0)
    assert meta2.best_altitude() == 100.0
    assert em.ImageGeoMetadata().best_altitude() is None


def test_to_dict_roundtrips_fields():
    meta = em.ImageGeoMetadata(image_name="a.jpg", has_gps=True, latitude=1.0, longitude=2.0)
    d = meta.to_dict()
    assert d["image_name"] == "a.jpg" and d["has_gps"] is True
    assert d["latitude"] == 1.0 and d["longitude"] == 2.0
