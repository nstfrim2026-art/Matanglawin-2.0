"""
test_integration.py - Comprehensive integration tests for the research-grade
UAV crack inspection system.

Run with: python tests/test_integration.py
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_all_imports():
    """Verify all modules import without errors."""
    print("=== MODULE IMPORT TESTS ===")
    import app  # noqa: F401
    import detector  # noqa: F401
    import capture_manager  # noqa: F401
    import inference_core  # noqa: F401
    import video_source  # noqa: F401
    import image_quality  # noqa: F401
    import crack_validator  # noqa: F401
    import inspection_db  # noqa: F401
    import report_generator  # noqa: F401
    import gps_provider  # noqa: F401

    # Verify letsview_source no longer exists
    import importlib
    letsview_spec = importlib.util.find_spec("letsview_source")
    assert letsview_spec is None, "letsview_source module should not exist"

    print("[PASS] All modules imported successfully (no letsview_source)")
    print()


def test_flask_endpoints():
    """Verify all Flask endpoints respond correctly."""
    print("=== FLASK ENDPOINT TESTS ===")
    from app import app

    c = app.test_client()

    # Basic endpoints
    endpoints = [
        ("/health", 200, True, False),
        ("/status", 200, True, False),
        ("/sources", 200, True, False),
        ("/capture/status", 200, True, False),
        ("/api/inspections", 200, True, False),
        ("/api/summary", 200, True, False),
        ("/api/export/csv", 200, False, False),
        ("/api/export/excel", 200, False, False),
        ("/inspections", 200, False, True),
        ("/", 200, False, True),
    ]

    for path, expected_status, expect_json, expect_html in endpoints:
        r = c.get(path)
        assert r.status_code == expected_status, (
            f"{path}: expected {expected_status}, got {r.status_code}"
        )
        if expect_json:
            data = r.get_json()
            assert data is not None, f"{path}: expected JSON response"
        if expect_html:
            assert b"<" in r.data and b">" in r.data, (
                f"{path}: expected HTML response"
            )
        print(f"  [PASS] {path}: status={r.status_code}")

    # Verify dashboard does not contain source-selector or old LIVE FEED text
    r = c.get("/")
    assert b"source-selector" not in r.data, "Dashboard still has source-selector"
    assert b"DRONE POV" in r.data, "Dashboard missing DRONE POV"
    assert b"crack-alert-banner" in r.data, "Dashboard missing crack-alert-banner"
    print("  [PASS] Dashboard: no source-selector, has DRONE POV, has crack-alert-banner")

    # Verify /api/summary fields
    r = c.get("/api/summary")
    data = r.get_json()
    required_fields = [
        "total_inspections",
        "total_confirmed_cracks",
        "average_confidence",
        "largest_crack",
        "average_crack_size",
        "inspections_today",
    ]
    for field in required_fields:
        assert field in data, f"/api/summary missing field: {field}"
    print("  [PASS] /api/summary has all required fields")

    # Verify /api/inspections structure
    r = c.get("/api/inspections")
    data = r.get_json()
    for field in ["items", "total", "page", "per_page"]:
        assert field in data, f"/api/inspections missing field: {field}"
    print("  [PASS] /api/inspections has paginated structure")

    # Verify /capture/status fields
    r = c.get("/capture/status")
    data = r.get_json()
    for field in [
        "state",
        "threshold",
        "cooldown",
        "capture_count",
        "consecutive_detections",
        "verification_required",
    ]:
        assert field in data, f"/capture/status missing field: {field}"
    print("  [PASS] /capture/status has all required fields")

    # Test DELETE /api/inspections/<id> (non-existent)
    r = c.delete("/api/inspections/NONEXISTENT-999")
    assert r.status_code == 404, f"DELETE non-existent: expected 404, got {r.status_code}"
    print("  [PASS] DELETE /api/inspections/<id> returns 404 for non-existent")

    # Test PATCH /api/inspections/<id>/notes (non-existent)
    r = c.patch(
        "/api/inspections/NONEXISTENT-999/notes",
        json={"notes": "test"},
        content_type="application/json",
    )
    assert r.status_code == 404, f"PATCH notes non-existent: expected 404, got {r.status_code}"
    print("  [PASS] PATCH /api/inspections/<id>/notes returns 404 for non-existent")

    # Test PATCH /api/inspections/<id>/notes (missing field)
    r = c.patch(
        "/api/inspections/NONEXISTENT-999/notes",
        json={},
        content_type="application/json",
    )
    assert r.status_code == 400, f"PATCH notes missing field: expected 400, got {r.status_code}"
    print("  [PASS] PATCH /api/inspections/<id>/notes returns 400 when missing notes field")

    # Test full DELETE workflow: insert then delete via API
    import inspection_db
    inspection_db.insert_inspection({
        "capture_id": "TEST-API-DEL-001",
        "timestamp": "2024-06-15T12:00:00Z",
        "confidence": 0.90,
        "classification": "crack",
    })
    r = c.delete("/api/inspections/TEST-API-DEL-001")
    assert r.status_code == 200, f"DELETE existing: expected 200, got {r.status_code}"
    data = r.get_json()
    assert data["ok"] is True
    print("  [PASS] DELETE /api/inspections/<id> successfully deletes existing record")

    # Test full PATCH workflow: insert then update notes via API
    inspection_db.insert_inspection({
        "capture_id": "TEST-API-NOTE-001",
        "timestamp": "2024-06-15T12:00:00Z",
        "confidence": 0.88,
        "classification": "crack",
    })
    r = c.patch(
        "/api/inspections/TEST-API-NOTE-001/notes",
        json={"notes": "Updated note"},
        content_type="application/json",
    )
    assert r.status_code == 200, f"PATCH notes: expected 200, got {r.status_code}"
    data = r.get_json()
    assert data["ok"] is True
    assert data["notes"] == "Updated note"
    print("  [PASS] PATCH /api/inspections/<id>/notes successfully updates notes")

    # Cleanup
    conn = sqlite3.connect(inspection_db._DB_PATH)
    conn.execute("DELETE FROM inspections WHERE capture_id='TEST-API-NOTE-001'")
    conn.commit()
    conn.close()

    print()


def test_database():
    """Verify SQLite database operations."""
    print("=== DATABASE TESTS ===")
    import inspection_db

    # init_db
    inspection_db.init_db()
    assert os.path.exists(
        inspection_db._DB_PATH
    ), "inspections.db not created"
    print("  [PASS] init_db() creates database file")

    # Verify notes column exists after migration
    conn = sqlite3.connect(inspection_db._DB_PATH)
    cursor = conn.execute("PRAGMA table_info(inspections)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    assert "notes" in columns, "notes column not present after migration"
    print("  [PASS] 'notes' column exists in schema")

    # insert_inspection
    row_id = inspection_db.insert_inspection(
        {
            "capture_id": "TEST-INTEG-001",
            "timestamp": "2024-06-15T12:00:00Z",
            "confidence": 0.92,
            "crack_area": 200,
            "estimated_length": 55.0,
            "estimated_width": 6.0,
            "bbox": [10, 20, 100, 200],
            "camera_source": "test_camera",
            "detection_threshold": 0.85,
            "image_resolution": "640x480",
            "classification": "crack",
            "image_path": "/test/image.jpg",
            "overlay_path": "/test/overlay.jpg",
            "metadata_path": "/test/meta.json",
            "report_path": "/test/report.pdf",
        }
    )
    assert row_id > 0, f"insert_inspection returned invalid id: {row_id}"
    print(f"  [PASS] insert_inspection() returned row_id={row_id}")

    # get_inspections
    result = inspection_db.get_inspections()
    assert result["total"] >= 1
    found = any(
        item["capture_id"] == "TEST-INTEG-001" for item in result["items"]
    )
    assert found, "Test record not found in query"
    print("  [PASS] get_inspections() returns test record")

    # get_summary_stats
    summary = inspection_db.get_summary_stats()
    assert summary["total_inspections"] >= 1
    assert summary["total_confirmed_cracks"] >= 1
    assert summary["average_confidence"] > 0
    print(f"  [PASS] get_summary_stats() returns valid stats")

    # update_inspection_notes
    updated = inspection_db.update_inspection_notes("TEST-INTEG-001", "Test note")
    assert updated is True, "update_inspection_notes returned False"
    record = inspection_db.get_inspection_by_id("TEST-INTEG-001")
    assert record["notes"] == "Test note", f"Notes not updated: {record.get('notes')}"
    print("  [PASS] update_inspection_notes() works correctly")

    # delete_inspection
    deleted = inspection_db.delete_inspection("TEST-INTEG-001")
    assert deleted is True, "delete_inspection returned False"
    record = inspection_db.get_inspection_by_id("TEST-INTEG-001")
    assert record is None, "Record still exists after deletion"
    print("  [PASS] delete_inspection() removes record")

    # delete non-existent
    deleted_none = inspection_db.delete_inspection("NONEXISTENT-ID")
    assert deleted_none is False, "delete_inspection should return False for non-existent"
    print("  [PASS] delete_inspection() returns False for non-existent ID")
    print()


def test_report_generator():
    """Verify PDF report generation."""
    print("=== REPORT GENERATOR TESTS ===")
    from report_generator import generate_pdf_report

    tmp_dir = tempfile.mkdtemp()
    try:
        # Create dummy images
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = os.path.join(tmp_dir, "test_image.jpg")
        overlay_path = os.path.join(tmp_dir, "test_overlay.jpg")
        output_path = os.path.join(tmp_dir, "reports", "test_report.pdf")
        cv2.imwrite(image_path, dummy)
        cv2.imwrite(overlay_path, dummy)

        inspection_data = {
            "capture_id": "TEST-PDF-001",
            "timestamp": "2024-06-15T12:00:00Z",
            "classification": "crack",
            "confidence": 0.95,
            "crack_area": 150,
            "estimated_length": 45.2,
            "estimated_width": 5.3,
            "bbox": [10, 20, 100, 200],
            "camera_source": "test_camera",
            "detection_threshold": 0.85,
            "image_resolution": "640x480",
        }

        result = generate_pdf_report(
            inspection_data, image_path, overlay_path, output_path
        )
        assert os.path.exists(output_path), "PDF file not created"
        size = os.path.getsize(output_path)
        assert size > 100, f"PDF too small: {size} bytes"
        print(f"  [PASS] generate_pdf_report() creates PDF ({size} bytes)")
    finally:
        shutil.rmtree(tmp_dir)
    print()


def test_crack_validator():
    """Verify crack validation filters."""
    print("=== CRACK VALIDATOR TESTS ===")
    from crack_validator import validate_cracks, DJI_UI_ZONE_FRACTION

    frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)

    # Small blobs rejected (threshold is now 150)
    small = [
        {
            "confidence": 0.95,
            "area_px": 50,
            "estimated_length_px": 10.0,
            "estimated_width_px": 5.0,
            "bbox": [10, 200, 30, 240],
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(small, frame)) == 0
    print("  [PASS] Small blobs (area < 150px) rejected")

    # Elongated cracks pass
    elongated = [
        {
            "confidence": 0.95,
            "area_px": 500,
            "estimated_length_px": 100.0,
            "estimated_width_px": 10.0,
            "bbox": [10, 100, 200, 140],
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(elongated, frame)) == 1
    print("  [PASS] Elongated cracks pass validation")

    # Round noise rejected (low aspect ratio < 2.5)
    noise = [
        {
            "confidence": 0.90,
            "area_px": 200,
            "estimated_length_px": 30.0,
            "estimated_width_px": 25.0,
            "bbox": [10, 200, 40, 245],
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(noise, frame)) == 0
    print("  [PASS] Round noise (low aspect ratio) rejected")

    # Short cracks rejected (threshold is now 30)
    short = [
        {
            "confidence": 0.90,
            "area_px": 200,
            "estimated_length_px": 25.0,
            "estimated_width_px": 3.0,
            "bbox": [10, 200, 35, 223],
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(short, frame)) == 0
    print("  [PASS] Short cracks (length < 30px) rejected")

    # DJI UI zone rejection: detection in top 15% of frame
    h = 480
    top_zone_max = int(h * DJI_UI_ZONE_FRACTION)  # 72 pixels
    top_detection = [
        {
            "confidence": 0.95,
            "area_px": 500,
            "estimated_length_px": 100.0,
            "estimated_width_px": 10.0,
            "bbox": [10, 5, 200, 60],  # entirely within top 15%
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(top_detection, frame)) == 0
    print("  [PASS] DJI UI zone: detection in top 15% rejected")

    # DJI UI zone rejection: detection in bottom 15% of frame
    bottom_zone_min = int(h * (1.0 - DJI_UI_ZONE_FRACTION))  # 408 pixels
    bottom_detection = [
        {
            "confidence": 0.95,
            "area_px": 500,
            "estimated_length_px": 100.0,
            "estimated_width_px": 10.0,
            "bbox": [10, 420, 200, 475],  # entirely within bottom 15%
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(bottom_detection, frame)) == 0
    print("  [PASS] DJI UI zone: detection in bottom 15% rejected")

    # Detection in center of frame should pass
    center_detection = [
        {
            "confidence": 0.95,
            "area_px": 500,
            "estimated_length_px": 100.0,
            "estimated_width_px": 10.0,
            "bbox": [10, 150, 200, 300],  # center of frame
            "classification": "crack",
        }
    ]
    assert len(validate_cracks(center_detection, frame)) == 1
    print("  [PASS] DJI UI zone: detection in center passes")
    print()


def test_image_quality():
    """Verify image quality validation."""
    print("=== IMAGE QUALITY TESTS ===")
    from image_quality import validate_image_quality

    # Black frame fails brightness
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = validate_image_quality(black)
    assert not result["passed"]
    assert not result["checks"]["brightness"]["passed"]
    print("  [PASS] Black frame fails brightness check")

    # Blurry frame fails blur
    blurry = np.ones((480, 640, 3), dtype=np.uint8) * 128
    blurry[:, :320] = 130
    blurry[:, 320:] = 126
    blurry = cv2.GaussianBlur(blurry, (51, 51), 25)
    result = validate_image_quality(blurry)
    assert not result["checks"]["blur"]["passed"]
    print("  [PASS] Blurry frame fails blur check")

    # Normal frame passes all
    normal = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    result = validate_image_quality(normal)
    assert result["passed"], f"Normal frame failed: {result['detail']}"
    print("  [PASS] Normal frame passes all checks")

    # Overexposed frame fails
    bright = np.ones((480, 640, 3), dtype=np.uint8) * 250
    result = validate_image_quality(bright)
    assert not result["passed"]
    print("  [PASS] Overexposed frame fails brightness check")
    print()


def test_preprocess_frame():
    """Verify preprocessing enhances dark frames."""
    print("=== PREPROCESS FRAME TESTS ===")
    from inference_core import preprocess_frame

    # Dark frame should be brightened by CLAHE + gamma
    dark_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    result = preprocess_frame(dark_frame)
    assert result.shape == dark_frame.shape
    assert result.dtype == np.uint8
    # CLAHE + gamma on a dark frame should produce a brighter result
    assert result.mean() > dark_frame.mean(), (
        f"Expected brighter output: input mean={dark_frame.mean():.1f}, "
        f"output mean={result.mean():.1f}"
    )
    print("  [PASS] Dark frame is brightened by preprocessing")

    # Normal frame should maintain similar dimensions and type
    normal_frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    result_normal = preprocess_frame(normal_frame)
    assert result_normal.shape == normal_frame.shape
    assert result_normal.dtype == np.uint8
    print("  [PASS] Normal frame maintains shape and dtype")

    # Empty/None handling
    empty = np.array([], dtype=np.uint8)
    result_empty = preprocess_frame(empty)
    assert result_empty.size == 0
    print("  [PASS] Empty frame handled gracefully")
    print()


def test_confidence_default():
    """Verify default confidence threshold is 0.40."""
    print("=== CONFIDENCE DEFAULT TESTS ===")
    import inspect
    from inference_core import annotate_frame

    sig = inspect.signature(annotate_frame)
    conf_default = sig.parameters["conf"].default
    assert conf_default == 0.40, f"Expected 0.40, got {conf_default}"
    print(f"  [PASS] inference_core.annotate_frame conf default = {conf_default}")

    import app
    assert app.CONF == 0.40, f"Expected app.CONF=0.40, got {app.CONF}"
    print(f"  [PASS] app.CONF = {app.CONF}")
    print()


def test_temporal_verification():
    """Verify temporal verification requires 5 consecutive frames spanning 1.5s."""
    print("=== TEMPORAL VERIFICATION TESTS ===")
    from capture_manager import CaptureManager, VERIFICATION_FRAME_COUNT, VERIFICATION_MIN_SPAN_SECONDS

    assert VERIFICATION_FRAME_COUNT == 5
    print(f"  [PASS] VERIFICATION_FRAME_COUNT = {VERIFICATION_FRAME_COUNT}")
    assert VERIFICATION_MIN_SPAN_SECONDS == 1.5
    print(f"  [PASS] VERIFICATION_MIN_SPAN_SECONDS = {VERIFICATION_MIN_SPAN_SECONDS}")

    cm = CaptureManager(threshold=0.85, cooldown=1.0)
    frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    meta = [
        {
            "confidence": 0.95,
            "area_px": 500,
            "estimated_length_px": 100.0,
            "estimated_width_px": 10.0,
            "bbox": [10, 100, 200, 140],
            "classification": "crack",
        }
    ]

    # 4 frames should not trigger
    for _ in range(4):
        cm.process_frame(frame, meta)
    status = cm.get_capture_status()
    assert status["state"] == "monitoring"
    assert status["consecutive_detections"] == 4
    print("  [PASS] 4 frames: still monitoring, 4 consecutive")

    # 5th frame should not trigger immediately (time span < 1.5s)
    cm.process_frame(frame, meta)
    time.sleep(0.1)
    status = cm.get_capture_status()
    # Should still be monitoring since frames arrived too fast
    assert status["state"] == "monitoring"
    print("  [PASS] 5 fast frames: still monitoring (time-spread not met)")

    # Reset and verify counter resets on no detection
    cm.reset()
    cm.process_frame(frame, meta)
    cm.process_frame(frame, meta)
    cm.process_frame(frame, [])  # No detection
    status = cm.get_capture_status()
    assert status["consecutive_detections"] == 0
    print("  [PASS] No-detection frame resets counter")
    print()


def test_gps_provider():
    """Verify GPS provider functionality."""
    print("=== GPS PROVIDER TESTS ===")
    from gps_provider import GPSProvider, DEFAULT_GPS_TOLERANCE_SECONDS

    # Instantiation
    gps = GPSProvider()
    assert gps is not None
    print("  [PASS] GPSProvider instantiates successfully")

    # get_current_position returns all None (no live GPS from DJI Neo 2)
    pos = gps.get_current_position()
    assert pos["latitude"] is None
    assert pos["longitude"] is None
    assert pos["altitude"] is None
    assert pos["timestamp"] is None
    print("  [PASS] get_current_position() returns None values (no live GPS)")

    # Default tolerance
    assert gps.tolerance_seconds == DEFAULT_GPS_TOLERANCE_SECONDS
    print(f"  [PASS] Default tolerance = {DEFAULT_GPS_TOLERANCE_SECONDS}s")

    # import_gps_log with valid data
    log_data = [
        {"timestamp": "2026-08-12T14:32:15.000", "latitude": 7.123450, "longitude": 125.123450, "altitude": 120.0},
        {"timestamp": "2026-08-12T14:32:18.000", "latitude": 7.123452, "longitude": 125.123455, "altitude": 123.4},
        {"timestamp": "2026-08-12T14:32:21.000", "latitude": 7.123454, "longitude": 125.123460, "altitude": 125.0},
    ]
    imported = gps.import_gps_log(log_data)
    assert imported == 3, f"Expected 3 imported, got {imported}"
    assert gps.log_size == 3
    print(f"  [PASS] import_gps_log() imported {imported} points")

    # get_position_at_timestamp - exact match
    pos = gps.get_position_at_timestamp("2026-08-12T14:32:18.000")
    assert pos["latitude"] == 7.123452
    assert pos["longitude"] == 125.123455
    assert pos["altitude"] == 123.4
    print("  [PASS] get_position_at_timestamp() exact match works")

    # get_position_at_timestamp - close match (within tolerance)
    pos = gps.get_position_at_timestamp("2026-08-12T14:32:19.000")
    # Should match the 14:32:18 point (1 second away)
    assert pos["latitude"] == 7.123452
    assert pos["longitude"] == 125.123455
    print("  [PASS] get_position_at_timestamp() close match within tolerance")

    # get_position_at_timestamp - out of tolerance
    pos = gps.get_position_at_timestamp("2026-08-12T14:33:00.000")
    assert pos["latitude"] is None
    assert pos["longitude"] is None
    print("  [PASS] get_position_at_timestamp() returns None when out of tolerance")

    # import_gps_log with invalid entries (missing fields)
    bad_log = [
        {"timestamp": "2026-08-12T14:32:18.000"},  # missing lat/lon
        {"latitude": 7.0, "longitude": 125.0},  # missing timestamp
        {"timestamp": "2026-08-12T14:32:18.000", "latitude": 7.0, "longitude": 125.0},  # valid
    ]
    gps2 = GPSProvider()
    imported2 = gps2.import_gps_log(bad_log)
    assert imported2 == 1, f"Expected 1 valid import from bad log, got {imported2}"
    print("  [PASS] import_gps_log() skips invalid entries")

    # Configurable tolerance
    gps3 = GPSProvider(tolerance_seconds=1.0)
    gps3.import_gps_log(log_data)
    pos = gps3.get_position_at_timestamp("2026-08-12T14:32:19.500")
    # 1.5 seconds away from nearest - exceeds 1.0 tolerance
    assert pos["latitude"] is None
    print("  [PASS] Custom tolerance (1.0s) correctly rejects match beyond threshold")
    print()


def test_database_gps_columns():
    """Verify GPS columns exist in the database schema."""
    print("=== DATABASE GPS COLUMNS TESTS ===")
    import inspection_db
    import sqlite3

    inspection_db.init_db()
    conn = sqlite3.connect(inspection_db._DB_PATH)
    cursor = conn.execute("PRAGMA table_info(inspections)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "latitude" in columns, "latitude column missing"
    assert "longitude" in columns, "longitude column missing"
    assert "altitude" in columns, "altitude column missing"
    assert "gps_timestamp" in columns, "gps_timestamp column missing"
    assert "crop_path" in columns, "crop_path column missing"
    print("  [PASS] GPS columns (latitude, longitude, altitude, gps_timestamp) exist")
    print("  [PASS] crop_path column exists")

    # Test insert with GPS data
    row_id = inspection_db.insert_inspection({
        "capture_id": "TEST-GPS-001",
        "timestamp": "2026-08-12T14:32:18Z",
        "confidence": 0.95,
        "classification": "crack",
        "latitude": 7.123452,
        "longitude": 125.123455,
        "altitude": 123.4,
        "gps_timestamp": "2026-08-12T14:32:18.000",
        "crop_path": "/test/crops/TEST-GPS-001_crop.jpg",
    })
    assert row_id > 0
    print(f"  [PASS] insert_inspection() with GPS data succeeded (id={row_id})")

    # Verify GPS data is retrievable
    record = inspection_db.get_inspection_by_id("TEST-GPS-001")
    assert record is not None
    assert record["latitude"] == 7.123452
    assert record["longitude"] == 125.123455
    assert record["altitude"] == 123.4
    assert record["gps_timestamp"] == "2026-08-12T14:32:18.000"
    assert record["crop_path"] == "/test/crops/TEST-GPS-001_crop.jpg"
    print("  [PASS] GPS data retrieved correctly from database")

    # Test insert without GPS data (nullable)
    row_id2 = inspection_db.insert_inspection({
        "capture_id": "TEST-GPS-002",
        "timestamp": "2026-08-12T14:35:00Z",
        "confidence": 0.88,
        "classification": "crack",
    })
    assert row_id2 > 0
    record2 = inspection_db.get_inspection_by_id("TEST-GPS-002")
    assert record2["latitude"] is None
    assert record2["longitude"] is None
    assert record2["altitude"] is None
    print("  [PASS] insert_inspection() without GPS data works (nullable)")

    # Cleanup
    inspection_db.delete_inspection("TEST-GPS-001")
    inspection_db.delete_inspection("TEST-GPS-002")
    print()


def test_report_generator_with_gps():
    """Verify PDF report with GPS data and crop image."""
    print("=== REPORT GENERATOR GPS TESTS ===")
    from report_generator import generate_pdf_report, CSV_COLUMNS

    tmp_dir = tempfile.mkdtemp()
    try:
        # Create dummy images
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = os.path.join(tmp_dir, "test_image.jpg")
        overlay_path = os.path.join(tmp_dir, "test_overlay.jpg")
        crop_path = os.path.join(tmp_dir, "test_crop.jpg")
        cv2.imwrite(image_path, dummy)
        cv2.imwrite(overlay_path, dummy)
        # Create a smaller crop
        crop_img = dummy[100:300, 50:250]
        cv2.imwrite(crop_path, crop_img)

        # Test with GPS data
        inspection_data_gps = {
            "capture_id": "TEST-PDF-GPS-001",
            "timestamp": "2026-08-12T14:32:18Z",
            "classification": "crack",
            "confidence": 0.95,
            "crack_area": 150,
            "estimated_length": 45.2,
            "estimated_width": 5.3,
            "bbox": [10, 20, 100, 200],
            "camera_source": "drone",
            "detection_threshold": 0.85,
            "image_resolution": "640x480",
            "latitude": 7.123452,
            "longitude": 125.123455,
            "altitude": 123.4,
        }

        output_path_gps = os.path.join(tmp_dir, "reports", "test_gps_report.pdf")
        result = generate_pdf_report(
            inspection_data_gps, image_path, overlay_path, output_path_gps,
            crop_path=crop_path
        )
        assert os.path.exists(output_path_gps), "PDF with GPS not created"
        size = os.path.getsize(output_path_gps)
        assert size > 100, f"PDF with GPS too small: {size} bytes"
        print(f"  [PASS] PDF report with GPS data generated ({size} bytes)")

        # Test without GPS data
        inspection_data_no_gps = {
            "capture_id": "TEST-PDF-NOGPS-001",
            "timestamp": "2026-08-12T14:35:00Z",
            "classification": "crack",
            "confidence": 0.88,
            "crack_area": 120,
            "estimated_length": 38.0,
            "estimated_width": 4.0,
            "bbox": [20, 30, 110, 180],
            "camera_source": "drone",
            "detection_threshold": 0.85,
            "image_resolution": "640x480",
            "latitude": None,
            "longitude": None,
            "altitude": None,
        }

        output_path_no_gps = os.path.join(tmp_dir, "reports", "test_no_gps_report.pdf")
        result = generate_pdf_report(
            inspection_data_no_gps, image_path, overlay_path, output_path_no_gps
        )
        assert os.path.exists(output_path_no_gps), "PDF without GPS not created"
        size2 = os.path.getsize(output_path_no_gps)
        assert size2 > 100, f"PDF without GPS too small: {size2} bytes"
        print(f"  [PASS] PDF report without GPS data generated ({size2} bytes)")

        # Test CSV columns include GPS
        assert "latitude" in CSV_COLUMNS, "CSV_COLUMNS missing latitude"
        assert "longitude" in CSV_COLUMNS, "CSV_COLUMNS missing longitude"
        assert "altitude" in CSV_COLUMNS, "CSV_COLUMNS missing altitude"
        print("  [PASS] CSV_COLUMNS includes latitude, longitude, altitude")

    finally:
        shutil.rmtree(tmp_dir)
    print()


def test_crop_padding_ratio():
    """Verify crop padding ratio constant exists in capture_manager."""
    print("=== CROP PADDING RATIO TESTS ===")
    from capture_manager import CROP_PADDING_RATIO

    assert CROP_PADDING_RATIO == 0.15, f"Expected 0.15, got {CROP_PADDING_RATIO}"
    print(f"  [PASS] CROP_PADDING_RATIO = {CROP_PADDING_RATIO}")
    print()


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("MATANGLAWIN RESEARCH-GRADE INSPECTION - INTEGRATION TESTS")
    print("=" * 60)
    print()

    test_all_imports()
    test_flask_endpoints()
    test_database()
    test_report_generator()
    test_crack_validator()
    test_image_quality()
    test_preprocess_frame()
    test_confidence_default()
    test_temporal_verification()
    test_gps_provider()
    test_database_gps_columns()
    test_report_generator_with_gps()
    test_crop_padding_ratio()

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
