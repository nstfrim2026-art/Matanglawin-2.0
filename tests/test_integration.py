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

    print("[PASS] All modules imported successfully")
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

    # Cleanup
    conn = sqlite3.connect(inspection_db._DB_PATH)
    conn.execute(
        "DELETE FROM inspections WHERE capture_id='TEST-INTEG-001'"
    )
    conn.commit()
    conn.close()
    print("  [PASS] Test record cleaned up")
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

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
