"""
Guarantees the app never performs continuous live-video inference. The
live DJI POV is display-only (raw MediaMTX WebRTC); detection happens
ONCE per still photo. These are structural checks that fail loudly if
live YOLO / an RTSP frame reader is ever reintroduced.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_PY_FILES = [
    "app.py",
    "network_config.py",
    "inference_core.py",
    "infer_overlay.py",
    "detector.py",
    "inspection_db.py",
    "inspection_service.py",
    "photo_import.py",
    "import_ledger.py",
    "bridge_status.py",
    "dji_photo_bridge.py",
    "telemetry_store.py",
    "flightrecord_parser.py",
    "dji_gps_collector.py",
]


def test_live_detection_modules_absent():
    for gone in ("live_pipeline.py", "video_source.py", "capture_manager.py"):
        assert not (REPO_ROOT / gone).exists(), f"{gone} must not exist"


def test_no_runtime_module_references_a_live_pipeline():
    for name in RUNTIME_PY_FILES:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for banned in ("live_pipeline", "video_source", "capture_manager"):
            assert banned not in text, f"{name} still references {banned}"


def test_no_videocapture_in_runtime_code():
    """cv2.VideoCapture was the RTSP reader that drove continuous YOLO."""
    for name in RUNTIME_PY_FILES:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "VideoCapture" not in text, f"{name} must not use cv2.VideoCapture"


def test_stream_status_route_is_backed_by_mediamtx_api_not_a_pipeline():
    text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "get_stream_status" in text
    assert "LivePipeline" not in text
    assert "get_pipeline" not in text


def test_no_crack_only_or_confidence_leaks_in_backend():
    """
    The user-facing output is only status + original + red-highlighted.
    Ensure no crack-only extraction helper survives and the DB view never
    serializes a confidence/metric field.
    """
    detector_text = (REPO_ROOT / "detector.py").read_text(encoding="utf-8")
    assert "extract_crack_only" not in detector_text
    assert "crack_overlay_crop" not in detector_text

    db_text = (REPO_ROOT / "inspection_db.py").read_text(encoding="utf-8")
    # to_dict must not emit any of these prohibited keys.
    import inspection_db
    rec = inspection_db.InspectionRecord(
        id=1, timestamp="t", status=inspection_db.STATUS_CRACK, source="import",
        original_image_path="o", highlighted_image_path="h", num_instances=3,
    )
    d = rec.to_dict()
    for banned in ("confidence", "num_instances", "crack_image_paths",
                   "crack_image_names", "latitude", "longitude", "detection_info", "fps"):
        assert banned not in d, f"to_dict() leaks prohibited field: {banned}"
