"""
Guarantees that the app no longer performs continuous live-video
inference (the source of the old RTSP `DESCRIBE 404` / OpenCV timeout
errors). Detection now happens once per still photo only.

These are structural checks: they fail loudly if the live-detection
machinery is ever reintroduced.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_PY_FILES = [
    "app.py",
    "network_config.py",
    "gps_provider.py",
    "inference_core.py",
    "infer_overlay.py",
    "detector.py",
    "inspection_db.py",
    "inspection_service.py",
    "photo_import.py",
    "report_generator.py",
]


def test_live_detection_modules_are_deleted():
    for gone in ("live_pipeline.py", "video_source.py", "capture_manager.py"):
        assert not (REPO_ROOT / gone).exists(), f"{gone} should have been removed"


def test_no_runtime_module_imports_the_removed_pipeline():
    for name in RUNTIME_PY_FILES:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for banned in ("live_pipeline", "video_source", "capture_manager"):
            assert banned not in text, f"{name} still references {banned}"


def test_no_rtsp_videocapture_in_runtime_code():
    """
    Nothing may open a video stream for inference. cv2.VideoCapture was
    the RTSP reader that drove continuous YOLO; it must not appear in any
    runtime module anymore.
    """
    for name in RUNTIME_PY_FILES:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "VideoCapture" not in text, f"{name} must not use cv2.VideoCapture"


def test_app_does_not_expose_a_live_stream_state_from_video():
    """
    /api/stream/status is allowed (it's a display-only MediaMTX publisher
    probe), but it must be backed by the MediaMTX HTTP API, not by any
    frame-reading pipeline. Assert app.py routes it through
    network_config.get_stream_status and not a detector/pipeline object.
    """
    text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "get_stream_status" in text
    assert "LivePipeline" not in text
    assert "get_pipeline" not in text
