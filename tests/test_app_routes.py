"""
Tests for app.py's new live-pipeline routes, and a sanity check that the
original single-image upload routes are unaffected.

Each test gets its own isolated `matanglawin_data` directory (via
monkeypatching app module globals) so tests don't interfere with each
other or leave real state behind in the repo.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app as appmod

    # Redirect all data paths to a throwaway tmp dir instead of the real
    # matanglawin_data/ next to the repo, and reset the lazy singletons
    # so each test starts from a clean slate.
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(appmod, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(appmod, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(appmod, "CAPTURE_DIR", tmp_path / "captures")
    monkeypatch.setattr(appmod, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "inspections.db")
    for d in ["UPLOAD_DIR", "RESULT_DIR", "CAPTURE_DIR", "REPORT_DIR"]:
        getattr(appmod, d).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_pipeline", None)

    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c

    # best-effort pipeline shutdown so background threads don't leak
    # between tests
    if appmod._pipeline is not None:
        appmod._pipeline.stop()


def test_index_route_still_works(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_api_network_returns_json_and_never_errors(client):
    resp = client.get("/api/network")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "host_ip" in data
    assert "rtsp_url" in data
    assert data["rtsp_url"].startswith("rtsp://localhost:")


def test_api_mediamtx_status(client):
    resp = client.get("/api/mediamtx/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "reachable" in data


def test_api_stream_status_does_not_crash_without_mediamtx(client):
    resp = client.get("/api/stream/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["stream_state"] in ("OFFLINE", "CONNECTING", "LIVE")


def test_api_gps_reports_unavailable_by_default(client):
    resp = client.get("/api/gps")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is False


def test_api_inspections_empty_list_initially(client):
    resp = client.get("/api/inspections")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 0
    assert data["inspections"] == []


def test_api_inspections_latest_is_null_initially(client):
    resp = client.get("/api/inspections/latest")
    assert resp.status_code == 200
    assert resp.get_json() is None


def test_dashboard_page_renders(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Drone POV" in resp.data


def test_inspections_page_renders(client):
    resp = client.get("/inspections")
    assert resp.status_code == 200
    assert b"Recorded Inspections" in resp.data


def test_full_report_generates_even_with_zero_inspections(client):
    resp = client.get("/reports/full.pdf")
    assert resp.status_code == 200
    assert resp.content_type == "application/pdf"
    assert resp.data[:5] == b"%PDF-"


def test_single_report_for_nonexistent_inspection_is_404(client):
    resp = client.get("/reports/inspection/99999.pdf")
    assert resp.status_code == 404


def test_capture_file_route_serves_nested_subfolder_paths(client):
    """
    Captures now live under captures/original|crack|overlays/ (see
    capture_manager.py) - the /data/captures/<path:filename> route must
    serve files inside those subfolders, e.g. "crack/foo.jpg", not just
    flat top-level filenames.
    """
    import app as appmod
    import cv2
    import numpy as np

    crack_dir = appmod.CAPTURE_DIR / "crack"
    crack_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crack_dir / "test_crack.jpg"), np.zeros((10, 10, 3), dtype="uint8"))

    resp = client.get("/data/captures/crack/test_crack.jpg")
    assert resp.status_code == 200
    assert resp.content_type == "image/jpeg"


def test_stream_preview_route_does_not_exist(client):
    """
    /stream/preview.jpg (a full-frame annotated debug preview) has been
    removed entirely - there must be no code path that ever exposes an
    annotated full-frame image through the website.
    """
    import app as appmod

    rules = [str(rule) for rule in appmod.app.url_map.iter_rules()]
    assert not any("preview" in r for r in rules)

    resp = client.get("/stream/preview.jpg")
    assert resp.status_code == 404


def test_upload_route_rejects_missing_file(client):
    resp = client.post("/detect", data={}, content_type="multipart/form-data")
    # Missing file should redirect back to the upload form, not crash.
    assert resp.status_code in (302, 303)
