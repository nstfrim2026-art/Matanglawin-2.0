"""
Tests for app.py routes.

Each test gets an isolated data dir (via monkeypatching app globals) and a
stubbed YOLO (via inference_core.predict_masks) so no real model or network
is needed, and detection is deterministic.
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from tests import stubs  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app as appmod

    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(appmod, "UPLOAD_TMP_DIR", tmp_path / "upload_tmp")
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "inspections")
    monkeypatch.setattr(appmod, "IMPORT_DIR", tmp_path / "import")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "inspections.db")
    for attr in ["UPLOAD_TMP_DIR", "INSPECTIONS_DIR", "IMPORT_DIR"]:
        getattr(appmod, attr).mkdir(parents=True, exist_ok=True)

    # reset lazy singletons for a clean slate
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    monkeypatch.setattr(appmod, "_watcher", None)

    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


# -- pages ------------------------------------------------------------

def test_dashboard_is_the_main_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Live Drone POV" in resp.data
    # The production capture control lives on the dashboard.
    assert b"Capture Photo" in resp.data
    assert b'id="capture-btn"' in resp.data


def test_dashboard_pov_is_clean_display_only(client):
    resp = client.get("/")
    body = resp.data.decode()
    # The old "press the shutter on the controller ... transferred ... no upload,
    # no Analyze button" explanatory paragraph was removed entirely (the
    # workflow is now the Capture Photo button).
    assert "shutter" not in body.lower()
    assert "no analyze button" not in body.lower()
    assert "monitoring only" not in body.lower()
    # No bounding-box / confidence / FPS / overlay language anywhere on the page
    # (the live view is clean; AI results appear only in the Inspection section).
    for banned in ("bounding box", "confidence", "FPS", "IoU", "crack-only",
                   "segmentation overlay"):
        assert banned.lower() not in body.lower()


def test_dashboard_alias(client):
    assert client.get("/dashboard").status_code == 200


def test_upload_fallback_page_renders_and_is_labelled_fallback(client):
    resp = client.get("/upload")
    assert resp.status_code == 200
    assert b"Manual Upload" in resp.data
    assert b"Fallback" in resp.data


def test_inspections_page_renders(client):
    resp = client.get("/inspections")
    assert resp.status_code == 200
    assert b"Recorded Inspections" in resp.data


# -- network / stream APIs -------------------------------------------

def test_api_network_shape(client):
    data = client.get("/api/network").get_json()
    assert "host_ip" in data
    assert data["rtsp_url"].startswith("rtsp://localhost:")


def test_api_network_reflects_dynamic_ip_override(client, monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "172.20.10.3")  # example test value
    data = client.get("/api/network").get_json()
    assert data["host_ip"] == "172.20.10.3"
    assert data["rtmp_address"] == "rtmp://172.20.10.3:1935"


def test_api_stream_status_display_only_never_crashes(client):
    data = client.get("/api/stream/status").get_json()
    assert data["pov_state"] in ("LIVE", "OFFLINE", "UNKNOWN")


def test_api_mediamtx_config_downloads_yaml(client, monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "10.0.0.9")  # example test value
    resp = client.get("/api/mediamtx/config")
    assert resp.status_code == 200
    assert b"rtmpAddress" in resp.data
    assert b"attachment" in resp.headers.get("Content-Disposition", "").encode()


def test_api_inspections_empty_initially(client):
    data = client.get("/api/inspections").get_json()
    assert data["count"] == 0 and data["inspections"] == []


def test_api_latest_null_initially(client):
    assert client.get("/api/inspection/latest").get_json() is None


# -- analysis flow ----------------------------------------------------

def test_upload_no_crack_flow(client):
    stubs.stub_no_crack(inference_core)
    resp = client.post(
        "/detect",
        data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    page = client.get(resp.headers["Location"])
    assert page.status_code == 200
    assert b"NO CRACK DETECTED" in page.data
    # No red-highlighted figure caption is shown for a no-crack result.
    assert b"Red-highlighted" not in page.data


def test_api_inspect_crack_flow_serves_only_two_images(client):
    stubs.stub_one_crack(inference_core)
    j = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    ).get_json()

    assert j["status"] == "CRACK DETECTED"
    # Never leak prohibited model metrics (coordinates are allowed - they are
    # shown in the inspection details).
    for banned in ("confidence", "num_instances", "crack_image_names", "fps"):
        assert banned not in j
    urls = j["urls"]
    assert set(urls.keys()) == {"original", "highlighted"}  # no 'cracks'
    assert client.get(urls["original"]).status_code == 200
    assert client.get(urls["highlighted"]).status_code == 200


def test_result_page_crack_shows_original_and_analyzed_side_by_side(client):
    stubs.stub_one_crack(inference_core)
    j = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    ).get_json()
    page = client.get(f"/inspection/{j['id']}")
    body = page.data.decode()
    assert page.status_code == 200
    assert "CRACK DETECTED" in body
    # Side-by-side comparison: original on the left, analyzed/highlighted on
    # the right, each with its own clear label.
    assert body.count('class="shot-img"') == 2
    assert "Original Image" in body
    assert "Crack Detected Image" in body
    assert f"/api/inspection/{j['id']}/original" in body
    assert f"/api/inspection/{j['id']}/highlighted" in body
    # crack feedback: pulsing banner class + audio hook are wired in
    assert "status-alarm" in body and "AudioContext" in body
    # the result page offers a Download Results link
    assert f"/inspection/{j['id']}/download" in body


def test_download_results_report_contains_details_and_both_images(client):
    stubs.stub_one_crack(inference_core)
    j = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    ).get_json()

    resp = client.get(f"/inspection/{j['id']}/download")
    assert resp.status_code == 200
    # served as a single downloadable report
    assert "attachment" in resp.headers.get("Content-Disposition", "")
    assert f"matanglawin_inspection_{j['id']}" in resp.headers.get("Content-Disposition", "")
    body = resp.data.decode()
    # inspection details
    assert "CRACK DETECTED" in body
    for field in ("Status", "Latitude", "Longitude", "Date / Time", "Source"):
        assert field in body
    # both images embedded inline (base64 data URIs), each labelled
    assert "Original Image" in body and "Crack Detected Image" in body
    assert body.count("data:image/jpeg;base64,") == 2


def test_download_results_missing_inspection_returns_404(client):
    assert client.get("/inspection/999999/download").status_code == 404


def test_api_inspect_rejects_non_image(client):
    resp = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(b"not an image"), "t.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_upload_missing_file_redirects(client):
    resp = client.post("/detect", data={}, content_type="multipart/form-data")
    assert resp.status_code in (302, 303)


# -- prohibited surfaces are absent ----------------------------------

def test_no_crack_only_or_report_routes_exist(client):
    import app as appmod
    rules = [str(r) for r in appmod.app.url_map.iter_rules()]
    assert not any("/crack/" in r or r.endswith("/crack") for r in rules)
    assert not any("report" in r for r in rules)
    assert not any("gps" in r for r in rules)
    assert not any("preview" in r for r in rules)
    # crack-only image route must 404 (it does not exist).
    stubs.stub_one_crack(inference_core)
    j = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(stubs.jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    ).get_json()
    assert client.get(f"/api/inspection/{j['id']}/crack/crack_001.png").status_code == 404
