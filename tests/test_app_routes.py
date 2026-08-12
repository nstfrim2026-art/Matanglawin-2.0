"""
Tests for app.py routes in the photo-inspection architecture.

Each test gets an isolated data dir (via monkeypatching app globals) and
a stubbed YOLO (via inference_core.predict_masks) so no real model or
network is needed, and detection is deterministic.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402


class _Conf:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.v)


class _Boxes:
    def __init__(self, c):
        self.conf = _Conf(c)


class _MaskData:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Masks:
    def __init__(self, a):
        self._a = a

    @property
    def data(self):
        return _MaskData(self._a)

    def __len__(self):
        return self._a.shape[0]


class _Result:
    def __init__(self, masks, boxes, orig):
        self.masks = masks
        self.boxes = boxes
        self.orig_img = orig


def _stub_no_crack():
    inference_core.predict_masks = lambda *a, **k: _Result(None, None, a[0] if a else None)


def _stub_one_crack():
    def fake(img, *a, **k):
        m = np.zeros(img.shape[:2], dtype=np.float32)
        h, w = img.shape[:2]
        m[h // 4 : h // 2, w // 4 : 3 * w // 4] = 1.0
        return _Result(_Masks(np.array([m])), _Boxes([0.9]), img)

    inference_core.predict_masks = fake


def _jpg_bytes(value=120, size=64):
    import cv2

    img = np.full((size, size, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app as appmod

    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(appmod, "UPLOAD_TMP_DIR", tmp_path / "upload_tmp")
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "inspections")
    monkeypatch.setattr(appmod, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(appmod, "IMPORT_DIR", tmp_path / "import")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "inspections.db")
    for attr in ["UPLOAD_TMP_DIR", "INSPECTIONS_DIR", "REPORT_DIR", "IMPORT_DIR"]:
        getattr(appmod, attr).mkdir(parents=True, exist_ok=True)

    # reset lazy singletons for a clean slate
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    monkeypatch.setattr(appmod, "_watcher", None)

    _stub_no_crack()
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


# -- pages ------------------------------------------------------------

def test_index_renders(client):
    assert client.get("/").status_code == 200


def test_dashboard_renders_with_clean_pov(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Live Drone POV" in resp.data
    assert b"controller camera" in resp.data  # capture instruction present


def test_inspections_page_renders(client):
    resp = client.get("/inspections")
    assert resp.status_code == 200
    assert b"Recorded Inspections" in resp.data


# -- network / stream / gps APIs -------------------------------------

def test_api_network(client):
    data = client.get("/api/network").get_json()
    assert "host_ip" in data
    assert data["rtsp_url"].startswith("rtsp://localhost:")


def test_api_stream_status_is_display_only_and_never_crashes(client):
    data = client.get("/api/stream/status").get_json()
    assert data["pov_state"] in ("LIVE", "OFFLINE", "UNKNOWN")


def test_api_gps_unavailable_by_default(client):
    assert client.get("/api/gps").get_json()["available"] is False


def test_api_inspections_empty_initially(client):
    data = client.get("/api/inspections").get_json()
    assert data["count"] == 0 and data["inspections"] == []


def test_api_latest_null_initially(client):
    assert client.get("/api/inspection/latest").get_json() is None


# -- upload / analysis flow ------------------------------------------

def test_upload_no_crack_flow(client):
    _stub_no_crack()
    resp = client.post(
        "/detect",
        data={"image": (io.BytesIO(_jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    loc = resp.headers["Location"]
    page = client.get(loc)
    assert page.status_code == 200
    assert b"NO CRACK DETECTED" in page.data


def test_api_inspect_crack_flow_and_images_served(client):
    _stub_one_crack()
    j = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(_jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    ).get_json()
    assert j["status"] == "CRACK DETECTED"
    assert j["num_instances"] >= 1
    assert "confidence" not in j  # never leak confidence
    urls = j["urls"]
    assert client.get(urls["original"]).status_code == 200
    assert client.get(urls["highlighted"]).status_code == 200
    assert len(urls["cracks"]) == j["num_instances"]
    for u in urls["cracks"]:
        assert client.get(u).status_code == 200


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


def test_crack_image_route_blocks_path_traversal(client):
    _stub_one_crack()
    j = client.post(
        "/api/inspect",
        data={"image": (io.BytesIO(_jpg_bytes()), "t.jpg")},
        content_type="multipart/form-data",
    ).get_json()
    iid = j["id"]
    # a filename not belonging to this inspection must 404
    assert client.get(f"/api/inspection/{iid}/crack/notreal.png").status_code == 404


# -- reports ----------------------------------------------------------

def test_full_report_with_zero_inspections(client):
    resp = client.get("/reports/full.pdf")
    assert resp.status_code == 200
    assert resp.data[:5] == b"%PDF-"


def test_single_report_404_for_missing(client):
    assert client.get("/reports/inspection/99999.pdf").status_code == 404


# -- removed live-detection surface ----------------------------------

def test_no_live_detection_routes_exist(client):
    import app as appmod

    rules = [str(r) for r in appmod.app.url_map.iter_rules()]
    # No annotated full-frame preview, no live capture serving route.
    assert not any("preview" in r for r in rules)
    assert not any(r.startswith("/data/captures") for r in rules)
    assert not any("stream/status" in r and "pov" in r for r in rules)  # sanity
    assert client.get("/stream/preview.jpg").status_code == 404
