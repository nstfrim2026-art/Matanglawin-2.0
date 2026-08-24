"""
Tests for the dashboard summary counters: INSPECTIONS / CRACKS DETECTED /
CLEAR. Values come straight from the database and update automatically as
new inspections are created. Segmentation is stubbed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from bridge_status import BridgeStatus  # noqa: E402
from inspection_db import InspectionDB, STATUS_CRACK, STATUS_NO_CRACK  # noqa: E402
from tests import stubs  # noqa: E402


# -- DB counts (unit) ------------------------------------------------

def test_counts_from_db_ten_six_four(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    for _ in range(6):
        db.add_inspection(timestamp="t", status=STATUS_CRACK, source="import")
    for _ in range(4):
        db.add_inspection(timestamp="t", status=STATUS_NO_CRACK, source="import")
    c = db.counts()
    assert c == {"total": 10, "cracks": 6, "clear": 4}


def test_counts_empty_db_is_zeroed(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    assert db.counts() == {"total": 0, "cracks": 0, "clear": 0}


# -- summary endpoint + auto-update ----------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(appmod, "UPLOAD_TMP_DIR", tmp_path / "upload_tmp")
    monkeypatch.setattr(appmod, "IMPORT_TMP_DIR", tmp_path / "import_tmp")
    monkeypatch.setattr(appmod, "INSPECTIONS_DIR", tmp_path / "inspections")
    monkeypatch.setattr(appmod, "IMPORT_DIR", tmp_path / "import")
    monkeypatch.setattr(appmod, "IMPORT_LEDGER_PATH", tmp_path / "import_http_state.json")
    monkeypatch.setattr(appmod, "DB_PATH", tmp_path / "inspections.db")
    for attr in ["UPLOAD_TMP_DIR", "IMPORT_TMP_DIR", "INSPECTIONS_DIR", "IMPORT_DIR"]:
        getattr(appmod, attr).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(appmod, "_db", None)
    monkeypatch.setattr(appmod, "_service", None)
    monkeypatch.setattr(appmod, "_watcher", None)
    monkeypatch.setattr(appmod, "_import_ledger", None)
    monkeypatch.setattr(appmod, "bridge_status", BridgeStatus())
    stubs.stub_no_crack(inference_core)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


def _import(client, value, crack):
    (stubs.stub_one_crack if crack else stubs.stub_no_crack)(inference_core)
    return client.post("/api/import", data=stubs.jpg_bytes(value=value),
                       content_type="image/jpeg",
                       headers={"X-Filename": f"DJI_{value}.jpg"})


def test_summary_endpoint_reflects_db_and_updates(client):
    # start empty
    assert client.get("/api/inspections/summary").get_json() == {
        "total": 0, "cracks": 0, "clear": 0}

    # two cracks + one clear (unique bytes each so nothing is deduped)
    _import(client, 1, crack=True)
    _import(client, 2, crack=True)
    _import(client, 3, crack=False)
    assert client.get("/api/inspections/summary").get_json() == {
        "total": 3, "cracks": 2, "clear": 1}

    # a new inspection updates the counters automatically
    _import(client, 4, crack=False)
    assert client.get("/api/inspections/summary").get_json() == {
        "total": 4, "cracks": 2, "clear": 2}


def test_summary_counters_live_in_the_inspection_section_not_the_dashboard(client):
    # Inspection status (count / cracks / clear) belongs in the Inspection
    # section, NOT the Live Dashboard.
    dash = client.get("/").data.decode()
    assert 'id="summary-total"' not in dash
    assert "Cracks Detected" not in dash

    insp = client.get("/inspections").data.decode()
    assert 'id="summary-total"' in insp
    assert 'id="summary-cracks"' in insp
    assert 'id="summary-clear"' in insp
    assert "Cracks Detected" in insp
    assert "Clear" in insp


def test_inspections_page_polls_summary_endpoint():
    html = (Path(__file__).resolve().parent.parent / "templates" / "inspections.html").read_text()
    assert "/api/inspections/summary" in html
