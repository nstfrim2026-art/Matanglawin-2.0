"""
Tests for the offline local map tiles: the /maps/<z>/<x>/<y>.png route, the
/api/map/available check, and the map page's fallback wiring. No internet.
"""

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A tiny valid 1x1 transparent PNG (real image bytes; not a fake map tile).
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "MAP_TILES_DIR", tmp_path / "maps")
    (tmp_path / "maps").mkdir(parents=True, exist_ok=True)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c, tmp_path / "maps"


def _write_tile(tiles_dir: Path, z, x, y):
    p = tiles_dir / str(z) / str(x)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{y}.png").write_bytes(_PNG)


def test_tile_served_when_present(client):
    c, tiles = client
    _write_tile(tiles, 5, 10, 12)
    resp = c.get("/maps/5/10/12.png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.data == _PNG


def test_missing_tile_returns_404(client):
    c, _ = client
    assert c.get("/maps/9/9/9.png").status_code == 404


def test_tile_route_requires_integers(client):
    c, _ = client
    # Non-integer path components don't match the int converter -> 404,
    # so no path traversal is possible.
    assert c.get("/maps/1/2/notanint.png").status_code == 404
    assert c.get("/maps/a/b/c.png").status_code == 404


def test_map_available_false_then_true(client):
    c, tiles = client
    assert c.get("/api/map/available").get_json()["available"] is False
    _write_tile(tiles, 3, 4, 5)
    assert c.get("/api/map/available").get_json()["available"] is True


def test_map_page_uses_local_tiles_and_fallback(client):
    c, _ = client
    body = c.get("/map").data.decode()
    assert "/maps/{z}/{x}/{y}.png" in body          # Leaflet points at the local route
    assert "Map data unavailable" in body            # clean fallback text
    assert 'id="map-fallback"' in body
    assert "openstreetmap" not in body.lower()       # no external tile source
    assert "tile.osm" not in body.lower()
    assert "googleapis" not in body.lower()
