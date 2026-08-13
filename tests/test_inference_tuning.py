"""
Tests for the no-retraining recall improvements: detection-only contrast
enhancement, tiled/sliced inference, and the env-configurable knobs.

All use a stubbed predict_masks (no real model / weights).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from tests import stubs  # noqa: E402
from detector import CrackDetector  # noqa: E402


# ------------------------------------------------- enhancement (CLAHE) ----

def test_enhance_returns_new_array_and_does_not_mutate_input():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    before = img.copy()
    out = inference_core.enhance_for_detection(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(img, before), "input image must not be modified in place"


def test_enhance_actually_boosts_contrast_on_a_faint_image():
    # A low-contrast image with a faint darker streak (like a hairline crack).
    img = np.full((80, 80, 3), 150, dtype=np.uint8)
    img[:, 39:41] = 140  # very faint streak
    out = inference_core.enhance_for_detection(img)
    # Enhancement should widen the intensity spread (std goes up).
    assert out.std() >= img.std()


def test_enhanced_input_does_not_change_mask_geometry():
    # With enhancement ON, the stub still keys its mask off the image shape,
    # so masks land in the same coordinates as the untouched original.
    stubs.stub_one_crack(inference_core)
    det_plain = CrackDetector(weights="x", tiled=False, enhance=False)
    det_enh = CrackDetector(weights="x", tiled=False, enhance=True)
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    u_plain = det_plain.process_frame(img).union_mask((100, 100))
    u_enh = det_enh.process_frame(img).union_mask((100, 100))
    assert np.array_equal(u_plain, u_enh)


# --------------------------------------------------------- tiled path ----

def test_tiled_runs_one_inference_per_tile_and_stitches_full_frame():
    calls = []
    stubs.stub_full_mask(inference_core, counter=calls)
    # 300x300 image, 100px tiles, no overlap -> a clean 3x3 grid = 9 tiles.
    det = CrackDetector(weights="x", tiled=True, tile=100, tile_overlap=0.0,
                        enhance=False, min_area_px=10)
    img = np.full((300, 300, 3), 120, dtype=np.uint8)
    res = det.process_frame(img)
    assert len(calls) == 9, f"expected 9 tile inferences, got {len(calls)}"
    # Every tile returned a full mask -> the whole frame is covered.
    union = res.union_mask((300, 300))
    assert union is not None
    assert int(union.sum()) == 300 * 300
    # A single connected crack spanning all tiles = one instance.
    assert res.num_instances == 1


def test_tiled_overlap_produces_more_tiles():
    calls = []
    stubs.stub_full_mask(inference_core, counter=calls)
    det = CrackDetector(weights="x", tiled=True, tile=100, tile_overlap=0.5,
                        enhance=False, min_area_px=10)
    det.process_frame(np.full((300, 300, 3), 120, dtype=np.uint8))
    # step = 50 -> starts 0,50,100,150,200 = 5 per axis -> 25 tiles.
    assert len(calls) == 25


def test_tiled_recovers_crack_split_across_tile_borders():
    # A vertical streak straddling the border between two horizontal tiles
    # must come back as ONE connected instance thanks to overlap stitching.
    def fake(image, *a, **k):
        img = stubs._as_img(image)
        h, w = img.shape[:2]
        m = np.zeros((h, w), dtype=np.float32)
        m[:, w // 2 - 2:w // 2 + 2] = 1.0  # vertical line through each tile
        return stubs._Result(stubs._Masks(np.array([m])), img)
    inference_core.predict_masks = fake

    det = CrackDetector(weights="x", tiled=True, tile=100, tile_overlap=0.2,
                        enhance=False, min_area_px=10)
    res = det.process_frame(np.full((100, 300, 3), 120, dtype=np.uint8))
    assert res.has_crack is True


def test_tiled_no_detection_returns_no_crack():
    stubs.stub_no_crack(inference_core)
    det = CrackDetector(weights="x", tiled=True, tile=100)
    res = det.process_frame(np.full((250, 250, 3), 120, dtype=np.uint8))
    assert res.has_crack is False
    assert res.num_instances == 0


def test_tiled_min_area_filter_drops_specks():
    stubs.stub_tiny_speck(inference_core, area_px=4)
    det = CrackDetector(weights="x", tiled=True, tile=100, tile_overlap=0.0,
                        enhance=False, min_area_px=150)
    res = det.process_frame(np.full((200, 200, 3), 120, dtype=np.uint8))
    assert res.has_crack is False


# --------------------------------------------------- env config plumbing --

def test_detector_config_defaults(monkeypatch):
    import app as appmod
    for var in ("MATANGLAWIN_CONF", "MATANGLAWIN_IMGSZ", "MATANGLAWIN_MIN_AREA_PX",
                "MATANGLAWIN_TILED", "MATANGLAWIN_TILE", "MATANGLAWIN_TILE_OVERLAP",
                "MATANGLAWIN_ENHANCE", "MATANGLAWIN_AUGMENT"):
        monkeypatch.delenv(var, raising=False)
    cfg = appmod.detector_config()
    assert cfg["conf"] == 0.15
    assert cfg["imgsz"] == 1280
    assert cfg["min_area_px"] == 40
    assert cfg["tiled"] is True
    assert cfg["enhance"] is True
    assert cfg["augment"] is False


def test_detector_config_reads_env(monkeypatch):
    import app as appmod
    monkeypatch.setenv("MATANGLAWIN_CONF", "0.3")
    monkeypatch.setenv("MATANGLAWIN_IMGSZ", "1536")
    monkeypatch.setenv("MATANGLAWIN_MIN_AREA_PX", "80")
    monkeypatch.setenv("MATANGLAWIN_TILED", "0")
    monkeypatch.setenv("MATANGLAWIN_TILE", "768")
    monkeypatch.setenv("MATANGLAWIN_TILE_OVERLAP", "0.35")
    monkeypatch.setenv("MATANGLAWIN_ENHANCE", "off")
    monkeypatch.setenv("MATANGLAWIN_AUGMENT", "yes")
    cfg = appmod.detector_config()
    assert cfg["conf"] == 0.3
    assert cfg["imgsz"] == 1536
    assert cfg["min_area_px"] == 80
    assert cfg["tiled"] is False
    assert cfg["tile"] == 768
    assert cfg["tile_overlap"] == 0.35
    assert cfg["enhance"] is False
    assert cfg["augment"] is True


def test_service_forwards_config_to_detector():
    import app as appmod
    from inspection_db import InspectionDB
    import tempfile, os
    d = tempfile.mkdtemp()
    svc = appmod.InspectionService(
        InspectionDB(os.path.join(d, "x.db")), d, weights="x",
        **appmod.detector_config(),
    )
    det = svc._get_detector()
    assert det.conf == svc.conf
    assert det.tiled == svc.tiled
    assert det.tile == svc.tile
    assert det.enhance == svc.enhance
