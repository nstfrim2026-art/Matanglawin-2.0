"""
Tests for detector.py - single-image segmentation, no crack-only output,
no bounding boxes, and a noise-area filter. Uses a stubbed predict_masks
so no real model is loaded.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from tests import stubs  # noqa: E402
from detector import CrackDetector, build_union_mask  # noqa: E402


def _img():
    return np.full((100, 100, 3), 120, dtype=np.uint8)


def test_no_crack(monkeypatch):
    stubs.stub_no_crack(inference_core)
    det = CrackDetector(weights="x")
    res = det.process_frame(_img())
    assert res.has_crack is False
    assert res.num_instances == 0
    assert res.union_mask((100, 100)) is None


def test_one_crack_and_union_mask(monkeypatch):
    stubs.stub_one_crack(inference_core)
    det = CrackDetector(weights="x")
    res = det.process_frame(_img())
    assert res.has_crack is True
    assert res.num_instances == 1
    union = res.union_mask((100, 100))
    assert union is not None
    assert union.shape == (100, 100)
    assert int(union.sum()) > 0


def test_tiny_speck_filtered_as_noise(monkeypatch):
    stubs.stub_tiny_speck(inference_core, area_px=4)
    det = CrackDetector(weights="x", min_area_px=150)
    res = det.process_frame(_img())
    assert res.has_crack is False  # speck below min area -> not a crack


def test_empty_frame_is_safe():
    det = CrackDetector(weights="x")
    res = det.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res.has_crack is False


def test_detector_exposes_no_bbox_or_confidence_fields(monkeypatch):
    stubs.stub_one_crack(inference_core)
    det = CrackDetector(weights="x")
    res = det.process_frame(_img())
    inst = res.instances[0]
    # A CrackInstance carries only area + mask - never bbox/confidence.
    assert set(vars(inst).keys()) == {"area_px", "mask"}
