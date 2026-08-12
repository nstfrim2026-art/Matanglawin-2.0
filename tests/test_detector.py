"""
Tests for detector.py - per-frame YOLO11-seg wrapper, minimum-area
filtering, and mask-based (not bbox-based) crack extraction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
import detector  # noqa: E402


class _FakeConfArr:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.v)


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _FakeConfArr(confs)


class _FakeMasksT:
    def __init__(self, d):
        self.d = d

    def cpu(self):
        return self

    def numpy(self):
        return self.d


class _FakeMasks:
    def __init__(self, arr):
        self._arr = arr

    @property
    def data(self):
        return _FakeMasksT(self._arr)

    def __len__(self):
        return self._arr.shape[0]


class _FakeResult:
    def __init__(self, masks, boxes, orig_img):
        self.masks = masks
        self.boxes = boxes
        self.orig_img = orig_img


def _frame(size=200, value=200):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _square_mask(size=200, box=(50, 50, 90, 90)):
    mask = np.zeros((size, size), dtype=np.float32)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _diagonal_mask(size=200, box=(50, 50, 90, 90)):
    """A thin diagonal line inside `box`, covering a small fraction of it."""
    mask = np.zeros((size, size), dtype=np.float32)
    x1, y1, x2, y2 = box
    n = min(x2 - x1, y2 - y1)
    for i in range(n):
        mask[y1 + i, x1 + i] = 1.0
    return mask


@pytest.fixture(autouse=True)
def restore_predict_masks():
    original = inference_core.predict_masks
    yield
    inference_core.predict_masks = original


def test_no_masks_returns_no_crack(monkeypatch):
    frame = _frame()
    fake_result = _FakeResult(None, None, frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector()
    result = det.process_frame(frame)
    assert result.has_crack is False
    assert result.instances == []


def test_small_mask_is_filtered_as_noise():
    frame = _frame()
    mask = _square_mask(box=(0, 0, 5, 5))  # tiny - below default min_area_px
    fake_result = _FakeResult(_FakeMasks(np.array([mask])), _FakeBoxes([0.9]), frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector(min_area_px=150)
    result = det.process_frame(frame)
    assert result.has_crack is False


def test_valid_mask_produces_an_instance_with_its_own_mask_field():
    frame = _frame()
    mask = _square_mask()
    fake_result = _FakeResult(_FakeMasks(np.array([mask])), _FakeBoxes([0.87]), frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector(min_area_px=100)
    result = det.process_frame(frame)
    assert result.has_crack is True
    assert result.num_instances == 1
    inst = result.instances[0]
    assert inst.confidence == 0.87
    assert inst.bbox == (50, 50, 90, 90)
    assert inst.mask.shape == (40, 40)


def test_process_frame_never_produces_a_full_frame_overlay():
    """
    DetectionResult must not carry any full-frame annotated image - see
    the requirement that detection stays invisible to the live POV.
    """
    frame = _frame()
    mask = _square_mask()
    fake_result = _FakeResult(_FakeMasks(np.array([mask])), _FakeBoxes([0.9]), frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector(min_area_px=100)
    result = det.process_frame(frame)
    assert not hasattr(result, "overlay_frame")
    assert not hasattr(result, "union_mask")


def test_none_frame_is_handled_safely():
    det = detector.CrackDetector()
    result = det.process_frame(None)
    assert result.has_crack is False
    assert result.instances == []


def test_extract_crack_only_suppresses_background_outside_mask():
    """
    For a mask that covers only a thin diagonal strip of its bounding
    box, extract_crack_only() must black out everything else - the
    result must NOT be "a large rectangular piece of concrete".
    """
    frame = _frame(value=200)  # uniform gray background
    mask = _diagonal_mask()
    fake_result = _FakeResult(_FakeMasks(np.array([mask])), _FakeBoxes([0.9]), frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector(min_area_px=10)
    result = det.process_frame(frame)
    assert result.has_crack is True
    inst = result.instances[0]

    crack_only = detector.extract_crack_only(frame, inst)
    nonblack_fraction = (crack_only.sum(axis=2) > 0).mean()
    assert nonblack_fraction < 0.3  # thin diagonal line, not a filled square


def test_extract_crack_only_keeps_full_region_for_a_solid_mask():
    """A fully-filled mask should retain (almost) all of its crop's pixels."""
    frame = _frame(value=200)
    mask = _square_mask()
    fake_result = _FakeResult(_FakeMasks(np.array([mask])), _FakeBoxes([0.9]), frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector(min_area_px=10)
    result = det.process_frame(frame)
    inst = result.instances[0]

    crack_only = detector.extract_crack_only(frame, inst)
    nonblack_fraction = (crack_only.sum(axis=2) > 0).mean()
    assert nonblack_fraction > 0.5


def test_crack_overlay_crop_preserves_background_and_is_small():
    """
    Unlike extract_crack_only, the internal overlay crop keeps the
    background (with a red mask blended on top) - but it must still be
    a small crop around the instance, never the full frame.
    """
    frame = _frame(size=400, value=200)
    mask = _square_mask(size=400, box=(50, 50, 90, 90))
    fake_result = _FakeResult(_FakeMasks(np.array([mask])), _FakeBoxes([0.9]), frame)
    inference_core.predict_masks = lambda *a, **k: fake_result

    det = detector.CrackDetector(min_area_px=10)
    result = det.process_frame(frame)
    inst = result.instances[0]

    overlay = detector.crack_overlay_crop(frame, inst)
    assert overlay.shape[0] < frame.shape[0]
    assert overlay.shape[1] < frame.shape[1]
    nonblack_fraction = (overlay.sum(axis=2) > 0).mean()
    assert nonblack_fraction > 0.9  # background preserved, unlike extract_crack_only
