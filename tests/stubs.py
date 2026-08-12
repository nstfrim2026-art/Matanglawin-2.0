"""
tests/stubs.py - Lightweight fakes for the YOLO segmentation result.

These let the whole analysis pipeline (detector -> inspection_service ->
photo_import / app routes) be exercised deterministically WITHOUT loading
the real ultralytics/torch stack or the best.pt weights. We monkeypatch
``inference_core.predict_masks`` to return one of these fake results.
"""

from __future__ import annotations

import numpy as np


class _MaskData:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Masks:
    def __init__(self, arr):
        self._arr = arr

    @property
    def data(self):
        return _MaskData(self._arr)

    def __len__(self):
        return self._arr.shape[0]


class _Result:
    def __init__(self, masks, orig):
        self.masks = masks
        self.orig_img = orig


def stub_no_crack(inference_core):
    """Make predict_masks report zero cracks."""
    inference_core.predict_masks = lambda image, *a, **k: _Result(None, _as_img(image))


def stub_one_crack(inference_core):
    """Make predict_masks report a single sizable crack mask."""
    def fake(image, *a, **k):
        img = _as_img(image)
        h, w = img.shape[:2]
        m = np.zeros((h, w), dtype=np.float32)
        m[h // 4:h // 2, w // 4:3 * w // 4] = 1.0
        return _Result(_Masks(np.array([m])), img)

    inference_core.predict_masks = fake


def stub_tiny_speck(inference_core, area_px=9):
    """Make predict_masks report only a tiny noise speck (below min-area)."""
    def fake(image, *a, **k):
        img = _as_img(image)
        h, w = img.shape[:2]
        m = np.zeros((h, w), dtype=np.float32)
        side = max(1, int(area_px ** 0.5))
        m[0:side, 0:side] = 1.0
        return _Result(_Masks(np.array([m])), img)

    inference_core.predict_masks = fake


def _as_img(image):
    """predict_masks accepts a path or an array; tests always pass arrays."""
    if isinstance(image, np.ndarray):
        return image
    # Fallback for a path (not used by our tests): a small gray image.
    return np.full((64, 64, 3), 120, dtype=np.uint8)


def jpg_bytes(value=120, size=64):
    import cv2

    img = np.full((size, size, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def write_jpg(path, value=120, size=64):
    import cv2

    img = np.full((size, size, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path
