"""
inference_core.py - Shared crack-segmentation inference logic.

This factors out the core prediction + overlay-drawing logic that
infer_overlay.py uses on the command line, so it can also be reused by
the Flask web app (app.py) without duplicating code.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# `ultralytics` (and its `torch`/`torchvision` dependency) is a large,
# slow-to-import package. Importing it lazily, inside get_model(), lets
# the pure-OpenCV helpers in this module (draw_mask_overlay,
# union_mask_from_result) be imported and unit-tested without requiring
# the full YOLO stack to be installed.
_YOLO_CLASS = None

# Cache loaded models by weights path so we don't reload the .pt file on
# every request (loading is the slow part).
_MODEL_CACHE = {}


def get_model(weights: str = "best.pt"):
    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        from ultralytics import YOLO as _YOLO
        _YOLO_CLASS = _YOLO

    weights = str(weights)
    if weights not in _MODEL_CACHE:
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        _MODEL_CACHE[weights] = _YOLO_CLASS(weights)
    return _MODEL_CACHE[weights]


def draw_mask_overlay(
    image: np.ndarray,
    union_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
    outline: int = 2,
) -> np.ndarray:
    """
    Blend a binary `union_mask` (H, W) onto `image` (BGR) as a
    semi-transparent fill plus a contour outline, following the exact
    mask shape (no bounding boxes). Shared by both the single-image
    overlay path and the live-frame detector so the two never drift
    out of sync.
    """
    h, w = image.shape[:2]
    if union_mask.shape != (h, w):
        union_mask = cv2.resize(union_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros_like(image)
    color_layer[:] = color
    mask_bool = union_mask.astype(bool)
    blended = image.copy()
    blended[mask_bool] = cv2.addWeighted(
        image, 1 - alpha, color_layer, alpha, 0
    )[mask_bool]

    if outline > 0:
        contours, _ = cv2.findContours(
            union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended, contours, -1, color, outline)

    return blended


def predict_masks(
    image,
    weights: str = "best.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
):
    """
    Run YOLO11-seg on an already-loaded image (numpy array, BGR) OR a
    path, and return the raw `ultralytics` Result object for the first
    (only) image. This is the single inference call shared by the
    image-upload overlay flow and the live-frame detector.
    """
    model = get_model(weights)
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        retina_masks=True,
        device=device,
        verbose=False,
    )
    return results[0]


def union_mask_from_result(result, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Collapse all instance masks in `result` into one binary union mask, or None."""
    if result.masks is None or len(result.masks) == 0:
        return None
    masks = result.masks.data.cpu().numpy()  # (N, H, W) in [0,1]
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)
    h, w = shape_hw
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return union


def run_overlay(
    image_path: str,
    out_path: str,
    weights: str = "best.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 0, 255),  # B, G, R
    outline: int = 2,
    device: Optional[str] = None,
) -> dict:
    """
    Run YOLO11-seg crack prediction on `image_path` and save an overlay
    image (mask fill + contour outline) to `out_path`.

    Returns a dict: {"num_instances": int, "out_path": str}

    (Unchanged behavior from the original implementation - this is the
    function app.py's single-image upload flow has always called. It now
    delegates its pixel-drawing work to `draw_mask_overlay` so the same
    code path is exercised by the live-video detector, but the inputs,
    outputs, and saved file format are identical to before.)
    """
    result = predict_masks(str(image_path), weights=weights, conf=conf, imgsz=imgsz, device=device)

    image = result.orig_img.copy()  # BGR
    h, w = image.shape[:2]

    union = union_mask_from_result(result, (h, w))
    if union is None:
        cv2.imwrite(str(out_path), image)
        return {"num_instances": 0, "out_path": str(out_path)}

    blended = draw_mask_overlay(image, union, color=color, alpha=alpha, outline=outline)

    cv2.imwrite(str(out_path), blended)
    return {"num_instances": int(len(result.masks)), "out_path": str(out_path)}
