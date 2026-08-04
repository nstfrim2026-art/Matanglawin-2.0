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
from ultralytics import YOLO

# Cache loaded models by weights path so we don't reload the .pt file on
# every request (loading is the slow part).
_MODEL_CACHE = {}


def get_model(weights: str = "best.pt") -> YOLO:
    weights = str(weights)
    if weights not in _MODEL_CACHE:
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


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
    """
    model = get_model(weights)

    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        retina_masks=True,
        device=device,
        verbose=False,
    )
    result = results[0]

    image = result.orig_img.copy()  # BGR
    h, w = image.shape[:2]

    if result.masks is None or len(result.masks) == 0:
        cv2.imwrite(str(out_path), image)
        return {"num_instances": 0, "out_path": str(out_path)}

    masks = result.masks.data.cpu().numpy()  # (N, H, W) in [0,1]
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)

    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros_like(image)
    color_layer[:] = color
    mask_bool = union.astype(bool)
    blended = image.copy()
    blended[mask_bool] = cv2.addWeighted(
        image, 1 - alpha, color_layer, alpha, 0
    )[mask_bool]

    if outline > 0:
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color, outline)

    cv2.imwrite(str(out_path), blended)
    return {"num_instances": int(len(result.masks)), "out_path": str(out_path)}
