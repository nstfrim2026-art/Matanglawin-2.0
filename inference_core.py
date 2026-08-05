"""
inference_core.py - Shared crack-segmentation inference logic.

This factors out the core prediction + overlay-drawing logic that
infer_overlay.py uses on the command line, so it can also be reused by
the Flask web app (app.py) without duplicating code.
"""

import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Cache loaded models by weights path so we don't reload the .pt file on
# every request (loading is the slow part).
_MODEL_CACHE = {}

# Serializes model loading and all model.predict() calls.  YOLO/PyTorch
# models are not thread-safe for concurrent inference on a single instance,
# and the cache population itself has a TOCTOU race without this lock.
_MODEL_LOCK = threading.Lock()


def get_model(weights: str = "best.pt") -> YOLO:
    weights = str(weights)
    with _MODEL_LOCK:
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

    with _MODEL_LOCK:
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


def _classify_crack(area_px: int, h: int, w: int) -> str:
    """Classify a crack based on its mask area relative to total frame area."""
    frame_area = h * w
    if frame_area == 0:
        return "Hairline"
    area_ratio = area_px / frame_area
    if area_ratio < 0.002:
        return "Hairline"
    elif area_ratio < 0.01:
        return "Surface-Level"
    else:
        return "Structural"


def annotate_frame(
    frame_bgr: np.ndarray,
    weights: str = "best.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (0, 0, 255),  # B, G, R
    outline: int = 2,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, bool, list]:
    """
    Run YOLO11-seg crack prediction on a single BGR frame (e.g. one frame
    pulled from a live camera/video source) and return an annotated copy.

    This mirrors run_overlay() above - same mask-fill + contour-outline
    drawing logic - but operates entirely in memory on a frame instead of
    reading/writing an image file, so it's cheap enough to call on every
    frame of a continuous video stream.

    Returns:
        (annotated_frame_bgr, crack_present, crack_metadata_list) where
        crack_present is True if at least one crack instance mask was found
        in this frame, and crack_metadata_list is a list of dicts with
        per-detection analysis (classification, confidence, area, bbox,
        estimated length and width in pixels).
    """
    model = get_model(weights)

    with _MODEL_LOCK:
        results = model.predict(
            source=frame_bgr,
            conf=conf,
            imgsz=imgsz,
            retina_masks=True,
            device=device,
            verbose=False,
        )
    result = results[0]

    image = frame_bgr.copy()  # BGR
    h, w = image.shape[:2]

    if result.masks is None or len(result.masks) == 0:
        return image, False, []

    masks = result.masks.data.cpu().numpy()  # (N, H, W) in [0,1]
    num_instances = len(masks)

    # Build per-instance metadata
    crack_metadata_list = []
    for i in range(num_instances):
        mask_i = masks[i]
        # Resize mask to original frame dimensions if needed
        if mask_i.shape != (h, w):
            mask_i = cv2.resize(mask_i, (w, h), interpolation=cv2.INTER_NEAREST)

        binary_mask = (mask_i > 0.5).astype(np.uint8)
        area_px = int(binary_mask.sum())

        # Confidence score from YOLO boxes
        confidence = float(result.boxes.conf[i].cpu().numpy())

        # Bounding box from YOLO boxes
        bbox = result.boxes.xyxy[i].cpu().numpy().tolist()

        # Compute contour bounding rect for length/width estimates
        contours_i, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_i:
            # Use the largest contour for measurement
            largest = max(contours_i, key=cv2.contourArea)
            _, _, cw, ch = cv2.boundingRect(largest)
            estimated_length_px = float(max(cw, ch))
            estimated_width_px = float(min(cw, ch))
        else:
            estimated_length_px = 0.0
            estimated_width_px = 0.0

        crack_metadata_list.append({
            "classification": _classify_crack(area_px, h, w),
            "confidence": confidence,
            "area_px": area_px,
            "bbox": bbox,
            "estimated_length_px": estimated_length_px,
            "estimated_width_px": estimated_width_px,
        })

    # Draw overlay (union of all masks)
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

    return blended, True, crack_metadata_list
