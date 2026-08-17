"""
inference_core.py - Shared crack-segmentation inference logic.

This factors out the core prediction + overlay-drawing logic so it can be
reused by the CLI (infer_overlay.py), the central InspectionService, and
anything else, WITHOUT duplicating inference code (see requirement:
"do not create duplicate inference implementations").

Only two artifacts are ever produced from a photo's masks:
    * the untouched original image, and
    * a red-highlighted image (segmentation masks painted red).
There is deliberately no crack-only / cropped-crack / separate-mask
output here, and no bounding boxes are ever drawn.
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

# Tighten the red overlay to the actual dark crack line (see
# refine_crack_mask). On by default; disable with MATANGLAWIN_REFINE=0.
DEFAULT_REFINE = True


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
    semi-transparent fill plus a contour outline, following the EXACT
    mask shape (no bounding boxes). This is the single red-highlight
    renderer used everywhere, so the manual-upload path and the DJI
    photo-import path always produce identical-looking results.
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


def refine_crack_mask(
    image: np.ndarray,
    mask: np.ndarray,
    kernel_frac: float = 0.012,
    min_blackhat: int = 10,
) -> np.ndarray:
    """
    Tighten a (often fat/blobby) segmentation `mask` so it hugs the ACTUAL
    dark crack line, instead of painting a wide band of surrounding
    surface red.

    A crack is a thin DARK feature on a lighter wall. A morphological
    black-hat (closing minus the image) lights up exactly those dark thin
    structures. Inside the detected `mask` we keep only the pixels that are
    part of that dark structure (Otsu threshold on the in-mask black-hat
    response), give the line a little body, reconnect small gaps, and clamp
    the result to never exceed the original detection.

    Safeguard: if there is no clear dark structure inside the detection
    (peak black-hat response below `min_blackhat`, e.g. a bright/low-contrast
    crack or a shadowed region), the refinement is considered unreliable and
    the ORIGINAL mask is returned unchanged. This keeps the refinement a
    fidelity improvement that never erases a genuine detection. Note the
    refined mask can legitimately be a SMALL fraction of a fat input mask -
    that is exactly the point when the input over-covered the crack.

    Purely photometric + morphological: it only shrinks the mask within the
    already-detected region, so it can't invent new detections and never
    touches the stored original photo or the live POV.
    """
    if mask is None:
        return mask
    mask_u8 = mask.astype(np.uint8)
    total = int(mask_u8.sum())
    if total == 0:
        return mask_u8

    h, w = mask_u8.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.shape[:2] != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    k = int(max(9, round(kernel_frac * min(h, w))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    m = mask_u8.astype(bool)
    vals = blackhat[m]
    if vals.size == 0 or int(vals.max()) < min_blackhat:
        return mask_u8  # no dark structure to lock onto -> keep original

    otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(float(otsu), float(min_blackhat))

    core = ((blackhat >= thresh) & m).astype(np.uint8)
    # Give the thin line a little body and reconnect small along-crack gaps,
    # then clamp back inside the original detection.
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    core = core & mask_u8

    if int(core.sum()) == 0:
        return mask_u8  # nothing survived -> don't erase the detection
    return core


def enhance_for_detection(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    tile_grid: int = 8,
    unsharp: bool = True,
) -> np.ndarray:
    """
    Return a contrast-boosted COPY of `image` (BGR) for the model to look
    at. This makes faint / hairline cracks stand out (CLAHE on the L
    channel + a light unsharp mask) so the segmenter is more likely to
    pick them up.

    IMPORTANT: this is a DETECTION-ONLY transform. The caller feeds the
    enhanced copy to the model, but the photo that is stored and shown to
    the operator remains the untouched original, and the red highlight is
    drawn on that original. The transform is purely photometric (no
    resize/warp), so the masks it produces align pixel-for-pixel with the
    original. The input array is never modified in place.
    """
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if unsharp:
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    return out


def predict_masks(
    image,
    weights: str = "best.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    augment: bool = False,
):
    """
    Run YOLO11-seg on an already-loaded image (numpy array, BGR) OR a
    path, and return the raw `ultralytics` Result object for the first
    (only) image. This is the single inference call shared by every
    still-photo analysis path. It is NEVER called on a video stream.

    `imgsz` controls the inference resolution (higher keeps thin cracks
    resolvable); `augment` enables test-time augmentation (multi-scale +
    flips) for a further recall boost at extra compute cost.
    """
    model = get_model(weights)
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        retina_masks=True,
        device=device,
        augment=augment,
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
    Run YOLO11-seg crack prediction on `image_path` and save a
    red-highlighted overlay image (mask fill + contour outline) to
    `out_path`. Kept for the standalone CLI (infer_overlay.py).

    Returns a dict: {"num_instances": int, "out_path": str}
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
