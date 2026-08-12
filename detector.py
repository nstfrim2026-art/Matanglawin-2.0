"""
detector.py - Single-image crack detection (YOLO11-seg).

Wraps the trained YOLO11-seg model (best.pt) and the shared mask-drawing
helpers (inference_core.py) to analyze ONE still image at a time. It is
used by inspection_service.py for both manually uploaded photos and
photos imported from the DJI controller.

    image (BGR)  ->  CrackDetector.process_frame()  ->  DetectionResult
                                                              |
                                                              v
                                                   inspection_service.py
                                        (build result images + DB record)

There is deliberately NO continuous/video code path here: the model is
invoked once per photo, not on a stream of frames. The live drone POV is
served straight from MediaMTX's WebRTC endpoint to the browser and never
touches this module.

Responsibilities:
    - Run YOLO11-seg on a BGR image, keeping each crack instance's own
      segmentation mask (not just its bounding box).
    - Basic detection-quality validation (confidence floor, minimum
      crack pixel area) to drop obvious noise. Confidence is used only
      internally for filtering and is never surfaced to the UI.
    - Helpers to turn masks into result images:
        * build_union_mask()      - full-image mask of all cracks, for
                                     the red-highlighted result photo.
        * extract_crack_only()    - a crack-only image (background
                                     suppressed via the mask), not a
                                     rectangular slab of surface.
        * crack_overlay_crop()    - a small red-mask crop of one crack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import inference_core

DEFAULT_WEIGHTS = "best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_MIN_AREA_PX = 150  # ignore specks smaller than this many mask pixels
DEFAULT_MARGIN_RATIO = 0.08  # small context margin around the mask's bbox, not the whole frame


@dataclass
class CrackInstance:
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixel coords
    area_px: int
    mask: np.ndarray  # binary (0/1) mask, shape (y2-y1, x2-x1) - aligned to `bbox`, NOT the full frame


@dataclass
class DetectionResult:
    has_crack: bool
    instances: List[CrackInstance]
    max_confidence: float

    @property
    def num_instances(self) -> int:
        return len(self.instances)


class CrackDetector:
    """
    Thin, stateless-per-call wrapper around the shared YOLO11-seg model.
    Safe to call `process_frame()` repeatedly from a video loop; the
    underlying model is cached by `inference_core.get_model()`.
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        min_area_px: int = DEFAULT_MIN_AREA_PX,
        device: Optional[str] = None,
    ):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.min_area_px = min_area_px
        self.device = device

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single decoded BGR image. Returns a
        DetectionResult describing every crack instance that passed
        basic quality validation.

        Note: this deliberately does NOT produce any full-frame overlay
        or preview image. Only per-instance masks (each cropped to its
        own tight bbox, not the full frame) are kept, for use by
        extract_crack_only()/crack_overlay_crop() below. The live video
        feed shown to users must never be touched by detection output -
        see module docstring.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(has_crack=False, instances=[], max_confidence=0.0)

        result = inference_core.predict_masks(
            frame, weights=self.weights, conf=self.conf, imgsz=self.imgsz, device=self.device,
        )

        h, w = frame.shape[:2]

        instances: List[CrackInstance] = []
        max_conf = 0.0

        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            confs = (
                result.boxes.conf.cpu().numpy()
                if result.boxes is not None and result.boxes.conf is not None
                else [1.0] * len(masks)
            )
            for mask, conf_val in zip(masks, confs):
                mask_bin = (mask > 0.5).astype(np.uint8)
                if mask_bin.shape != (h, w):
                    mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                area = int(mask_bin.sum())
                if area < self.min_area_px:
                    continue  # too small - likely noise, not a real crack
                bbox = _bbox_from_mask(mask_bin, frame_shape=(h, w))
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                mask_crop = mask_bin[y1:y2, x1:x2]  # keep only the mask region aligned to bbox
                instances.append(
                    CrackInstance(confidence=float(conf_val), bbox=bbox, area_px=area, mask=mask_crop)
                )
                max_conf = max(max_conf, float(conf_val))

        return DetectionResult(
            has_crack=len(instances) > 0,
            instances=instances,
            max_confidence=max_conf,
        )


def build_union_mask(instances: List[CrackInstance], shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Reassemble a single full-image binary mask (H, W) covering every
    crack instance, by placing each instance's bbox-aligned mask back at
    its position in the frame. Used to draw the red-highlighted result
    image (the full photo with ALL detected cracks highlighted), which
    is a result-page artifact only - never the live POV.
    """
    h, w = shape_hw
    union = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        x1, y1, x2, y2 = inst.bbox
        mh, mw = inst.mask.shape
        # Clamp in case of any off-by-one from resizing/rounding.
        y2c, x2c = min(y1 + mh, h), min(x1 + mw, w)
        sub = inst.mask[: y2c - y1, : x2c - x1]
        union[y1:y2c, x1:x2c] = np.maximum(union[y1:y2c, x1:x2c], sub)
    return union


def _bbox_from_mask(mask_bin: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box (x1,y1,x2,y2) around the nonzero pixels of a mask."""
    ys, xs = np.where(mask_bin > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def _bbox_with_margin(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand `bbox` by `margin_ratio` on each side, clamped to `frame_shape` (h, w)."""
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    if x2 <= x1 or y2 <= y1:
        return bbox
    return x1, y1, x2, y2


def extract_crack_only(
    frame: np.ndarray,
    instance: CrackInstance,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Produce a crack-ONLY image using the segmentation mask, not just a
    bounding box: crops a small region around `instance.bbox` (with a
    thin context margin so the crack isn't cut off at the pixel edge),
    then blacks out every pixel the mask doesn't cover, so the result
    contains as little non-crack surface as technically possible - not
    "a large rectangular piece of concrete" with a crack somewhere in
    it.

    This never touches the full frame's pixels outside the small
    margin-expanded crop, and never produces a preview meant for the
    live POV - it is only ever used for a single saved capture.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = instance.bbox
    ex1, ey1, ex2, ey2 = _bbox_with_margin(instance.bbox, (h, w), margin_ratio)

    crop = frame[ey1:ey2, ex1:ex2].copy()

    # instance.mask is aligned to the ORIGINAL (unexpanded) bbox; place
    # it inside a mask sized to the margin-expanded crop before applying.
    full_mask = np.zeros((ey2 - ey1, ex2 - ex1), dtype=np.uint8)
    off_y, off_x = y1 - ey1, x1 - ex1
    mh, mw = instance.mask.shape
    full_mask[off_y : off_y + mh, off_x : off_x + mw] = instance.mask

    result = np.full_like(crop, background)
    mask_bool = full_mask.astype(bool)
    result[mask_bool] = crop[mask_bool]
    return result


def crack_overlay_crop(
    frame: np.ndarray,
    instance: CrackInstance,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
    outline: int = 2,
) -> np.ndarray:
    """
    Produce a small red-mask visualization of one crack instance,
    cropped to its own region (never the full frame). This is an
    internal diagnostic helper: the result page's red highlight uses the
    full-image build_union_mask + draw_mask_overlay path, and the
    crack-only images use extract_crack_only; this per-instance crop is
    available for QA but is not part of the main workflow.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = instance.bbox
    ex1, ey1, ex2, ey2 = _bbox_with_margin(instance.bbox, (h, w), margin_ratio)

    crop = frame[ey1:ey2, ex1:ex2].copy()

    full_mask = np.zeros((ey2 - ey1, ex2 - ex1), dtype=np.uint8)
    off_y, off_x = y1 - ey1, x1 - ex1
    mh, mw = instance.mask.shape
    full_mask[off_y : off_y + mh, off_x : off_x + mw] = instance.mask

    return inference_core.draw_mask_overlay(crop, full_mask, color=color, alpha=alpha, outline=outline)
