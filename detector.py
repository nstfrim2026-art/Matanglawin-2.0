"""
detector.py - Frame-by-frame crack detection for the live drone pipeline.

This wraps the same YOLO11-seg model (best.pt) and mask-drawing helpers
used by the single-image upload flow (inference_core.py), but operates
on individual video frames pulled from the RTSP stream (video_source.py)
instead of an uploaded file.

Pipeline position:

    RTSP frame -> Detector.process_frame() -> DetectionResult
                                                  |
                                                  v
                                          capture_manager.py
                                       (validation + auto-capture)

Responsibilities kept here:
    - Run YOLO11-seg inference on a raw BGR frame (reusing
      inference_core.predict_masks / union_mask_from_result so detection
      logic never diverges from the upload flow).
    - Keep each crack instance's own segmentation mask (not just its
      bounding box), so downstream code can extract "only the crack
      pixels", not a rectangular slab of surrounding surface.
    - Basic detection-quality validation (confidence floor, minimum
      crack pixel area) so obviously-noisy predictions don't trigger a
      capture.
    - Provide `extract_crack_only()` / `crack_overlay_crop()` helpers
      that turn a single instance's mask into (a) a crack-only image
      with the background suppressed, and (b) a small red-mask
      visualization crop - both operating on a small crop around the
      crack, never on the full frame.

Explicitly NOT this module's job:
    - Deciding whether to *save* a capture, cooldown/debounce, writing
      to the database - that's capture_manager.py.
    - Talking to RTSP/MediaMTX - that's video_source.py.
    - Producing any full-frame annotated preview. There is intentionally
      no code path here that overlays a mask on the *entire* live frame
      for display purposes - see capture_manager.py and README.md for
      why (the live POV must always stay a clean, unannotated feed).
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
        Run detection on a single BGR frame (as returned by OpenCV's
        VideoCapture). Returns a DetectionResult describing every crack
        instance that passed basic quality validation.

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
    cropped to its own region (never the full frame). Internal use only
    (see capture_manager.py's `captures/overlays/` folder) - this is NOT
    exposed through any Flask route or used as the live POV. It exists
    so the red-mask/contour visualization step described in the
    architecture is still actually performed and available for
  internal diagnostics, without ever becoming a full-frame annotated
    preview.
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
