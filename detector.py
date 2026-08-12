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
    - Compute a tight, valid bounding box crop around the crack region
      (used by capture_manager to save "only the crack region", not the
      full frame).
    - Basic detection-quality validation (confidence floor, minimum
      crack pixel area) so obviously-noisy predictions don't trigger a
      capture.
    - Produce an overlay frame (for optional live-preview annotation)
      using the exact same drawing routine as the image-upload result.

Explicitly NOT this module's job:
    - Deciding whether to *save* a capture, cooldown/debounce, writing
      to the database - that's capture_manager.py.
    - Talking to RTSP/MediaMTX - that's video_source.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import inference_core

DEFAULT_WEIGHTS = "best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_MIN_AREA_PX = 150  # ignore specks smaller than this many mask pixels


@dataclass
class CrackInstance:
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixel coords
    area_px: int


@dataclass
class DetectionResult:
    has_crack: bool
    instances: List[CrackInstance]
    union_mask: Optional[np.ndarray]
    overlay_frame: Optional[np.ndarray]
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

    def process_frame(self, frame: np.ndarray, draw_overlay: bool = False) -> DetectionResult:
        """
        Run detection on a single BGR frame (as returned by OpenCV's
        VideoCapture). Returns a DetectionResult describing every crack
        instance that passed basic quality validation.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                has_crack=False, instances=[], union_mask=None,
                overlay_frame=None, max_confidence=0.0,
            )

        result = inference_core.predict_masks(
            frame, weights=self.weights, conf=self.conf, imgsz=self.imgsz, device=self.device,
        )

        h, w = frame.shape[:2]
        union = inference_core.union_mask_from_result(result, (h, w))

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
                area = int(mask_bin.sum())
                if area < self.min_area_px:
                    continue  # too small - likely noise, not a real crack
                bbox = _bbox_from_mask(mask_bin, frame_shape=(h, w))
                if bbox is None:
                    continue
                instances.append(
                    CrackInstance(confidence=float(conf_val), bbox=bbox, area_px=area)
                )
                max_conf = max(max_conf, float(conf_val))

        overlay_frame = None
        if draw_overlay and union is not None:
            overlay_frame = inference_core.draw_mask_overlay(frame, union)

        return DetectionResult(
            has_crack=len(instances) > 0,
            instances=instances,
            union_mask=union,
            overlay_frame=overlay_frame,
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


def crop_with_margin(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.15,
) -> np.ndarray:
    """
    Crop `frame` to `bbox` (x1,y1,x2,y2) plus a small margin so the saved
    capture shows only the relevant crack region (not the entire drone
    frame), while still giving a bit of surrounding context.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin_ratio), int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2].copy()
