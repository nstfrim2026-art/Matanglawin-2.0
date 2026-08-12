"""
detector.py - Single-image crack detection (YOLO11-seg).

Wraps the trained YOLO11-seg model (best.pt) and the shared mask helpers
(inference_core.py) to analyze ONE still image at a time. Used by
inspection_service.py for BOTH manually uploaded photos and photos
imported automatically from the DJI controller - the two converge on
this one implementation.

    image (BGR)  ->  CrackDetector.process_frame()  ->  DetectionResult
                                                              |
                                                              v
                                                   inspection_service.py
                                          (original + red-highlighted image)

There is deliberately NO continuous/video code path here (no OpenCV
video-capture loop, no RTSP frame reader): the model is invoked once per
photo, never on a stream of frames. The live drone POV is served straight
from MediaMTX's WebRTC endpoint to the browser and never touches this
module.

What this module deliberately does NOT produce:
    * bounding boxes,
    * crack-only / cropped-crack / separate-mask images,
    * any confidence/score/metric surfaced to the UI.

Confidence is used ONLY internally (as YOLO's own detection threshold and
for dropping tiny noise specks) and is never returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import inference_core

DEFAULT_WEIGHTS = "best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_MIN_AREA_PX = 150  # ignore specks smaller than this many mask pixels


@dataclass
class CrackInstance:
    """One accepted crack segmentation instance (full-frame binary mask)."""
    area_px: int
    mask: np.ndarray  # binary (0/1) mask, shape == full frame (H, W)


@dataclass
class DetectionResult:
    has_crack: bool
    instances: List[CrackInstance] = field(default_factory=list)

    @property
    def num_instances(self) -> int:
        return len(self.instances)

    def union_mask(self, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """Combined binary mask (H, W) of every accepted crack, or None."""
        if not self.instances:
            return None
        return build_union_mask(self.instances, shape_hw)


class CrackDetector:
    """
    Thin wrapper around the shared YOLO11-seg model. The underlying model
    is cached by ``inference_core.get_model()``.
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
        Run segmentation on a single decoded BGR image and return a
        DetectionResult describing every crack instance that passed the
        minimum-area noise filter.

        No overlay/preview image is produced here - only the binary
        masks, which inspection_service turns into the red-highlighted
        result photo. The live video feed is never touched.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(has_crack=False, instances=[])

        result = inference_core.predict_masks(
            frame, weights=self.weights, conf=self.conf, imgsz=self.imgsz, device=self.device,
        )

        h, w = frame.shape[:2]
        instances: List[CrackInstance] = []

        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            for mask in masks:
                mask_bin = (mask > 0.5).astype(np.uint8)
                if mask_bin.shape != (h, w):
                    mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                area = int(mask_bin.sum())
                if area < self.min_area_px:
                    continue  # too small - likely noise, not a real crack
                instances.append(CrackInstance(area_px=area, mask=mask_bin))

        return DetectionResult(has_crack=len(instances) > 0, instances=instances)


def build_union_mask(instances: List[CrackInstance], shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Combine every instance's full-frame binary mask into one binary
    union mask (H, W), used to draw the single red-highlighted result
    image (the full photo with ALL detected cracks highlighted). This is
    a result-page artifact only - never applied to the live POV.
    """
    h, w = shape_hw
    union = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        m = inst.mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, m)
    return union
