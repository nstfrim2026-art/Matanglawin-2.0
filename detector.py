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

Recall on thin / hairline cracks (the hard case for high-resolution drone
photos) is improved WITHOUT retraining, purely at inference time:

    * enhance   - a detection-only contrast boost (CLAHE + unsharp) so
                  faint cracks stand out. The stored/displayed photo stays
                  the untouched original; only the model's input is boosted.
    * tiled     - "sliced" inference: the photo is cut into overlapping
                  tiles, each analyzed at near-native resolution, and the
                  masks are stitched back together. A hairline crack that
                  would vanish when the whole 4000px photo is shrunk to
                  imgsz stays several pixels wide inside a tile.
    * imgsz     - inference resolution for the whole-image path.
    * conf      - detection threshold (lower recovers faint segments).
    * min_area  - noise floor (mask pixels) below which a blob is dropped.
    * augment   - optional test-time augmentation (multi-scale + flips).

All of these are inference-time knobs; the weights (best.pt) are unchanged.

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
# Recall-tuned defaults for high-resolution DJI stills with thin cracks.
# (Historically conf=0.25/imgsz=640/min_area=150, which under-detected
# hairline cracks after the ~6x downscale of a full-res photo.)
DEFAULT_CONF = 0.15
DEFAULT_IMGSZ = 1280
DEFAULT_MIN_AREA_PX = 40  # ignore specks smaller than this many mask pixels
DEFAULT_TILED = True
DEFAULT_TILE = 1024       # tile size (px) and per-tile inference resolution
DEFAULT_TILE_OVERLAP = 0.2
DEFAULT_ENHANCE = True
DEFAULT_AUGMENT = False


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
        tiled: bool = DEFAULT_TILED,
        tile: int = DEFAULT_TILE,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        enhance: bool = DEFAULT_ENHANCE,
        augment: bool = DEFAULT_AUGMENT,
    ):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.min_area_px = min_area_px
        self.device = device
        self.tiled = tiled
        self.tile = max(64, int(tile))
        self.tile_overlap = min(0.9, max(0.0, float(tile_overlap)))
        self.enhance = enhance
        self.augment = augment

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Run segmentation on a single decoded BGR image and return a
        DetectionResult describing every crack instance that passed the
        minimum-area noise filter.

        No overlay/preview image is produced here - only the binary masks,
        which inspection_service turns into the red-highlighted result
        photo (drawn on the ORIGINAL, not the enhanced copy). The live
        video feed is never touched.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(has_crack=False, instances=[])

        # Detection-only enhancement: feed a boosted COPY to the model.
        infer_img = frame
        if self.enhance:
            infer_img = inference_core.enhance_for_detection(frame)

        h, w = frame.shape[:2]
        if self.tiled:
            union = self._infer_tiled(infer_img, (h, w))
            instances = self._instances_from_binary(union) if union is not None else []
        else:
            result = inference_core.predict_masks(
                infer_img, weights=self.weights, conf=self.conf,
                imgsz=self.imgsz, device=self.device, augment=self.augment,
            )
            instances = self._instances_from_result(result, (h, w))

        return DetectionResult(has_crack=len(instances) > 0, instances=instances)

    # -- whole-image path ----------------------------------------------

    def _instances_from_result(self, result, shape_hw: Tuple[int, int]) -> List["CrackInstance"]:
        h, w = shape_hw
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
        return instances

    # -- tiled ("sliced") path -----------------------------------------

    @staticmethod
    def _tile_starts(size: int, tile: int, step: int) -> List[int]:
        """Start offsets that tile [0, size) with the last tile flush to the edge."""
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, step))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    def _infer_tiled(self, infer_img: np.ndarray, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Run the model on overlapping tiles and OR the per-tile masks back
        into one full-frame binary mask. Returns None if nothing was
        detected anywhere.
        """
        h, w = shape_hw
        th, tw = min(self.tile, h), min(self.tile, w)
        step_y = max(1, int(th * (1.0 - self.tile_overlap)))
        step_x = max(1, int(tw * (1.0 - self.tile_overlap)))

        accum = np.zeros((h, w), dtype=np.uint8)
        found = False
        for y in self._tile_starts(h, th, step_y):
            for x in self._tile_starts(w, tw, step_x):
                y2, x2 = min(y + th, h), min(x + tw, w)
                crop = infer_img[y:y2, x:x2]
                if crop.size == 0:
                    continue
                result = inference_core.predict_masks(
                    crop, weights=self.weights, conf=self.conf,
                    imgsz=self.tile, device=self.device, augment=self.augment,
                )
                if result.masks is None or len(result.masks) == 0:
                    continue
                masks = result.masks.data.cpu().numpy()  # (N, ch, cw)
                tile_mask = (masks > 0.5).any(axis=0).astype(np.uint8)
                ch, cw = crop.shape[:2]
                if tile_mask.shape != (ch, cw):
                    tile_mask = cv2.resize(tile_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
                accum[y:y2, x:x2] = np.maximum(accum[y:y2, x:x2], tile_mask)
                found = True
        return accum if found else None

    def _instances_from_binary(self, binary: np.ndarray) -> List["CrackInstance"]:
        """
        Split a stitched full-frame binary mask into connected components,
        one CrackInstance per component (dropping sub-threshold specks).
        Overlapping tiles naturally merge a crack that spans tile borders
        into a single component.
        """
        num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        instances: List[CrackInstance] = []
        for lbl in range(1, num):
            comp = (labels == lbl).astype(np.uint8)
            area = int(comp.sum())
            if area < self.min_area_px:
                continue
            instances.append(CrackInstance(area_px=area, mask=comp))
        return instances


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
