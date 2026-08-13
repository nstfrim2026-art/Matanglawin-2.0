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

Detection quality is tuned entirely at INFERENCE TIME (best.pt is never
retrained). Two competing goals are balanced with plain, explainable knobs:

  Recall (find thin/hairline cracks that vanish when a 4000px photo is
  shrunk to a small inference size):
    * tiled     - "sliced" inference: cut the photo into overlapping tiles,
                  analyze each near-native, stitch the masks back.
    * enhance   - a detection-only contrast boost (CLAHE [+ optional
                  unsharp]) so faint cracks stand out. Applied only to the
                  model's input; the stored/displayed photo stays original.
    * imgsz     - whole-image inference resolution.
    * conf      - detection threshold (lower = more sensitive).

  Precision (do NOT paint wall texture, stains, or shadows red):
    * min_area        - drop tiny specks (mask pixels).
    * min_thinness    - keep only elongated, crack-like shapes and reject
                        compact blobs. Uses a rotation/curvature-invariant
                        thinness = perimeter^2 / (4*pi*area): a disk ~= 1,
                        a long thin crack >> 1.
    * min_length_frac - drop short isolated fragments: a component's long
                        side must be at least this fraction of the image's
                        larger dimension. Real cracks are long; false-
                        positive dashes are short.

A real crack is thin, long, and connected; typical false positives are
short dashes or roundish stains. The shape filter targets exactly that
difference, so precision can be raised without discarding the true crack.

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
for filtering); it is never returned to the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import inference_core

DEFAULT_WEIGHTS = "best.pt"
# Balanced defaults: tiling + light enhancement keep recall on thin cracks,
# while the shape filter (thinness + length) keeps precision high so plain
# wall texture / stains are not painted as cracks.
DEFAULT_CONF = 0.20
DEFAULT_IMGSZ = 1280
DEFAULT_MIN_AREA_PX = 60       # drop specks smaller than this many mask pixels
DEFAULT_TILED = True
DEFAULT_TILE = 1024            # tile size (px) and per-tile inference resolution
DEFAULT_TILE_OVERLAP = 0.2
DEFAULT_ENHANCE = True
DEFAULT_AUGMENT = False
# Detection-only enhancement strength (kept gentle so it doesn't turn wall
# texture into false cracks).
DEFAULT_CLAHE_CLIP = 1.5
DEFAULT_UNSHARP = False
# Shape-based precision filter (set either to 0 to disable that check).
DEFAULT_MIN_THINNESS = 3.0     # reject compact blobs (disk ~= 1.0)
DEFAULT_MIN_LENGTH_FRAC = 0.05  # reject short fragments (< 5% of the long side)


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
        clahe_clip: float = DEFAULT_CLAHE_CLIP,
        unsharp: bool = DEFAULT_UNSHARP,
        min_thinness: float = DEFAULT_MIN_THINNESS,
        min_length_frac: float = DEFAULT_MIN_LENGTH_FRAC,
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
        self.clahe_clip = clahe_clip
        self.unsharp = unsharp
        self.min_thinness = max(0.0, float(min_thinness))
        self.min_length_frac = max(0.0, float(min_length_frac))

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Run segmentation on a single decoded BGR image and return a
        DetectionResult describing every crack instance that passed the
        area + shape (thinness/length) filters.

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
            infer_img = inference_core.enhance_for_detection(
                frame, clahe_clip=self.clahe_clip, unsharp=self.unsharp
            )

        h, w = frame.shape[:2]
        if self.tiled:
            union = self._infer_tiled(infer_img, (h, w))
            candidates = self._components(union) if union is not None else []
        else:
            result = inference_core.predict_masks(
                infer_img, weights=self.weights, conf=self.conf,
                imgsz=self.imgsz, device=self.device, augment=self.augment,
            )
            candidates = self._result_masks(result, (h, w))

        # Shared area + shape filtering for BOTH paths.
        instances = [
            CrackInstance(area_px=int(m.sum()), mask=m)
            for m in candidates
            if self._accept_mask(m, (h, w))
        ]
        return DetectionResult(has_crack=len(instances) > 0, instances=instances)

    # -- candidate masks: whole-image path -----------------------------

    def _result_masks(self, result, shape_hw: Tuple[int, int]) -> List[np.ndarray]:
        h, w = shape_hw
        out: List[np.ndarray] = []
        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            for mask in masks:
                mask_bin = (mask > 0.5).astype(np.uint8)
                if mask_bin.shape != (h, w):
                    mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                out.append(mask_bin)
        return out

    # -- candidate masks: tiled ("sliced") path ------------------------

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

    def _components(self, binary: np.ndarray) -> List[np.ndarray]:
        """
        Split a stitched full-frame binary mask into connected components
        (one candidate mask per component). Overlapping tiles naturally
        merge a crack that spans tile borders into a single component.
        """
        num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return [(labels == lbl).astype(np.uint8) for lbl in range(1, num)]

    # -- shared area + shape acceptance --------------------------------

    def _accept_mask(self, mask: np.ndarray, shape_hw: Tuple[int, int]) -> bool:
        """
        Keep only crack-like blobs: big enough (min_area), elongated enough
        (min_thinness), and long enough (min_length_frac). This is what
        rejects wall texture / stains / short dashes while keeping the long,
        thin, real crack.
        """
        area = int(mask.sum())
        if area < self.min_area_px:
            return False
        if self.min_thinness <= 0 and self.min_length_frac <= 0:
            return True

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)

        if self.min_thinness > 0:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                return False
            thinness = (perimeter * perimeter) / (4.0 * math.pi * area)
            if thinness < self.min_thinness:
                return False  # compact blob (stain / texture patch), not a crack

        if self.min_length_frac > 0:
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            length = max(rw, rh)
            if length < self.min_length_frac * max(shape_hw):
                return False  # short isolated fragment, not the real crack

        return True


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
