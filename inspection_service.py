"""
inspection_service.py - THE single, central photo-inspection pipeline.

Both entry points into the app converge here so there is exactly one
inference/result-generation implementation (see requirement: "the manual
upload and automatic DJI photo path must converge into the same analysis
service"):

    manual upload  (app.py /detect, /api/inspect) ────┐
                                                       ├─> InspectionService.analyze_*()
    DJI photo import (photo_import.py watch folder) ───┘

For one still photo it:
    1. Loads the image (rejecting invalid/corrupt files).
    2. Runs YOLO11-seg ONCE (detector.CrackDetector) - never on a video
       stream.
    3. Determines status: "CRACK DETECTED" or "NO CRACK DETECTED".
    4. Writes exactly TWO result artifacts under <inspections_dir>/<uid>/:
           original/photo.jpg                 - the untouched photo
           highlighted/crack_highlighted.jpg  - the full photo with every
                                                crack segmentation mask
                                                painted RED (result/history
                                                only; never the live POV).
                                                For a no-crack photo this is
                                                simply a copy of the original.
    5. Writes an InspectionRecord to the database and returns it.

There is NO crack-only image, NO cropped crack, NO separate mask image,
NO bounding box, and NO confidence anywhere in the output - only the
status and the two images above.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

import inference_core
from detector import (
    CrackDetector,
    DEFAULT_AUGMENT,
    DEFAULT_CONF,
    DEFAULT_ENHANCE,
    DEFAULT_IMGSZ,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_TILE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILED,
)
from inspection_db import InspectionDB, InspectionRecord, STATUS_CRACK, STATUS_NO_CRACK


class InvalidImageError(Exception):
    """Raised when a file can't be decoded as an image."""


class InspectionService:
    def __init__(
        self,
        db: InspectionDB,
        inspections_dir: str,
        weights: str,
        conf: float = DEFAULT_CONF,
        min_area_px: int = DEFAULT_MIN_AREA_PX,
        imgsz: int = DEFAULT_IMGSZ,
        tiled: bool = DEFAULT_TILED,
        tile: int = DEFAULT_TILE,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        enhance: bool = DEFAULT_ENHANCE,
        augment: bool = DEFAULT_AUGMENT,
    ):
        self.db = db
        self.inspections_dir = Path(inspections_dir)
        self.inspections_dir.mkdir(parents=True, exist_ok=True)
        self.weights = weights
        # Inference-time recall knobs (no retraining). Forwarded verbatim
        # to the single CrackDetector; see detector.py for what each does.
        self.conf = conf
        self.min_area_px = min_area_px
        self.imgsz = imgsz
        self.tiled = tiled
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.enhance = enhance
        self.augment = augment

        self._detector: Optional[CrackDetector] = None
        self._detector_lock = threading.Lock()

    # -- model (lazy, shared, thread-safe) -----------------------------

    def _get_detector(self) -> CrackDetector:
        # Lazy so importing this module / constructing the service never
        # loads the (heavy) YOLO weights until the first actual analysis.
        if self._detector is None:
            with self._detector_lock:
                if self._detector is None:
                    self._detector = CrackDetector(
                        weights=self.weights,
                        conf=self.conf,
                        min_area_px=self.min_area_px,
                        imgsz=self.imgsz,
                        tiled=self.tiled,
                        tile=self.tile,
                        tile_overlap=self.tile_overlap,
                        enhance=self.enhance,
                        augment=self.augment,
                    )
        return self._detector

    # -- public API ----------------------------------------------------

    def analyze_file(self, image_path: str, source: str = "upload") -> InspectionRecord:
        """Analyze an image file on disk. Raises InvalidImageError if unreadable."""
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            raise InvalidImageError(f"Could not read image: {image_path}")
        return self._analyze(img, source=source)

    def analyze_array(self, img_bgr, source: str = "upload") -> InspectionRecord:
        """Analyze an already-decoded BGR image array."""
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            raise InvalidImageError("Empty image array")
        return self._analyze(img_bgr, source=source)

    # -- internals -----------------------------------------------------

    def _analyze(self, img_bgr, source: str) -> InspectionRecord:
        detector = self._get_detector()
        result = detector.process_frame(img_bgr)

        now = time.time()
        ts_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        uid = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + "_" + uuid.uuid4().hex[:8]

        base = self.inspections_dir / uid
        original_dir = base / "original"
        highlighted_dir = base / "highlighted"
        for d in (original_dir, highlighted_dir):
            d.mkdir(parents=True, exist_ok=True)

        original_path = original_dir / "photo.jpg"
        highlighted_path = highlighted_dir / "crack_highlighted.jpg"

        # Always preserve the untouched original photo first.
        cv2.imwrite(str(original_path), img_bgr)

        if result.has_crack:
            status = STATUS_CRACK
            # Full photo with EVERY crack mask highlighted red, following
            # the exact segmentation masks (no boxes). Result/history
            # artifact only - the live POV never gets this.
            union = result.union_mask(img_bgr.shape[:2])
            highlighted = inference_core.draw_mask_overlay(img_bgr, union)
            cv2.imwrite(str(highlighted_path), highlighted)
        else:
            status = STATUS_NO_CRACK
            # No crack -> the "highlighted" image is just a clean copy of
            # the original (there is no red-highlighted image to show).
            cv2.imwrite(str(highlighted_path), img_bgr)

        inspection_id = self.db.add_inspection(
            timestamp=ts_str,
            status=status,
            source=source,
            original_image_path=str(original_path),
            highlighted_image_path=str(highlighted_path),
            num_instances=result.num_instances,
        )

        return self.db.get_inspection(inspection_id)
