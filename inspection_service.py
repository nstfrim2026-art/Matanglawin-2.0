"""
inspection_service.py - THE single, central photo-inspection pipeline.

Both entry points into the app converge here so there is exactly one
inference/result-generation implementation:

    manual upload  (app.py /detect, /api/inspect) ─┐
                                                    ├─> InspectionService.analyze_*()
    DJI photo import (photo_import.py watch folder) ┘

For one still photo it:
    1. Loads the image (rejecting invalid/corrupt files).
    2. Runs YOLO11-seg ONCE (detector.CrackDetector) - never on a video
       stream.
    3. Determines status: "CRACK DETECTED" or "NO CRACK DETECTED".
    4. Writes result artifacts under
       <inspections_dir>/<uid>/:
           original/photo.jpg              - the untouched photo
           highlighted/crack_highlighted.jpg - full photo with every
                                             crack mask highlighted RED
                                             (result page only; never the
                                             live POV). For a no-crack
                                             photo this is just a copy of
                                             the original.
           crack/crack_001.png, ...        - one crack-ONLY image per
                                             detected instance, with the
                                             background suppressed via the
                                             segmentation mask (not a
                                             rectangular crop of surface).
           metadata.json                   - a copy of the record.
    5. Attaches a GPS fix if one is available (never fabricated).
    6. Writes an InspectionRecord to the database and returns it.

Confidence is used only internally (detection filtering + stored in
metadata/detection_info); it is never returned to the UI.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

import gps_provider
import inference_core
from detector import (
    CrackDetector,
    DEFAULT_CONF,
    DEFAULT_MIN_AREA_PX,
    build_union_mask,
    extract_crack_only,
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
    ):
        self.db = db
        self.inspections_dir = Path(inspections_dir)
        self.inspections_dir.mkdir(parents=True, exist_ok=True)
        self.weights = weights
        self.conf = conf
        self.min_area_px = min_area_px

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
                        weights=self.weights, conf=self.conf, min_area_px=self.min_area_px
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
        crack_dir = base / "crack"
        for d in (original_dir, highlighted_dir, crack_dir):
            d.mkdir(parents=True, exist_ok=True)

        original_path = original_dir / "photo.jpg"
        highlighted_path = highlighted_dir / "crack_highlighted.jpg"
        cv2.imwrite(str(original_path), img_bgr)

        crack_paths = []
        detection_info = {"source": source}

        if result.has_crack:
            status = STATUS_CRACK

            # Full photo with EVERY crack mask highlighted red (result
            # page artifact only - the live POV never gets this).
            union = build_union_mask(result.instances, img_bgr.shape[:2])
            highlighted = inference_core.draw_mask_overlay(img_bgr, union)
            cv2.imwrite(str(highlighted_path), highlighted)

            # One crack-only image per instance, highest confidence first
            # for stable numbering. PNG keeps the masked edges crisp.
            ordered = sorted(result.instances, key=lambda i: i.confidence, reverse=True)
            for idx, inst in enumerate(ordered, start=1):
                crop = extract_crack_only(img_bgr, inst)
                crack_path = crack_dir / f"crack_{idx:03d}.png"
                cv2.imwrite(str(crack_path), crop)
                crack_paths.append(str(crack_path))

            # Confidence/bbox kept internally only (never shown in UI).
            detection_info["instances"] = [
                {"confidence": round(float(i.confidence), 4), "bbox": list(i.bbox), "area_px": i.area_px}
                for i in ordered
            ]
        else:
            status = STATUS_NO_CRACK
            # Preserve the original; provide a "highlighted" that is just
            # the clean photo so the result page always has three slots.
            cv2.imwrite(str(highlighted_path), img_bgr)

        fix = gps_provider.get_current_fix()

        inspection_id = self.db.add_inspection(
            timestamp=ts_str,
            status=status,
            source=source,
            original_image_path=str(original_path),
            highlighted_image_path=str(highlighted_path),
            crack_image_paths=crack_paths,
            num_instances=result.num_instances,
            latitude=fix.latitude,
            longitude=fix.longitude,
            altitude=fix.altitude,
            detection_info=detection_info,
        )

        record = self.db.get_inspection(inspection_id)
        self._write_metadata(base, record)
        return record

    @staticmethod
    def _write_metadata(base: Path, record: InspectionRecord) -> None:
        meta = record.to_dict()
        meta["original_image_path"] = record.original_image_path
        meta["highlighted_image_path"] = record.highlighted_image_path
        meta["crack_image_paths"] = record.crack_image_paths
        try:
            (base / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            # Metadata is a convenience sidecar; failing to write it must
            # not fail the whole inspection (the DB record is canonical).
            pass
