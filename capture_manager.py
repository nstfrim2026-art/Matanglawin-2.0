"""
capture_manager.py - Automatic crack capture orchestration.

Wires together the pieces that turn a single frame's `DetectionResult`
(detector.py) into a saved inspection record:

    DetectionResult (from detector.py)
            |
            v
    validate (confidence floor, cooldown/debounce)
            |
            v
    crop to the crack region only (detector.crop_with_margin)
            |
            v
    save cropped image + full frame to disk
            |
            v
    attach current GPS fix (gps_provider.get_current_fix - or
    "Unavailable", never fabricated)
            |
            v
    write InspectionRecord to inspection_db.InspectionDB

Cooldown/debounce: a capture is only saved if enough time has passed
since the last saved capture (default 5s) OR the previous captured
crack region has moved significantly. This avoids saving many
near-duplicate frames while the drone hovers over the same crack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

import gps_provider
from detector import DetectionResult, crop_with_margin
from inspection_db import InspectionDB

DEFAULT_COOLDOWN_SEC = 5.0
DEFAULT_MIN_CONFIDENCE = 0.25


@dataclass
class CaptureResult:
    saved: bool
    inspection_id: Optional[int] = None
    reason: Optional[str] = None  # why nothing was saved, e.g. "cooldown", "low_confidence"


class CaptureManager:
    """
    Stateful (holds last-capture time) orchestrator. One instance per
    running video pipeline - not safe to share across independent
    streams without separate instances.
    """

    def __init__(
        self,
        db: InspectionDB,
        capture_dir: str,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.db = db
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_sec = cooldown_sec
        self.min_confidence = min_confidence

        self._last_capture_at: Optional[float] = None

    def maybe_capture(self, frame, detection: DetectionResult) -> CaptureResult:
        """
        Given the raw frame and its DetectionResult, decide whether to
        save a new inspection record. Returns a CaptureResult describing
        the outcome either way (never raises for "no detection" /
        "cooldown active" - those are expected, common outcomes).
        """
        if not detection.has_crack:
            return CaptureResult(saved=False, reason="no_detection")

        if detection.max_confidence < self.min_confidence:
            return CaptureResult(saved=False, reason="low_confidence")

        now = time.time()
        if self._last_capture_at is not None and (now - self._last_capture_at) < self.cooldown_sec:
            return CaptureResult(saved=False, reason="cooldown")

        # Pick the highest-confidence instance to crop/save.
        best = max(detection.instances, key=lambda inst: inst.confidence)
        crop = crop_with_margin(frame, best.bbox)

        ts_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        file_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        crop_filename = f"crack_{file_stamp}.jpg"
        crop_path = self.capture_dir / crop_filename

        try:
            ok = cv2.imwrite(str(crop_path), crop)
            if not ok:
                return CaptureResult(saved=False, reason="write_failed")
        except Exception:
            return CaptureResult(saved=False, reason="write_failed")

        fix = gps_provider.get_current_fix()

        detection_info = {
            "bbox": list(best.bbox),
            "area_px": best.area_px,
            "instances": [
                {"confidence": inst.confidence, "bbox": list(inst.bbox), "area_px": inst.area_px}
                for inst in detection.instances
            ],
        }

        inspection_id = self.db.add_inspection(
            timestamp=ts_str,
            cropped_image_path=str(crop_path),
            confidence=detection.max_confidence,
            num_instances=detection.num_instances,
            latitude=fix.latitude,
            longitude=fix.longitude,
            altitude=fix.altitude,
            detection_info=detection_info,
        )

        self._last_capture_at = now
        return CaptureResult(saved=True, inspection_id=inspection_id)

    def reset_cooldown(self) -> None:
        """Allow the very next valid detection to be captured immediately."""
        self._last_capture_at = None
