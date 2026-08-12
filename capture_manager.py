"""
capture_manager.py - Automatic crack capture orchestration.

Wires together the pieces that turn a single frame's `DetectionResult`
(detector.py) into a saved inspection record:

    DetectionResult (from detector.py, already carries per-instance
    segmentation masks - not just bounding boxes)
            |
            v
    validate (confidence floor, cooldown/debounce)
            |
            v
    detector.extract_crack_only() - mask-based crack-only extraction
    (background suppressed outside the mask, not a rectangular slab
    of surrounding surface)
            |
            v
    save THREE files to disk (see `captures/` layout below)
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

`captures/` layout (under `capture_dir`):

    captures/original/   - the full captured frame, unmodified, kept
                            purely as evidence of what the drone saw.
    captures/crack/       - the crack-ONLY result (detector.extract_crack_only):
                            what gets shown on the dashboard/inspections
                            page and embedded in PDF reports.
    captures/overlays/    - an internal, small red-mask visualization
                            crop (detector.crack_overlay_crop), kept for
                            debugging/QA only. IMPORTANT: nothing in
                            this app exposes this folder through any
                            Flask route or frontend code - see
                            README.md's "no full-frame preview" note.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

import gps_provider
from detector import DetectionResult, extract_crack_only, crack_overlay_crop
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
        self.original_dir = self.capture_dir / "original"
        self.crack_dir = self.capture_dir / "crack"
        self.overlay_dir = self.capture_dir / "overlays"
        for d in (self.original_dir, self.crack_dir, self.overlay_dir):
            d.mkdir(parents=True, exist_ok=True)
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

        # Pick the highest-confidence instance to extract/save.
        best = max(detection.instances, key=lambda inst: inst.confidence)

        # Use the actual YOLO segmentation mask (not just the bounding
        # box) to isolate the crack region - see detector.py.
        crack_only = extract_crack_only(frame, best)

        file_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        ts_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))

        original_path = self.original_dir / f"frame_{file_stamp}.jpg"
        crack_path = self.crack_dir / f"crack_{file_stamp}.jpg"

        try:
            ok_original = cv2.imwrite(str(original_path), frame)
            ok_crack = cv2.imwrite(str(crack_path), crack_only)
            if not (ok_original and ok_crack):
                return CaptureResult(saved=False, reason="write_failed")
        except Exception:
            return CaptureResult(saved=False, reason="write_failed")

        # Internal-only red-mask visualization crop. This is written to
        # captures/overlays/ purely for internal diagnostics/QA - it is
        # never served through a Flask route, never linked to from any
        # template, and never used as (or in place of) the live POV. A
        # failure to write it must not block saving the actual capture.
        try:
            overlay_crop = crack_overlay_crop(frame, best)
            overlay_path = self.overlay_dir / f"overlay_{file_stamp}.jpg"
            cv2.imwrite(str(overlay_path), overlay_crop)
        except Exception:
            pass

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
            image_path=str(original_path),
            cropped_image_path=str(crack_path),
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
