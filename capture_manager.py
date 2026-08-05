"""
capture_manager.py - Automatic image capture workflow manager.

Implements a two-stage inspection pipeline:
  Stage 1: Continuous live monitoring for possible cracks.
  Stage 2: Automatic high-confidence image capture followed by detailed
           crack analysis on the captured image.

State machine:
  monitoring -> possible_crack -> threshold_reached -> capturing ->
  analyzing -> analysis_complete -> cooldown -> monitoring (loop)

Thread-safe: all state mutations are protected by a threading lock.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from inference_core import annotate_frame


# Valid states in the capture workflow
STATES = (
    "monitoring",
    "possible_crack",
    "threshold_reached",
    "capturing",
    "analyzing",
    "analysis_complete",
    "cooldown",
)


class CaptureManager:
    """Manages automatic crack image capture based on confidence threshold."""

    def __init__(
        self,
        weights: str = "best.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        threshold: float = 0.85,
        cooldown: float = 5.0,
    ):
        self._weights = weights
        self._conf = conf
        self._imgsz = imgsz

        self._threshold = threshold
        self._cooldown = cooldown

        self._lock = threading.Lock()
        self._state = "monitoring"

        # Timing
        self._state_change_time: float = time.monotonic()
        self._cooldown_start: float = 0.0

        # Latest capture results (kept in memory for serving to frontend)
        self._latest_capture_jpeg: Optional[bytes] = None
        self._latest_metadata: Optional[dict] = None
        self._capture_count: int = 0

        # Captures directory
        self._captures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "captures"
        )

        # Initialize capture count from existing files
        self._capture_count = self._scan_existing_captures()

    # -- Configuration getters/setters --------------------------------

    def get_threshold(self) -> float:
        with self._lock:
            return self._threshold

    def set_threshold(self, value: float) -> bool:
        """Set the confidence threshold. Must be between 0.50 and 1.00."""
        if not (0.50 <= value <= 1.00):
            return False
        with self._lock:
            self._threshold = value
        return True

    def get_cooldown(self) -> float:
        with self._lock:
            return self._cooldown

    def set_cooldown(self, value: float) -> bool:
        """Set the cooldown period in seconds. Must be positive."""
        if value < 0:
            return False
        with self._lock:
            self._cooldown = value
        return True

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def reset(self) -> None:
        """Reset the state machine back to monitoring."""
        with self._lock:
            self._state = "monitoring"
            self._state_change_time = time.monotonic()

    # -- Public query methods ------------------------------------------

    def get_capture_status(self) -> dict:
        """Return the full capture workflow status as a dict."""
        with self._lock:
            result = {
                "state": self._state,
                "threshold": self._threshold,
                "cooldown": self._cooldown,
                "capture_count": self._capture_count,
            }
            if self._latest_metadata is not None:
                result["latest_analysis"] = self._latest_metadata
            return result

    def get_latest_capture_jpeg(self) -> Optional[bytes]:
        """Return the latest captured+annotated image as JPEG bytes."""
        with self._lock:
            return self._latest_capture_jpeg

    # -- Core logic called by detector on each frame -------------------

    def process_frame(
        self, raw_frame: np.ndarray, crack_metadata_list: list
    ) -> None:
        """
        Called by detector.py after each live inference pass.

        Checks if any detection meets the threshold and triggers the
        capture workflow. This method handles all state transitions.
        """
        with self._lock:
            state = self._state

        if state == "monitoring":
            self._handle_monitoring(raw_frame, crack_metadata_list)
        elif state == "possible_crack":
            self._handle_possible_crack(raw_frame, crack_metadata_list)
        elif state == "cooldown":
            self._handle_cooldown()
        # Other states (threshold_reached, capturing, analyzing,
        # analysis_complete) are transient and handled in the capture flow

    def _handle_monitoring(
        self, raw_frame: np.ndarray, crack_metadata_list: list
    ) -> None:
        """Check if any detection is close to threshold."""
        if not crack_metadata_list:
            return

        max_conf = max(m["confidence"] for m in crack_metadata_list)

        with self._lock:
            threshold = self._threshold

        # If any detection is within 90% of threshold, transition to possible_crack
        if max_conf >= threshold * 0.90:
            if max_conf >= threshold:
                # Directly meets threshold - start capture
                self._trigger_capture(raw_frame, crack_metadata_list)
            else:
                with self._lock:
                    self._state = "possible_crack"
                    self._state_change_time = time.monotonic()

    def _handle_possible_crack(
        self, raw_frame: np.ndarray, crack_metadata_list: list
    ) -> None:
        """In possible_crack state, check if threshold is now met."""
        if not crack_metadata_list:
            # No detection - go back to monitoring
            with self._lock:
                self._state = "monitoring"
                self._state_change_time = time.monotonic()
            return

        max_conf = max(m["confidence"] for m in crack_metadata_list)

        with self._lock:
            threshold = self._threshold

        if max_conf >= threshold:
            self._trigger_capture(raw_frame, crack_metadata_list)
        else:
            # Check if we've been in possible_crack too long (2 seconds)
            with self._lock:
                elapsed = time.monotonic() - self._state_change_time
            if elapsed > 2.0:
                with self._lock:
                    self._state = "monitoring"
                    self._state_change_time = time.monotonic()

    def _handle_cooldown(self) -> None:
        """Check if cooldown period has elapsed."""
        with self._lock:
            elapsed = time.monotonic() - self._cooldown_start
            if elapsed >= self._cooldown:
                self._state = "monitoring"
                self._state_change_time = time.monotonic()

    def _trigger_capture(
        self, raw_frame: np.ndarray, crack_metadata_list: list
    ) -> None:
        """Execute the full capture workflow (state transitions happen inline)."""
        with self._lock:
            self._state = "threshold_reached"
            self._state_change_time = time.monotonic()

        with self._lock:
            self._state = "capturing"
            self._state_change_time = time.monotonic()

        # Save the raw frame as high-quality JPEG
        timestamp = datetime.now(timezone.utc)
        ok, buf = cv2.imencode(
            ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not ok:
            # Failed to encode - go back to monitoring
            with self._lock:
                self._state = "monitoring"
                self._state_change_time = time.monotonic()
            return

        raw_jpeg = buf.tobytes()

        with self._lock:
            self._state = "analyzing"
            self._state_change_time = time.monotonic()

        # Run full analysis on the captured frame
        try:
            annotated, crack_present, analysis_metadata = annotate_frame(
                raw_frame,
                weights=self._weights,
                conf=self._conf,
                imgsz=self._imgsz,
            )
        except Exception:
            annotated = raw_frame
            crack_present = False
            analysis_metadata = crack_metadata_list

        # Encode annotated frame as JPEG for display
        ok_ann, buf_ann = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        annotated_jpeg = buf_ann.tobytes() if ok_ann else raw_jpeg

        # Build metadata
        # Use the analysis results (re-run on captured frame) if available,
        # otherwise fall back to the live detection metadata
        final_metadata_list = analysis_metadata if analysis_metadata else crack_metadata_list

        # Pick the highest-confidence detection as the primary result
        if final_metadata_list:
            primary = max(final_metadata_list, key=lambda m: m["confidence"])
        else:
            primary = {
                "classification": "Unknown",
                "confidence": 0.0,
                "area_px": 0,
                "bbox": [],
                "estimated_length_px": 0.0,
                "estimated_width_px": 0.0,
            }

        with self._lock:
            self._capture_count += 1
            capture_num = self._capture_count

        # Format filename
        image_filename = f"crack_{capture_num:04d}.jpg"
        json_filename = f"crack_{capture_num:04d}.json"

        metadata = {
            "timestamp": timestamp.isoformat(),
            "image_filename": image_filename,
            "capture_number": capture_num,
            "classification": primary["classification"],
            "confidence": primary["confidence"],
            "estimated_length_px": primary["estimated_length_px"],
            "estimated_width_px": primary["estimated_width_px"],
            "area_px": primary["area_px"],
            "bbox": primary["bbox"],
            "crack_metadata_list": final_metadata_list,
        }

        # Save to disk
        self._save_capture(image_filename, json_filename, raw_jpeg, metadata)

        # Store in memory for frontend access
        with self._lock:
            self._latest_capture_jpeg = annotated_jpeg
            self._latest_metadata = metadata
            self._state = "analysis_complete"
            self._state_change_time = time.monotonic()

        # Brief pause in analysis_complete state, then enter cooldown
        time.sleep(0.1)

        with self._lock:
            self._state = "cooldown"
            self._cooldown_start = time.monotonic()
            self._state_change_time = time.monotonic()

    def _save_capture(
        self,
        image_filename: str,
        json_filename: str,
        jpeg_bytes: bytes,
        metadata: dict,
    ) -> None:
        """Save capture image and metadata JSON to the captures/ directory."""
        os.makedirs(self._captures_dir, exist_ok=True)

        image_path = os.path.join(self._captures_dir, image_filename)
        json_path = os.path.join(self._captures_dir, json_filename)

        with open(image_path, "wb") as f:
            f.write(jpeg_bytes)

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _scan_existing_captures(self) -> int:
        """Scan captures/ directory to find the highest existing capture number."""
        if not os.path.isdir(self._captures_dir):
            return 0

        max_num = 0
        for fname in os.listdir(self._captures_dir):
            if fname.startswith("crack_") and fname.endswith(".jpg"):
                try:
                    num_str = fname[6:10]  # crack_NNNN.jpg
                    num = int(num_str)
                    if num > max_num:
                        max_num = num
                except (ValueError, IndexError):
                    continue
        return max_num
