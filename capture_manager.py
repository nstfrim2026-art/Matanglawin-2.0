"""
capture_manager.py - Automatic image capture workflow manager.

Implements a research-grade inspection pipeline:
  Stage 1: Continuous live monitoring for possible cracks.
  Stage 2: Temporal verification (5 consecutive above-threshold frames).
  Stage 3: Image quality validation (blur, brightness, exposure).
  Stage 4: Full crack analysis on the captured frame.
  Stage 5: Advanced crack validation to reject false positives.
  Stage 6: Save confirmed inspection (images, overlays, metadata, reports).

State machine:
  monitoring -> threshold_reached -> capturing ->
  analyzing -> analysis_complete -> cooldown -> monitoring (loop)

Thread-safe: all state mutations are protected by a threading lock.
Capture work runs in a background thread to avoid blocking the detector loop.

Output directory structure:
  captures/
    images/        - Original high-quality captured frames
    overlays/      - Annotated frames with segmentation overlays
    metadata/      - JSON metadata per inspection
    reports/       - PDF inspection reports
    exports/       - Shared CSV file for all inspections
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from crack_validator import validate_cracks
from image_quality import validate_image_quality
from inference_core import annotate_frame
from inspection_db import init_db, insert_inspection
from report_generator import generate_csv_record, generate_pdf_report
from gps_provider import GPSProvider

logger = logging.getLogger(__name__)


# Valid states in the capture workflow
STATES = (
    "monitoring",
    "threshold_reached",
    "capturing",
    "analyzing",
    "analysis_complete",
    "cooldown",
)

# Maximum number of captures to keep on disk (oldest are rotated out)
MAX_CAPTURES = 500

# Minimum allowed cooldown period in seconds
MIN_COOLDOWN = 1.0

# How long to hold analysis_complete state before transitioning to cooldown
ANALYSIS_COMPLETE_HOLD_SECONDS = 2.0

# Temporal verification: consecutive frames required above threshold
VERIFICATION_FRAME_COUNT = 5

# Minimum time span (seconds) that verification frames must cover
VERIFICATION_MIN_SPAN_SECONDS = 1.5

# Maximum time (seconds) for the capture worker before force-reset
CAPTURE_WORKER_TIMEOUT_SECONDS = 30.0

# Padding ratio for crack region cropping (15% of bbox dimensions on each side)
CROP_PADDING_RATIO = 0.15


class CaptureManager:
    """Manages automatic crack image capture based on confidence threshold."""

    def __init__(
        self,
        weights: str = "best.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        threshold: float = 0.85,
        cooldown: float = 5.0,
        max_captures: int = MAX_CAPTURES,
    ):
        self._weights = weights
        self._conf = conf
        self._imgsz = imgsz

        self._threshold = threshold
        self._cooldown = max(cooldown, MIN_COOLDOWN)
        self._max_captures = max_captures

        self._lock = threading.Lock()
        self._state = "monitoring"

        # Timing
        self._state_change_time: float = time.monotonic()
        self._cooldown_start: float = 0.0

        # Temporal verification state
        self._consecutive_detections: int = 0
        self._verification_buffer: list = []  # list of (frame, metadata, timestamp) tuples

        # Latest capture results (kept in memory for serving to frontend)
        self._latest_capture_jpeg: Optional[bytes] = None
        self._latest_metadata: Optional[dict] = None
        self._capture_count: int = 0

        # Flag to prevent re-entrant capture triggers while one is in progress
        self._capture_in_progress = False

        # Camera source name for metadata (set externally by Detector)
        self._camera_source: str = "unknown"

        # Captures directory (base)
        self._captures_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "captures"
        )

        # Sub-directories
        self._images_dir = os.path.join(self._captures_dir, "images")
        self._overlays_dir = os.path.join(self._captures_dir, "overlays")
        self._metadata_dir = os.path.join(self._captures_dir, "metadata")
        self._reports_dir = os.path.join(self._captures_dir, "reports")
        self._exports_dir = os.path.join(self._captures_dir, "exports")
        self._crops_dir = os.path.join(self._captures_dir, "crops")

        # GPS provider for geolocation
        self._gps_provider = GPSProvider()

        # Initialize database
        init_db()

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
        """Set the cooldown period in seconds. Must be >= 1.0."""
        if value < MIN_COOLDOWN:
            return False
        with self._lock:
            self._cooldown = value
        return True

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def set_camera_source(self, source_name: str) -> None:
        """Set the current camera source name for metadata recording."""
        with self._lock:
            self._camera_source = source_name

    def reset(self) -> None:
        """Reset the state machine back to monitoring."""
        with self._lock:
            self._state = "monitoring"
            self._state_change_time = time.monotonic()
            self._capture_in_progress = False
            self._consecutive_detections = 0
            self._verification_buffer = []

    # -- Public query methods ------------------------------------------

    def get_capture_status(self) -> dict:
        """Return the full capture workflow status as a dict."""
        with self._lock:
            result = {
                "state": self._state,
                "threshold": self._threshold,
                "cooldown": self._cooldown,
                "capture_count": self._capture_count,
                "consecutive_detections": self._consecutive_detections,
                "verification_required": VERIFICATION_FRAME_COUNT,
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

        Implements temporal verification before triggering capture.
        This method handles state transitions.
        The actual capture work runs in a background thread.
        """
        with self._lock:
            state = self._state

        if state == "monitoring":
            self._handle_monitoring(raw_frame, crack_metadata_list)
        elif state == "cooldown":
            self._handle_cooldown()
        # Other states (threshold_reached, capturing, analyzing,
        # analysis_complete) are transient and handled by the capture thread

    def _handle_monitoring(
        self, raw_frame: np.ndarray, crack_metadata_list: list
    ) -> None:
        """
        Temporal verification: require VERIFICATION_FRAME_COUNT consecutive
        frames with max confidence >= threshold AND spanning at least
        VERIFICATION_MIN_SPAN_SECONDS before triggering capture.
        """
        with self._lock:
            threshold = self._threshold
            if self._capture_in_progress:
                return

        if not crack_metadata_list:
            # No detections - reset consecutive counter
            with self._lock:
                self._consecutive_detections = 0
                self._verification_buffer = []
            return

        max_conf = max(m["confidence"] for m in crack_metadata_list)

        if max_conf >= threshold:
            now = time.monotonic()
            with self._lock:
                self._consecutive_detections += 1
                self._verification_buffer.append(
                    (raw_frame.copy(), list(crack_metadata_list), now)
                )
                # Keep buffer size bounded
                if len(self._verification_buffer) > VERIFICATION_FRAME_COUNT:
                    self._verification_buffer = self._verification_buffer[
                        -VERIFICATION_FRAME_COUNT:
                    ]
                count = self._consecutive_detections
                buffer = list(self._verification_buffer)
        else:
            # Below threshold - reset
            with self._lock:
                self._consecutive_detections = 0
                self._verification_buffer = []
            return

        # Check if temporal verification has passed:
        # Need at least VERIFICATION_FRAME_COUNT frames AND they must span
        # at least VERIFICATION_MIN_SPAN_SECONDS
        if count >= VERIFICATION_FRAME_COUNT and len(buffer) >= VERIFICATION_FRAME_COUNT:
            # Check time span between first and last frame in buffer
            first_time = buffer[0][2]
            last_time = buffer[-1][2]
            time_span = last_time - first_time

            if time_span < VERIFICATION_MIN_SPAN_SECONDS:
                # Not enough time has elapsed - keep waiting
                return

            with self._lock:
                self._capture_in_progress = True
                self._state = "threshold_reached"
                self._state_change_time = time.monotonic()

                # Pick the best frame from the verification buffer
                # (highest confidence detection)
                best_frame = None
                best_metadata = None
                best_conf = 0.0
                for buf_frame, buf_meta, _ts in self._verification_buffer:
                    frame_max = max(m["confidence"] for m in buf_meta)
                    if frame_max > best_conf:
                        best_conf = frame_max
                        best_frame = buf_frame
                        best_metadata = buf_meta

                # Reset verification state
                self._consecutive_detections = 0
                self._verification_buffer = []

            if best_frame is None:
                best_frame = raw_frame.copy()
                best_metadata = list(crack_metadata_list)

            capture_thread = threading.Thread(
                target=self._capture_worker,
                args=(best_frame, best_metadata),
                daemon=True,
            )
            capture_thread.start()

            # Start a timeout watchdog thread
            timeout_thread = threading.Thread(
                target=self._capture_timeout_watchdog,
                daemon=True,
            )
            timeout_thread.start()

    def _handle_cooldown(self) -> None:
        """Check if cooldown period has elapsed."""
        with self._lock:
            elapsed = time.monotonic() - self._cooldown_start
            if elapsed >= self._cooldown:
                self._state = "monitoring"
                self._state_change_time = time.monotonic()

    def _capture_timeout_watchdog(self) -> None:
        """
        Monitor the capture worker and force-reset to monitoring if it takes
        longer than CAPTURE_WORKER_TIMEOUT_SECONDS.
        """
        start = time.monotonic()
        while True:
            time.sleep(1.0)
            with self._lock:
                if not self._capture_in_progress:
                    # Capture finished normally
                    return
            elapsed = time.monotonic() - start
            if elapsed >= CAPTURE_WORKER_TIMEOUT_SECONDS:
                # Force reset
                with self._lock:
                    self._state = "monitoring"
                    self._state_change_time = time.monotonic()
                    self._capture_in_progress = False
                return

    def _capture_worker(
        self, raw_frame: np.ndarray, crack_metadata_list: list
    ) -> None:
        """
        Execute the full capture workflow in a background thread.

        Steps:
          1. Image quality validation
          2. Full analysis (annotate_frame on captured image)
          3. Crack validation (filter false positives)
          4. Save outputs (images, overlays, metadata, reports, DB)
        """
        try:
            with self._lock:
                self._state = "capturing"
                self._state_change_time = time.monotonic()

            # Step 1: Image quality validation
            quality_result = validate_image_quality(raw_frame)
            if not quality_result["passed"]:
                # Image quality too low - discard and return to monitoring
                with self._lock:
                    self._state = "monitoring"
                    self._state_change_time = time.monotonic()
                    self._capture_in_progress = False
                return

            # Encode the raw frame as high-quality JPEG
            timestamp = datetime.now(timezone.utc)
            ok, buf = cv2.imencode(
                ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            if not ok:
                with self._lock:
                    self._state = "monitoring"
                    self._state_change_time = time.monotonic()
                    self._capture_in_progress = False
                return

            raw_jpeg = buf.tobytes()

            with self._lock:
                self._state = "analyzing"
                self._state_change_time = time.monotonic()

            # Step 2: Run full analysis on the captured frame
            try:
                annotated, crack_present, analysis_metadata, analysis_masks = annotate_frame(
                    raw_frame,
                    weights=self._weights,
                    conf=self._conf,
                    imgsz=self._imgsz,
                )
            except Exception:
                annotated = raw_frame
                crack_present = False
                analysis_metadata = crack_metadata_list
                analysis_masks = None

            # If no cracks found in re-analysis, discard
            if not crack_present or not analysis_metadata:
                with self._lock:
                    self._state = "monitoring"
                    self._state_change_time = time.monotonic()
                    self._capture_in_progress = False
                return

            # Step 3: Advanced crack validation
            # Pass masks from the YOLO result to enable morphological checks
            validated_metadata = validate_cracks(
                analysis_metadata, raw_frame, masks=analysis_masks
            )

            if not validated_metadata:
                # All detections filtered as false positives
                with self._lock:
                    self._state = "monitoring"
                    self._state_change_time = time.monotonic()
                    self._capture_in_progress = False
                return

            # Encode annotated frame as JPEG for display
            ok_ann, buf_ann = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            annotated_jpeg = buf_ann.tobytes() if ok_ann else raw_jpeg

            # Pick the highest-confidence validated detection as primary
            primary = max(validated_metadata, key=lambda m: m["confidence"])

            with self._lock:
                self._capture_count += 1
                capture_num = self._capture_count
                camera_source = self._camera_source
                threshold = self._threshold

            # Build capture ID
            capture_id = f"INSP-{capture_num:05d}"
            h, w = raw_frame.shape[:2]
            image_resolution = f"{w}x{h}"

            # Crop crack region from raw frame using primary detection bbox
            crop_path = None
            try:
                bbox = primary["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                # Add padding (15% of bbox dimensions on each side)
                pad_x = int(bbox_w * CROP_PADDING_RATIO)
                pad_y = int(bbox_h * CROP_PADDING_RATIO)

                # Clamp to frame boundaries
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w, x2 + pad_x)
                crop_y2 = min(h, y2 + pad_y)

                cropped = raw_frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if cropped.size > 0:
                    os.makedirs(self._crops_dir, exist_ok=True)
                    crop_filename = f"{capture_id}_crop.jpg"
                    crop_path = os.path.join(self._crops_dir, crop_filename)
                    cv2.imwrite(crop_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except Exception:
                crop_path = None

            # Get GPS position for this capture
            gps_data = self._gps_provider.get_current_position()
            # Try timestamp-based matching if GPS log is loaded
            if gps_data["latitude"] is None and self._gps_provider.log_size > 0:
                gps_data = self._gps_provider.get_position_at_timestamp(timestamp)

            # Step 4: Save to new directory structure
            os.makedirs(self._images_dir, exist_ok=True)
            os.makedirs(self._overlays_dir, exist_ok=True)
            os.makedirs(self._metadata_dir, exist_ok=True)
            os.makedirs(self._reports_dir, exist_ok=True)
            os.makedirs(self._exports_dir, exist_ok=True)
            os.makedirs(self._crops_dir, exist_ok=True)

            image_filename = f"{capture_id}.jpg"
            image_path = os.path.join(self._images_dir, image_filename)
            overlay_path = os.path.join(self._overlays_dir, image_filename)
            metadata_path = os.path.join(
                self._metadata_dir, f"{capture_id}.json"
            )
            report_path = os.path.join(
                self._reports_dir, f"{capture_id}.pdf"
            )
            csv_path = os.path.join(self._exports_dir, "inspections.csv")

            # Save raw image
            with open(image_path, "wb") as f:
                f.write(raw_jpeg)

            # Save overlay image
            with open(overlay_path, "wb") as f:
                f.write(annotated_jpeg)

            # Build metadata
            metadata = {
                "capture_id": capture_id,
                "timestamp": timestamp.isoformat(),
                "capture_number": capture_num,
                "classification": primary["classification"],
                "confidence": primary["confidence"],
                "crack_area": primary["area_px"],
                "estimated_length": primary["estimated_length_px"],
                "estimated_width": primary["estimated_width_px"],
                "bbox": primary["bbox"],
                "camera_source": camera_source,
                "detection_threshold": threshold,
                "image_resolution": image_resolution,
                "image_path": image_path,
                "overlay_path": overlay_path,
                "metadata_path": metadata_path,
                "report_path": report_path,
                "crop_path": crop_path,
                "latitude": gps_data["latitude"],
                "longitude": gps_data["longitude"],
                "altitude": gps_data["altitude"],
                "gps_timestamp": gps_data["timestamp"],
                "all_detections": validated_metadata,
                "quality_check": quality_result,
            }

            # Save JSON metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Generate PDF report
            try:
                generate_pdf_report(
                    metadata, image_path, overlay_path, report_path,
                    crop_path=crop_path
                )
            except Exception:
                pass  # Non-fatal: report generation failure should not stop workflow

            # Append to CSV
            csv_record = {
                "capture_id": capture_id,
                "timestamp": timestamp.isoformat(),
                "classification": primary["classification"],
                "confidence": primary["confidence"],
                "crack_area": primary["area_px"],
                "estimated_length": primary["estimated_length_px"],
                "estimated_width": primary["estimated_width_px"],
                "bbox": str(primary["bbox"]),
                "camera_source": camera_source,
                "detection_threshold": threshold,
                "image_resolution": image_resolution,
                "latitude": gps_data["latitude"],
                "longitude": gps_data["longitude"],
                "altitude": gps_data["altitude"],
            }
            try:
                generate_csv_record(csv_record, csv_path)
            except Exception:
                pass

            # Insert into SQLite database
            db_record = {
                "capture_id": capture_id,
                "timestamp": timestamp.isoformat(),
                "confidence": primary["confidence"],
                "crack_area": primary["area_px"],
                "estimated_length": primary["estimated_length_px"],
                "estimated_width": primary["estimated_width_px"],
                "bbox": primary["bbox"],
                "camera_source": camera_source,
                "detection_threshold": threshold,
                "image_resolution": image_resolution,
                "classification": primary["classification"],
                "image_path": image_path,
                "overlay_path": overlay_path,
                "metadata_path": metadata_path,
                "report_path": report_path,
                "latitude": gps_data["latitude"],
                "longitude": gps_data["longitude"],
                "altitude": gps_data["altitude"],
                "gps_timestamp": gps_data["timestamp"],
                "crop_path": crop_path,
            }
            try:
                insert_inspection(db_record)
            except Exception:
                pass

            # Store in memory for frontend access
            with self._lock:
                self._latest_capture_jpeg = annotated_jpeg
                self._latest_metadata = metadata
                self._state = "analysis_complete"
                self._state_change_time = time.monotonic()

            # Hold analysis_complete state for frontend to observe
            time.sleep(ANALYSIS_COMPLETE_HOLD_SECONDS)

            with self._lock:
                self._state = "cooldown"
                self._cooldown_start = time.monotonic()
                self._state_change_time = time.monotonic()

        except Exception:
            # On any unexpected error, log the traceback and reset to monitoring
            logger.exception("Capture worker failed")
            with self._lock:
                self._state = "monitoring"
                self._state_change_time = time.monotonic()
        finally:
            with self._lock:
                self._capture_in_progress = False

    def _scan_existing_captures(self) -> int:
        """Scan captures/images/ directory to find the highest existing capture number."""
        import re

        if not os.path.isdir(self._images_dir):
            return 0

        pattern = re.compile(r"^INSP-(\d+)\.jpg$")
        max_num = 0
        for fname in os.listdir(self._images_dir):
            match = pattern.match(fname)
            if match:
                try:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                except (ValueError, IndexError):
                    pass
        return max_num
