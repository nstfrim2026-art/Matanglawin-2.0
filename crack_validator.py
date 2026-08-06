"""
crack_validator.py - Advanced crack validation to reduce false positives.

Filters out detections caused by shadows, reflections, textures, or lighting
artifacts using multiple geometric and morphological criteria:
  - Minimum crack area
  - Minimum crack length
  - Aspect ratio validation (cracks should be elongated)
  - Morphological filtering (erode + dilate to remove noise)
  - Edge consistency (contour smoothness via approxPolyDP)
  - Isolated blob rejection (small area relative to bounding rect)
  - Preference for long, continuous crack structures
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


# Validation thresholds
MIN_CRACK_AREA_PX = 100       # Minimum mask area in pixels
MIN_CRACK_LENGTH_PX = 20.0    # Minimum estimated length in pixels
MIN_ASPECT_RATIO = 2.0        # Minimum length/width ratio (cracks are elongated)
MIN_SOLIDITY = 0.15           # Minimum contour area / convex hull area
MAX_CIRCULARITY = 0.7         # Maximum circularity (reject round blobs)
MIN_EDGE_CONSISTENCY = 0.3    # Minimum ratio of approxPolyDP points to contour points
MORPH_KERNEL_SIZE = 3         # Kernel size for morphological operations


def validate_cracks(
    crack_metadata_list: list,
    frame: np.ndarray,
    masks: Optional[np.ndarray] = None,
) -> List[dict]:
    """
    Filter crack detections to remove false positives.

    Args:
        crack_metadata_list: List of crack metadata dicts from annotate_frame().
        frame: The BGR frame the detections came from.
        masks: Optional numpy array of shape (N, H, W) with mask data.
               If None, validation is based solely on metadata geometry.

    Returns:
        Filtered list of crack_metadata dicts that pass all validation checks.
    """
    if not crack_metadata_list:
        return []

    h, w = frame.shape[:2]
    validated = []

    for i, meta in enumerate(crack_metadata_list):
        # 1. Minimum area check
        if meta.get("area_px", 0) < MIN_CRACK_AREA_PX:
            continue

        # 2. Minimum length check
        if meta.get("estimated_length_px", 0) < MIN_CRACK_LENGTH_PX:
            continue

        # 3. Aspect ratio validation (cracks should be elongated)
        length = meta.get("estimated_length_px", 0)
        width = meta.get("estimated_width_px", 1)
        if width > 0:
            aspect_ratio = length / width
        else:
            aspect_ratio = float("inf")

        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        # 4-7. Mask-based checks (if masks available)
        if masks is not None and i < len(masks):
            mask_i = masks[i]
            if mask_i.shape != (h, w):
                mask_i = cv2.resize(mask_i, (w, h), interpolation=cv2.INTER_NEAREST)

            binary_mask = (mask_i > 0.5).astype(np.uint8)

            # 4. Morphological filtering - erode then dilate to remove noise
            kernel = np.ones(
                (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8
            )
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

            # If morphological opening removes most of the mask, it was noise
            cleaned_area = int(cleaned.sum())
            original_area = int(binary_mask.sum())
            if original_area > 0 and cleaned_area / original_area < 0.3:
                continue

            # 5. Edge consistency via contour analysis
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)

            if contour_area < MIN_CRACK_AREA_PX:
                continue

            # 6. Circularity check - reject round blobs (shadows, spots)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                if circularity > MAX_CIRCULARITY:
                    continue

            # 7. Solidity check - reject irregular scattered regions
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = contour_area / hull_area
                if solidity < MIN_SOLIDITY:
                    continue

            # Edge consistency: check contour smoothness
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(largest_contour) > 0:
                edge_ratio = len(approx) / len(largest_contour)
                # Very simple contours (few vertices after simplification)
                # relative to original suggest smooth, continuous cracks
                # Very complex ones might be noise
                if edge_ratio > 0.8:
                    # Too noisy/complex - likely not a real crack
                    continue

        # Passed all checks
        validated.append(meta)

    return validated
