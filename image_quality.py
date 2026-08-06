"""
image_quality.py - Image quality validation for captured frames.

Validates that a captured frame meets minimum quality standards before
running expensive crack analysis. Checks:
  - Blur detection (Laplacian variance)
  - Brightness validation (mean luminance)
  - Exposure validation (histogram distribution)

If any check fails, the capture is discarded and monitoring resumes.
"""

from __future__ import annotations

import cv2
import numpy as np


# Thresholds (tuned for typical UAV inspection images)
BLUR_THRESHOLD = 50.0          # Laplacian variance below this = blurry
BRIGHTNESS_MIN = 30.0          # Mean luminance below this = too dark
BRIGHTNESS_MAX = 230.0         # Mean luminance above this = too bright
EXPOSURE_DARK_RATIO = 0.60     # More than 60% pixels very dark = underexposed
EXPOSURE_BRIGHT_RATIO = 0.60   # More than 60% pixels very bright = overexposed
EXPOSURE_DARK_THRESHOLD = 30   # Pixel value considered "very dark"
EXPOSURE_BRIGHT_THRESHOLD = 225  # Pixel value considered "very bright"


def check_blur(frame: np.ndarray) -> dict:
    """
    Check if the frame is too blurry using Laplacian variance.

    Returns:
        dict with keys: passed (bool), laplacian_variance (float),
        threshold (float), detail (str)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    passed = variance >= BLUR_THRESHOLD
    detail = "OK" if passed else f"Image too blurry (variance={variance:.1f} < {BLUR_THRESHOLD})"

    return {
        "passed": passed,
        "laplacian_variance": variance,
        "threshold": BLUR_THRESHOLD,
        "detail": detail,
    }


def check_brightness(frame: np.ndarray) -> dict:
    """
    Check if the frame has acceptable brightness (mean luminance).

    Returns:
        dict with keys: passed (bool), mean_brightness (float),
        min_threshold (float), max_threshold (float), detail (str)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())

    passed = BRIGHTNESS_MIN <= mean_brightness <= BRIGHTNESS_MAX
    if mean_brightness < BRIGHTNESS_MIN:
        detail = f"Image too dark (mean={mean_brightness:.1f} < {BRIGHTNESS_MIN})"
    elif mean_brightness > BRIGHTNESS_MAX:
        detail = f"Image too bright (mean={mean_brightness:.1f} > {BRIGHTNESS_MAX})"
    else:
        detail = "OK"

    return {
        "passed": passed,
        "mean_brightness": mean_brightness,
        "min_threshold": BRIGHTNESS_MIN,
        "max_threshold": BRIGHTNESS_MAX,
        "detail": detail,
    }


def check_exposure(frame: np.ndarray) -> dict:
    """
    Check if the frame has acceptable exposure by analyzing histogram distribution.

    Rejects frames where too many pixels are clustered at extremes (under/overexposed).

    Returns:
        dict with keys: passed (bool), dark_ratio (float), bright_ratio (float),
        detail (str)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    dark_pixels = int(np.sum(gray < EXPOSURE_DARK_THRESHOLD))
    bright_pixels = int(np.sum(gray > EXPOSURE_BRIGHT_THRESHOLD))

    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels

    underexposed = dark_ratio > EXPOSURE_DARK_RATIO
    overexposed = bright_ratio > EXPOSURE_BRIGHT_RATIO

    passed = not underexposed and not overexposed

    if underexposed:
        detail = f"Underexposed ({dark_ratio:.1%} pixels below {EXPOSURE_DARK_THRESHOLD})"
    elif overexposed:
        detail = f"Overexposed ({bright_ratio:.1%} pixels above {EXPOSURE_BRIGHT_THRESHOLD})"
    else:
        detail = "OK"

    return {
        "passed": passed,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "detail": detail,
    }


def validate_image_quality(frame: np.ndarray) -> dict:
    """
    Run all image quality checks on a frame.

    Returns:
        dict with keys:
          - passed (bool): True only if ALL checks pass
          - checks (dict): individual check results keyed by name
          - detail (str): summary of failures or "All checks passed"
    """
    checks = {
        "blur": check_blur(frame),
        "brightness": check_brightness(frame),
        "exposure": check_exposure(frame),
    }

    all_passed = all(c["passed"] for c in checks.values())
    failures = [c["detail"] for c in checks.values() if not c["passed"]]

    return {
        "passed": all_passed,
        "checks": checks,
        "detail": "All checks passed" if all_passed else "; ".join(failures),
    }
