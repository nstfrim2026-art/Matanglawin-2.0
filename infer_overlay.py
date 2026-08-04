#!/usr/bin/env python3
"""
infer_overlay.py - Overlay predicted crack masks onto an input image.

Versatile / portable: runs anywhere Python runs (VS Code, PyCharm, plain
terminal, Windows / macOS / Linux). NOT tied to Google Colab.

Every time you run it, it ASKS YOU FOR AN IMAGE:
  1. It first tries to open a native file-picker dialog (via tkinter).
  2. If no GUI is available (e.g. a headless server / SSH session), it
     falls back to typing the image path in the console.

It runs YOLO11-seg prediction, then paints each predicted crack mask onto
the original image as a semi-transparent fill that follows the *exact
crack contour* (the pixel-mask outline), plus a crisp solid outline along
that same contour. No bounding boxes are drawn. The result is saved next
to the input image and (if a display is available) shown in a window.

This script is fully INDEPENDENT of training: it only needs a trained
YOLO-seg weights file (e.g. best.pt). Rerun it anytime without retraining.

Install
-------
    pip install ultralytics opencv-python

Usage
-----
    # Simplest - will prompt for the image, uses ./best.pt as weights:
    python infer_overlay.py

    # Point at specific weights and tweak options:
    python infer_overlay.py --weights path/to/best.pt --conf 0.25 \
        --alpha 0.5 --color 0,0,255 --outline 2

Notes
-----
- --color is given as B,G,R (OpenCV order). Default is red (0,0,255).
- retina_masks=True makes the predicted masks match the original image
  resolution exactly, so the overlay hugs the true crack shape.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_color(value: str) -> tuple:
    """Parse a 'B,G,R' string into an (int, int, int) tuple."""
    parts = [int(c.strip()) for c in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be 'B,G,R', e.g. 0,0,255")
    return tuple(max(0, min(255, c)) for c in parts)


def ask_for_image() -> Path:
    """Ask the user for an image every run: GUI file dialog, else console."""
    # 1) Try a native file-picker dialog (works on desktop OSes).
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()          # hide the empty root window
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select an image to run crack segmentation on",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.update()
        root.destroy()
        if path:
            return Path(path)
    except Exception:
        pass  # no display / tkinter unavailable -> fall back to console

    # 2) Console fallback (headless servers, SSH, etc.).
    while True:
        raw = input("Enter path to an image (or 'q' to quit): ").strip().strip('"').strip("'")
        if raw.lower() in {"q", "quit", "exit"}:
            raise SystemExit("No image selected. Exiting.")
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        print(f"  '{p}' is not a file - try again.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay predicted crack masks on an image.")
    p.add_argument("--weights", default="best.pt", help="Path to trained YOLO-seg weights (default: best.pt).")
    p.add_argument("--out", default=None, help="Output path. Default: <image>_overlay.jpg next to the input.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25).")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640).")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask fill opacity 0..1 (default 0.5).")
    p.add_argument("--color", type=parse_color, default=(0, 0, 255), help="Fill/outline color as B,G,R (default red).")
    p.add_argument("--outline", type=int, default=2, help="Contour outline thickness in px; 0 disables (default 2).")
    p.add_argument("--device", default=None, help="Device, e.g. '0', 'cpu'. Default: auto.")
    p.add_argument("--no-show", action="store_true", help="Do not open a preview window (just save).")
    return p.parse_args()


def show_window(image) -> None:
    """Best-effort preview window; silently skips if no display is available."""
    try:
        cv2.imshow("Crack overlay (press any key to close)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}. Pass --weights /path/to/best.pt"
        )

    # Always ask for an image, every run.
    src = ask_for_image()
    print("Using image:", src)

    out_path = Path(args.out) if args.out else src.with_name(f"{src.stem}_overlay.jpg")

    # Load the trained model (weights only - no training state needed).
    model = YOLO(str(weights))

    # retina_masks=True -> masks returned at ORIGINAL image resolution.
    results = model.predict(
        source=str(src),
        conf=args.conf,
        imgsz=args.imgsz,
        retina_masks=True,
        device=args.device,
        verbose=False,
    )
    result = results[0]

    image = result.orig_img.copy()   # BGR
    h, w = image.shape[:2]

    if result.masks is None or len(result.masks) == 0:
        print("No cracks detected - saving a copy of the original image "
              "(try a lower --conf, e.g. 0.1).")
        cv2.imwrite(str(out_path), image)
        print(f"Saved: {out_path.resolve()}")
        if not args.no_show:
            show_window(image)
        return

    # Combine all instance masks into one binary union mask (single class).
    masks = result.masks.data.cpu().numpy()          # (N, H, W) in [0,1]
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)

    # Safety: ensure the mask matches the image size.
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

    # 1) Semi-transparent color fill following the exact mask shape.
    color_layer = np.zeros_like(image)
    color_layer[:] = args.color
    mask_bool = union.astype(bool)
    blended = image.copy()
    blended[mask_bool] = cv2.addWeighted(
        image, 1 - args.alpha, color_layer, args.alpha, 0
    )[mask_bool]

    # 2) Crisp solid outline along the true crack contour (not a bbox).
    if args.outline > 0:
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, args.color, args.outline)

    cv2.imwrite(str(out_path), blended)
    print(f"Detected {len(result.masks)} crack instance(s). Saved: {out_path.resolve()}")
    if not args.no_show:
        show_window(blended)


if __name__ == "__main__":
    main()
