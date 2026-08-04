#!/usr/bin/env python3
"""
main.py - FastAPI backend for the Matanglawin crack-detection website.

Wraps the same YOLO11-seg overlay logic used in infer_overlay.py (mask
union -> semi-transparent fill + contour outline) behind a small HTTP API,
and serves the static frontend (frontend/index.html, style.css, app.js).

Run locally:
    cd webapp
    pip install -r requirements.txt
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser.
"""

import base64
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
REPO_ROOT = WEBAPP_DIR.parent
FRONTEND_DIR = WEBAPP_DIR / "frontend"
WEIGHTS_PATH = REPO_ROOT / "best.pt"

# ---------------------------------------------------------------------------
# Model - loaded once at startup, reused for every request.
# ---------------------------------------------------------------------------
_model = None


def get_model():
    """Lazily load the YOLO11-seg model (kept in memory after first call)."""
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
        # Imported here so the server can still start up (and show a clear
        # error) even if ultralytics/torch aren't installed yet.
        from ultralytics import YOLO

        _model = YOLO(str(WEIGHTS_PATH))
    return _model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Matanglawin Crack Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_color(color_str: str) -> tuple:
    """Parse an 'R,G,B' string (frontend order) into OpenCV's B,G,R tuple."""
    try:
        r, g, b = (int(c.strip()) for c in color_str.split(","))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="color must be 'R,G,B'") from exc
    clamp = lambda v: max(0, min(255, v))  # noqa: E731
    return (clamp(b), clamp(g), clamp(r))


@app.get("/api/health")
def health():
    return {"status": "ok", "weights_found": WEIGHTS_PATH.exists()}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
    alpha: float = Form(0.5),
    color: str = Form("255,0,0"),  # R,G,B from the frontend color picker
    outline: int = Form(2),
):
    """Run YOLO11-seg crack detection and return an overlay image + stats."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.") from exc

    # PIL (RGB) -> OpenCV (BGR), matching infer_overlay.py's expectations.
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    bgr_color = parse_color(color)

    model = get_model()
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )
    result = results[0]

    num_detections = 0 if result.masks is None else len(result.masks)

    if result.masks is None or num_detections == 0:
        blended = image
    else:
        masks = result.masks.data.cpu().numpy()  # (N, H, W) in [0, 1]
        union = np.any(masks > 0.5, axis=0).astype(np.uint8)

        if union.shape != (h, w):
            union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

        color_layer = np.zeros_like(image)
        color_layer[:] = bgr_color
        mask_bool = union.astype(bool)
        blended = image.copy()
        blended[mask_bool] = cv2.addWeighted(
            image, 1 - alpha, color_layer, alpha, 0
        )[mask_bool]

        if outline > 0:
            contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, bgr_color, outline)

    ok, buf = cv2.imencode(".png", blended)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image.")

    encoded = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "detections": num_detections,
        "width": w,
        "height": h,
        "image": f"data:image/png;base64,{encoded}",
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
