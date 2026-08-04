#!/usr/bin/env python3
"""
app.py - Matanglawin crack-detection web app.

A small Flask website that lets a user upload a photo of a surface
(road, wall, pipe, etc.), runs it through the trained YOLO11-seg model
(best.pt) using the same overlay logic as infer_overlay.py, and shows
the crack-mask overlay result in the browser.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

Environment variables (optional):
    WEIGHTS   Path to the .pt weights file (default: best.pt)
    PORT      Port to listen on (default: 5000)
"""

import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename

from inference_core import run_overlay, get_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = os.environ.get("WEIGHTS", str(BASE_DIR / "best.pt"))
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("SECRET_KEY", "matanglawin-dev-secret")


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")

    if file is None or file.filename == "":
        flash("Please choose an image file to upload.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG, PNG, BMP, TIF, or WEBP image.")
        return redirect(url_for("index"))

    # Read tuning options from the form, with sane defaults.
    try:
        conf = float(request.form.get("conf", 0.25))
        alpha = float(request.form.get("alpha", 0.5))
        outline = int(request.form.get("outline", 2))
    except ValueError:
        conf, alpha, outline = 0.25, 0.5, 2
    conf = min(max(conf, 0.0), 1.0)
    alpha = min(max(alpha, 0.0), 1.0)
    outline = min(max(outline, 0), 10)

    # Save the upload under a unique name to avoid collisions.
    ext = Path(secure_filename(file.filename)).suffix.lower()
    uid = uuid.uuid4().hex
    upload_name = f"{uid}{ext}"
    upload_path = UPLOAD_DIR / upload_name
    file.save(upload_path)

    result_name = f"{uid}_overlay.jpg"
    result_path = RESULT_DIR / result_name

    try:
        info = run_overlay(
            image_path=str(upload_path),
            out_path=str(result_path),
            weights=WEIGHTS,
            conf=conf,
            imgsz=640,
            alpha=alpha,
            color=(0, 0, 255),  # red, B,G,R
            outline=outline,
        )
    except Exception as exc:  # noqa: BLE001 - surface the error to the user
        flash(f"Inference failed: {exc}")
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        original_url=url_for("static", filename=f"uploads/{upload_name}"),
        result_url=url_for("static", filename=f"results/{result_name}"),
        num_instances=info["num_instances"],
        conf=conf,
        alpha=alpha,
        outline=outline,
    )


@app.route("/health", methods=["GET"])
def health():
    """Simple health check that also confirms the model can be loaded."""
    try:
        get_model(WEIGHTS)
        return {"status": "ok", "weights": WEIGHTS}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
