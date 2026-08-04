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
import sys
import uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename

from inference_core import run_overlay, get_model

# ---------------------------------------------------------------------------
# Resource paths.
#
# When this app is bundled into a one-file executable with PyInstaller, all
# read-only resources (templates/, static/style.css, best.pt) are extracted
# to a temporary directory exposed as `sys._MEIPASS`. That directory is
# read-only for practical purposes, so anything the app *writes* at runtime
# (uploaded images, result overlays) is instead stored in a separate,
# writable directory next to the executable (or next to app.py in dev mode).
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    RESOURCE_DIR = Path(sys._MEIPASS)          # read-only bundled files
    APP_DIR = Path(sys.executable).resolve().parent  # writable, next to the .exe
else:
    RESOURCE_DIR = Path(__file__).resolve().parent
    APP_DIR = RESOURCE_DIR

DATA_DIR = APP_DIR / "matanglawin_data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = os.environ.get("WEIGHTS", str(RESOURCE_DIR / "best.pt"))
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

app = Flask(
    __name__,
    template_folder=str(RESOURCE_DIR / "templates"),
    static_folder=str(RESOURCE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("SECRET_KEY", "matanglawin-dev-secret")


@app.route("/data/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/data/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename)


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
        original_url=url_for("uploaded_file", filename=upload_name),
        result_url=url_for("result_file", filename=result_name),
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


def _find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    """Return `preferred` if free, otherwise ask the OS for any free port."""
    import socket

    for candidate in [preferred, 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, candidate))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Could not find a free port")


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    preferred_port = int(os.environ.get("PORT", 5000))
    port = _find_free_port(preferred_port, host)
    url = f"http://{host}:{port}/"

    print("=" * 60)
    print("  Matanglawin - Crack Detection")
    print(f"  Starting server at {url}")
    print("  Close this window to stop the app.")
    print("=" * 60)

    # Open the default browser shortly after the server starts, but only
    # when running as the packaged executable (or when explicitly asked
    # to in dev mode via AUTO_OPEN=1), so `flask run`/debugging isn't
    # interrupted by extra browser tabs on every auto-reload.
    if getattr(sys, "frozen", False) or os.environ.get("AUTO_OPEN") == "1":
        import threading
        import webbrowser

        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
