#!/usr/bin/env python3
"""
app.py - Matanglawin real-time drone-assisted crack detection dashboard.

Flask routes ONLY. All video-source handling lives in video_source.py,
all continuous inference lives in detector.py (which itself reuses the
same YOLO11-seg model + overlay logic as inference_core.py), and the
dashboard UI lives in templates/ + static/.

Routes:
    GET  /            Dashboard page (loading screen -> live dashboard).
    GET  /video_feed  MJPEG stream of the annotated live camera feed.
    GET  /status      JSON: {"camera_connected", "crack_present", "source"}.
    POST /set_source   Body: {"source": "webcam"|"droidcam"|"drone"}.
    GET  /sources     JSON list of available camera sources.
    GET  /health      Basic health check (also confirms the model loads).

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

Environment variables (optional):
    WEIGHTS        Path to the .pt weights file (default: best.pt)
    HOST           Host to bind to (default: 127.0.0.1)
    PORT           Preferred port (default: 5000)
    CONF            Detection confidence threshold (default: 0.25)
    TARGET_FPS      Cap on inference loop rate (default: 8)
    DEFAULT_SOURCE  "webcam" | "droidcam" | "drone" (default: webcam)
    DROIDCAM_URL    e.g. http://192.168.1.50:4747/video
    DRONE_URL       Placeholder URL/RTSP for a future drone camera feed
"""

import os
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from detector import Detector
from inference_core import get_model
from video_source import build_registry

# ---------------------------------------------------------------------------
# Resource paths (also supports being bundled into a one-file executable
# with PyInstaller, where read-only resources are extracted to sys._MEIPASS).
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    RESOURCE_DIR = Path(sys._MEIPASS)
else:
    RESOURCE_DIR = Path(__file__).resolve().parent

WEIGHTS = os.environ.get("WEIGHTS", str(RESOURCE_DIR / "best.pt"))
CONF = float(os.environ.get("CONF", 0.25))
TARGET_FPS = float(os.environ.get("TARGET_FPS", 8))
DEFAULT_SOURCE = os.environ.get("DEFAULT_SOURCE", "webcam")
DROIDCAM_URL = os.environ.get("DROIDCAM_URL", "http://192.168.1.50:4747/video")
DRONE_URL = os.environ.get("DRONE_URL", "http://192.168.1.60:8080/video")

app = Flask(
    __name__,
    template_folder=str(RESOURCE_DIR / "templates"),
    static_folder=str(RESOURCE_DIR / "static"),
)

REGISTRY = build_registry(droidcam_url=DROIDCAM_URL, drone_url=DRONE_URL)
detector = Detector(
    registry=REGISTRY,
    weights=WEIGHTS,
    conf=CONF,
    target_fps=TARGET_FPS,
    default_source_key=DEFAULT_SOURCE if DEFAULT_SOURCE in REGISTRY else "webcam",
)


@app.route("/", methods=["GET"])
def index():
    return render_template("dashboard.html", sources=detector.available_sources())


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = detector.get_latest_jpeg()
        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )
        time.sleep(0.05)


@app.route("/video_feed", methods=["GET"])
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status", methods=["GET"])
def status():
    return jsonify(detector.get_status())


@app.route("/sources", methods=["GET"])
def sources():
    return jsonify(detector.available_sources())


@app.route("/set_source", methods=["POST"])
def set_source():
    payload = request.get_json(silent=True) or {}
    source_key = payload.get("source", "")
    ok = detector.set_source(source_key)
    if not ok:
        return jsonify({"ok": False, "error": "Unknown source"}), 400
    return jsonify({"ok": True, "source": source_key})


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

    detector.start()

    print("=" * 60)
    print("  MATANGLAWIN - Real-Time Drone Visual Crack Detection")
    print(f"  Starting server at {url}")
    print("  Close this window to stop the app.")
    print("=" * 60)

    if getattr(sys, "frozen", False) or os.environ.get("AUTO_OPEN") == "1":
        import threading
        import webbrowser

        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
