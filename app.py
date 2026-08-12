#!/usr/bin/env python3
"""
app.py - Matanglawin crack-detection web app.

A small Flask website that lets a user upload a photo of a surface
(road, wall, pipe, etc.), runs it through the trained YOLO11-seg model
(best.pt) using the same overlay logic as infer_overlay.py, and shows
the crack-mask overlay result in the browser.

Also serves the live DJI Neo 2 drone-inspection workflow (see
network_config.py, live_pipeline.py, and README.md for the full
architecture): /dashboard shows the live drone POV (via MediaMTX
WebRTC) and detection status, /inspections lists every automatically
captured crack with PDF report downloads.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

Environment variables (optional):
    WEIGHTS                  Path to the .pt weights file (default: best.pt)
    PORT                     Port to listen on (default: 5000)
    MATANGLAWIN_HOST_IP      Force a specific LAN IP (default: auto-detect)
    MATANGLAWIN_STREAM_KEY   MediaMTX stream path/key (default: matanglawin)
    MATANGLAWIN_RTMP_PORT    MediaMTX RTMP port (default: 1935)
    MATANGLAWIN_RTSP_PORT    MediaMTX RTSP port (default: 8554)
    MATANGLAWIN_WEBRTC_PORT  MediaMTX WebRTC port (default: 8889)
    MATANGLAWIN_GPS_LAT/LON/ALT  Manual GPS override (see gps_provider.py)
"""

import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    url_for,
    flash,
    redirect,
    send_from_directory,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename

from inference_core import run_overlay, get_model
import network_config
import gps_provider
from inspection_db import InspectionDB
from live_pipeline import LivePipeline
from report_generator import generate_single_report, generate_full_report

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
CAPTURE_DIR = DATA_DIR / "captures"       # auto-captured cropped crack images from the live pipeline
REPORT_DIR = DATA_DIR / "reports"         # generated PDF inspection reports
DB_PATH = DATA_DIR / "inspections.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# Live drone inspection pipeline (DJI Neo 2 -> MediaMTX -> RTSP -> YOLO).
#
# This is initialized lazily (on first use) rather than at import time, so
# importing app.py (e.g. for the test suite) never tries to open a network
# socket, connect to MediaMTX, or load the YOLO model.
# ---------------------------------------------------------------------------
_db: InspectionDB = None
_pipeline: LivePipeline = None
_pipeline_lock_obj = None


def _get_lock():
    # threading.RLock (not Lock) - get_pipeline() calls get_db() while
    # already holding this lock on the same thread, which would deadlock
    # on a plain non-reentrant Lock.
    global _pipeline_lock_obj
    if _pipeline_lock_obj is None:
        import threading
        _pipeline_lock_obj = threading.RLock()
    return _pipeline_lock_obj


def get_db() -> InspectionDB:
    global _db
    if _db is None:
        with _get_lock():
            if _db is None:
                _db = InspectionDB(str(DB_PATH))
    return _db


def get_pipeline() -> LivePipeline:
    """
    Lazily create (and start) the background live-video pipeline the
    first time it's needed - e.g. the first dashboard load or the first
    /api/stream/status poll - rather than unconditionally at process
    startup, so the plain image-upload workflow never pays the cost of
    (or depends on) MediaMTX being available.
    """
    global _pipeline
    if _pipeline is None:
        with _get_lock():
            if _pipeline is None:
                db = get_db()
                info = network_config.get_network_info()
                _pipeline = LivePipeline(
                    rtsp_url=info.rtsp_url,
                    weights=WEIGHTS,
                    db=db,
                    capture_dir=str(CAPTURE_DIR),
                )
                _pipeline.start()
    return _pipeline


@app.route("/api/network", methods=["GET"])
def api_network():
    """
    Dynamic network configuration for the dashboard: current LAN IP,
    the DJI RTMP address to type into DJI Fly, the stream key, and the
    localhost RTSP/WebRTC URLs the AI pipeline and website use.

    Never crashes: if no LAN IP can be determined, host_ip/rtmp_* are
    simply null (see network_config.get_network_info()).
    """
    try:
        info = network_config.get_network_info()
        return jsonify(info.to_dict())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"host_ip": None, "error": str(exc)}), 200


@app.route("/api/mediamtx/status", methods=["GET"])
def api_mediamtx_status():
    """Best-effort reachability check of MediaMTX's RTMP/RTSP/WebRTC ports."""
    try:
        return jsonify(network_config.get_mediamtx_status())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"reachable": False, "error": str(exc)}), 200


@app.route("/api/stream/status", methods=["GET"])
def api_stream_status():
    """
    Live drone stream + detection pipeline status for the dashboard:
    OFFLINE / CONNECTING / LIVE, plus whether the YOLO model loaded ok.
    """
    try:
        pipeline = get_pipeline()
        status = pipeline.get_status()
        return jsonify(
            {
                "stream_state": status.stream_state,
                "detector_ready": status.detector_ready,
                "detector_error": status.detector_error,
                "last_detection_at": status.last_detection_at,
                "last_capture_id": status.last_capture_id,
                "frames_processed": status.frames_processed,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"stream_state": "OFFLINE", "error": str(exc)}), 200


@app.route("/api/gps", methods=["GET"])
def api_gps():
    """Current GPS fix (or Unavailable) - see gps_provider.py."""
    fix = gps_provider.get_current_fix()
    return jsonify(fix.to_dict())


@app.route("/api/inspections", methods=["GET"])
def api_inspections():
    """List recent inspection records (most recent first)."""
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        limit = 50
    db = get_db()
    records = db.list_inspections(limit=limit)
    return jsonify({"count": db.count(), "inspections": [r.to_dict() for r in records]})


@app.route("/api/inspections/latest", methods=["GET"])
def api_inspections_latest():
    """The most recently captured crack inspection, or null if none yet."""
    db = get_db()
    latest = db.get_latest()
    return jsonify(latest.to_dict() if latest else None)


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Live drone inspection dashboard: POV, detection status, network info."""
    return render_template("dashboard.html")


@app.route("/inspections", methods=["GET"])
def inspections_page():
    """Inspection history page with PDF report download links."""
    return render_template("inspections.html")


@app.route("/data/captures/<path:filename>")
def capture_file(filename):
    return send_from_directory(CAPTURE_DIR, filename)


@app.route("/reports/inspection/<int:inspection_id>.pdf", methods=["GET"])
def report_single(inspection_id):
    """Generate (or re-generate) and download a single-crack PDF report."""
    db = get_db()
    record = db.get_inspection(inspection_id)
    if record is None:
        abort(404)
    out_path = REPORT_DIR / f"inspection_{inspection_id}.pdf"
    try:
        generate_single_report(record, str(out_path))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "detail": f"PDF generation failed: {exc}"}), 500
    return send_file(str(out_path), mimetype="application/pdf", as_attachment=True)


@app.route("/reports/full.pdf", methods=["GET"])
def report_full():
    """Generate and download a PDF covering every recorded inspection."""
    db = get_db()
    records = db.list_inspections(limit=1000)
    out_path = REPORT_DIR / "full_report.pdf"
    try:
        generate_full_report(records, str(out_path))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "detail": f"PDF generation failed: {exc}"}), 500
    return send_file(str(out_path), mimetype="application/pdf", as_attachment=True)


@app.route("/stream/preview.jpg", methods=["GET"])
def stream_preview():
    """
    Latest annotated (overlay) frame from the live pipeline, as a single
    JPEG. This is a lightweight fallback/debug preview - the production
    "LIVE DRONE POV" on the dashboard consumes MediaMTX's own WebRTC
    endpoint directly in the browser, not this route (see requirement
    #6/#24 - inference stays server-side, video delivery stays on
    MediaMTX's WebRTC output).
    """
    import cv2

    pipeline = get_pipeline()
    frame = pipeline.get_latest_overlay_jpeg()
    if frame is None:
        abort(404)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        abort(500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


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
