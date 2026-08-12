#!/usr/bin/env python3
"""
app.py - Matanglawin photo-based crack inspection web app.

Two ways a photo gets inspected, both funnelled through ONE analysis
pipeline (inspection_service.InspectionService):

    1. Manual upload      - the user uploads a photo in the browser
                            (/  ->  /detect  ->  /inspection/<id>), or a
                            client POSTs to /api/inspect.
    2. DJI photo import   - the user drops a DJI-captured photo into the
                            watch folder; photo_import.PhotoImportWatcher
                            picks it up and analyzes it once.

The live DJI drone POV (/dashboard) is DISPLAY-ONLY: it is the raw
MediaMTX WebRTC feed embedded in the browser. The backend never reads
the RTSP stream and never runs YOLO on video - inference happens once
per still photo. The live-video path and the image-analysis path are
fully independent, so MediaMTX being down never blocks inspection.

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
    MATANGLAWIN_API_PORT     MediaMTX HTTP API port (default: 9997)
    MATANGLAWIN_IMPORT_DIR   Folder watched for DJI-imported photos
                             (default: <data>/import)
    MATANGLAWIN_GPS_LAT/LON/ALT  Manual GPS override (see gps_provider.py)
"""

import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    url_for,
    flash,
    redirect,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename

from inference_core import get_model
import network_config
import gps_provider
from inspection_db import InspectionDB
from inspection_service import InspectionService, InvalidImageError
from photo_import import PhotoImportWatcher
from report_generator import generate_single_report, generate_full_report

# ---------------------------------------------------------------------------
# Resource paths (read-only bundled files vs. writable runtime data).
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    RESOURCE_DIR = Path(sys._MEIPASS)                # read-only bundled files
    APP_DIR = Path(sys.executable).resolve().parent  # writable, next to the .exe
else:
    RESOURCE_DIR = Path(__file__).resolve().parent
    APP_DIR = RESOURCE_DIR

DATA_DIR = APP_DIR / "matanglawin_data"
UPLOAD_TMP_DIR = DATA_DIR / "upload_tmp"        # transient: incoming manual uploads
INSPECTIONS_DIR = DATA_DIR / "inspections"      # per-inspection results (original/highlighted/crack)
REPORT_DIR = DATA_DIR / "reports"               # generated PDF reports
IMPORT_DIR = Path(os.environ.get("MATANGLAWIN_IMPORT_DIR", str(DATA_DIR / "import")))
DB_PATH = DATA_DIR / "inspections.db"
for d in (UPLOAD_TMP_DIR, INSPECTIONS_DIR, REPORT_DIR, IMPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

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


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


# ---------------------------------------------------------------------------
# Lazily-created singletons (DB, analysis service, import watcher). Lazy so
# importing app.py (e.g. in tests) never loads YOLO weights or starts threads.
# ---------------------------------------------------------------------------
_db: InspectionDB = None
_service: InspectionService = None
_watcher: PhotoImportWatcher = None
_lock_obj = None


def _get_lock():
    global _lock_obj
    if _lock_obj is None:
        import threading
        _lock_obj = threading.RLock()
    return _lock_obj


def get_db() -> InspectionDB:
    global _db
    if _db is None:
        with _get_lock():
            if _db is None:
                _db = InspectionDB(str(DB_PATH))
    return _db


def get_service() -> InspectionService:
    global _service
    if _service is None:
        with _get_lock():
            if _service is None:
                _service = InspectionService(get_db(), str(INSPECTIONS_DIR), weights=WEIGHTS)
    return _service


def get_watcher() -> PhotoImportWatcher:
    global _watcher
    if _watcher is None:
        with _get_lock():
            if _watcher is None:
                _watcher = PhotoImportWatcher(str(IMPORT_DIR), get_service())
    return _watcher


# ---------------------------------------------------------------------------
# URL helpers for serving a record's result images.
# ---------------------------------------------------------------------------
def _inspection_urls(record) -> dict:
    return {
        "original": url_for("inspection_original", inspection_id=record.id),
        "highlighted": url_for("inspection_highlighted", inspection_id=record.id),
        "cracks": [
            url_for("inspection_crack", inspection_id=record.id, filename=name)
            for name in record.crack_image_names()
        ],
    }


# ===========================================================================
# Pages
# ===========================================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Manual upload -> central analysis -> redirect to the result page."""
    file = request.files.get("image")
    if file is None or file.filename == "":
        flash("Please choose an image file to upload.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG, PNG, BMP, TIF, or WEBP image.")
        return redirect(url_for("index"))

    ext = Path(secure_filename(file.filename)).suffix.lower()
    tmp_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(tmp_path)

    try:
        record = get_service().analyze_file(str(tmp_path), source="upload")
    except InvalidImageError:
        flash("That file could not be read as an image. Please try another photo.")
        return redirect(url_for("index"))
    except Exception as exc:  # noqa: BLE001
        flash(f"Inspection failed: {exc}")
        return redirect(url_for("index"))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    return redirect(url_for("inspection_result", inspection_id=record.id))


@app.route("/inspection/<int:inspection_id>", methods=["GET"])
def inspection_result(inspection_id):
    """Result page for a single inspection (upload or import)."""
    record = get_db().get_inspection(inspection_id)
    if record is None:
        abort(404)
    urls = _inspection_urls(record)
    return render_template(
        "result.html",
        status=record.status,
        has_crack=record.has_crack,
        num_instances=record.num_instances,
        timestamp=record.timestamp,
        source=record.source,
        gps_available=record.gps_available,
        latitude=record.latitude,
        longitude=record.longitude,
        original_url=urls["original"],
        highlighted_url=urls["highlighted"],
        crack_urls=urls["cracks"],
        inspection_id=record.id,
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Live display-only drone POV + latest inspection result."""
    return render_template("dashboard.html")


@app.route("/inspections", methods=["GET"])
def inspections_page():
    return render_template("inspections.html")


@app.route("/health", methods=["GET"])
def health():
    try:
        get_model(WEIGHTS)
        return {"status": "ok", "weights": WEIGHTS}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}, 500


# ===========================================================================
# APIs
# ===========================================================================
@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """Upload one image (multipart 'image') for analysis; returns JSON."""
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "no image provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    ext = Path(secure_filename(file.filename)).suffix.lower()
    tmp_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(tmp_path)
    try:
        record = get_service().analyze_file(str(tmp_path), source="upload")
    except InvalidImageError:
        return jsonify({"error": "invalid or corrupt image"}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"analysis failed: {exc}"}), 500
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    payload = record.to_dict()
    payload["urls"] = _inspection_urls(record)
    return jsonify(payload)


@app.route("/api/inspection/<int:inspection_id>", methods=["GET"])
def api_inspection(inspection_id):
    record = get_db().get_inspection(inspection_id)
    if record is None:
        abort(404)
    payload = record.to_dict()
    payload["urls"] = _inspection_urls(record)
    return jsonify(payload)


@app.route("/api/inspection/latest", methods=["GET"])
@app.route("/api/inspections/latest", methods=["GET"])  # alias
def api_inspection_latest():
    record = get_db().get_latest()
    if record is None:
        return jsonify(None)
    payload = record.to_dict()
    payload["urls"] = _inspection_urls(record)
    return jsonify(payload)


@app.route("/api/inspections", methods=["GET"])
def api_inspections():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        limit = 50
    db = get_db()
    records = db.list_inspections(limit=limit)
    items = []
    for r in records:
        d = r.to_dict()
        d["urls"] = _inspection_urls(r)
        items.append(d)
    return jsonify({"count": db.count(), "inspections": items})


def _send_record_image(inspection_id, which):
    record = get_db().get_inspection(inspection_id)
    if record is None:
        abort(404)
    path = record.original_image_path if which == "original" else record.highlighted_image_path
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path)


@app.route("/api/inspection/<int:inspection_id>/original", methods=["GET"])
def inspection_original(inspection_id):
    return _send_record_image(inspection_id, "original")


@app.route("/api/inspection/<int:inspection_id>/highlighted", methods=["GET"])
def inspection_highlighted(inspection_id):
    return _send_record_image(inspection_id, "highlighted")


@app.route("/api/inspection/<int:inspection_id>/crack/<path:filename>", methods=["GET"])
def inspection_crack(inspection_id, filename):
    record = get_db().get_inspection(inspection_id)
    if record is None:
        abort(404)
    # Only serve files that actually belong to this inspection's crack
    # set - never an arbitrary path from the URL.
    for p in record.crack_image_paths:
        if Path(p).name == filename and Path(p).exists():
            return send_file(p)
    abort(404)


@app.route("/api/network", methods=["GET"])
def api_network():
    try:
        info = network_config.get_network_info()
        return jsonify(info.to_dict())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"host_ip": None, "error": str(exc)}), 200


@app.route("/api/mediamtx/status", methods=["GET"])
def api_mediamtx_status():
    try:
        return jsonify(network_config.get_mediamtx_status())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"reachable": False, "error": str(exc)}), 200


@app.route("/api/stream/status", methods=["GET"])
def api_stream_status():
    """
    DISPLAY-ONLY live POV publisher state (LIVE/OFFLINE/UNKNOWN), from
    MediaMTX's HTTP API. This does NOT run YOLO or read the RTSP stream.
    """
    try:
        return jsonify(network_config.get_stream_status())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"pov_state": "UNKNOWN", "api_reachable": False, "error": str(exc)}), 200


@app.route("/api/gps", methods=["GET"])
def api_gps():
    fix = gps_provider.get_current_fix()
    return jsonify(fix.to_dict())


# ===========================================================================
# Reports
# ===========================================================================
@app.route("/reports/inspection/<int:inspection_id>.pdf", methods=["GET"])
def report_single(inspection_id):
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
    db = get_db()
    records = db.list_inspections(limit=1000)
    out_path = REPORT_DIR / "full_report.pdf"
    try:
        generate_full_report(records, str(out_path))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "detail": f"PDF generation failed: {exc}"}), 500
    return send_file(str(out_path), mimetype="application/pdf", as_attachment=True)


def _find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
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

    # Start watching the DJI photo import folder. This is independent of
    # MediaMTX/RTSP - imported/uploaded photos are analyzed whether or
    # not the live stream is up.
    try:
        get_watcher().start()
        print(f"  Watching for DJI photos in: {IMPORT_DIR}")
    except Exception as exc:  # noqa: BLE001 - watcher failure must not stop the web app
        print(f"  WARNING: photo import watcher failed to start: {exc}")

    print("=" * 60)
    print("  Matanglawin - Photo-based Crack Inspection")
    print(f"  Starting server at {url}")
    print("  Close this window to stop the app.")
    print("=" * 60)

    if getattr(sys, "frozen", False) or os.environ.get("AUTO_OPEN") == "1":
        import threading
        import webbrowser

        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
