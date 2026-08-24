#!/usr/bin/env python3
"""
app.py - MatanglaWIN UAV crack-inspection web app.

TWO completely independent pipelines, exactly as required:

  Pipeline A - LIVE DRONE POV (display only):
      DJI Neo 2 -> DJI Fly -> RTMP -> MediaMTX -> MatanglaWIN -> clean POV
      The live feed is embedded straight from MediaMTX's WebRTC endpoint.
      The backend NEVER reads the video, NEVER runs YOLO on it, and shows
      NO overlay/box/mask on it. It is for monitoring only.

  Pipeline B - DJI STILL PHOTO INSPECTION (automatic):
      DJI Neo 2 -> operator presses PHOTO -> automatic transfer bridge
      -> MatanglaWIN -> InspectionService -> best.pt/YOLO segmentation
      -> CRACK DETECTED / NO CRACK DETECTED + original + red-highlighted
      -> automatic website update -> inspection history.
      No manual upload step and no Analyze button for DJI photos.

      The original photo reaches MatanglaWIN by EITHER transport, both of
      which feed the exact same InspectionService and are de-duplicated:
        * a watch folder (photo_import.PhotoImportWatcher) - point any
          folder-sync tool at MATANGLAWIN_IMPORT_DIR; or
        * POST /api/import - the companion uploader (dji_photo_bridge.py)
          pushes the original JPEG straight to the PC over the LAN.
      A small PHOTO BRIDGE status (/api/bridge/status) reflects the
      companion uploader's heartbeat.

Both pipelines are independent: MediaMTX being down never blocks photo
inspection, and a running inspection never interrupts the live POV.

A manual upload path (/upload -> /detect, or POST /api/inspect) is kept
ONLY as a fallback/testing convenience; it converges on the very same
InspectionService as the automatic DJI import.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

Environment variables (optional):
    WEIGHTS                  Path to the .pt weights file (default: best.pt)
    HOST / PORT              Bind host / preferred port
    MATANGLAWIN_HOST_IP      Force a specific LAN IP (default: auto-detect)
    MATANGLAWIN_STREAM_KEY   MediaMTX stream path/key (default: matanglawin)
    MATANGLAWIN_RTMP_PORT    MediaMTX RTMP port (default: 1935)
    MATANGLAWIN_RTSP_PORT    MediaMTX RTSP port (default: 8554)
    MATANGLAWIN_WEBRTC_PORT  MediaMTX WebRTC port (default: 8889)
    MATANGLAWIN_API_PORT     MediaMTX HTTP API port (default: 9997)
    MATANGLAWIN_IMPORT_DIR   Folder watched for DJI-imported photos
                             (default: <data>/import)
"""

import logging
import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

import network_config
from bridge_status import BridgeStatus
from detector import (
    DEFAULT_AUGMENT,
    DEFAULT_CLAHE_CLIP,
    DEFAULT_CONF,
    DEFAULT_ENHANCE,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_THICKNESS_FRAC,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_MIN_LENGTH_FRAC,
    DEFAULT_MIN_THINNESS,
    DEFAULT_TILE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILED,
    DEFAULT_UNSHARP,
)
from capture_service import CaptureService, CaptureError, default_pictures_dir
from import_ledger import ImportLedger, hash_bytes
import telemetry_store as _telemetry_module
from telemetry_store import TelemetryStore
from srt_watcher import SrtWatcher
from inference_core import DEFAULT_REFINE, get_model
from inspection_db import InspectionDB
from inspection_service import InspectionService, InvalidImageError
from photo_import import PhotoImportWatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("matanglawin")

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
IMPORT_TMP_DIR = DATA_DIR / "import_tmp"        # transient: incoming bridge (HTTP) photos
INSPECTIONS_DIR = DATA_DIR / "inspections"      # per-inspection results (original/highlighted)
IMPORT_DIR = Path(os.environ.get("MATANGLAWIN_IMPORT_DIR", str(DATA_DIR / "import")))
IMPORT_LEDGER_PATH = DATA_DIR / "import_http_state.json"  # cross-restart de-dup for /api/import
DB_PATH = DATA_DIR / "inspections.db"
for d in (UPLOAD_TMP_DIR, IMPORT_TMP_DIR, INSPECTIONS_DIR, IMPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _configured_dirs(env_name: str, default: str = "") -> list:
    """
    Resolve one-or-more directories from an env var (os.pathsep-separated),
    expanding ~ and environment vars like %USERPROFILE% / $HOME. Never
    hardcodes a username. Falls back to `default` when unset.
    """
    raw = os.environ.get(env_name, "").strip()
    parts = raw.split(os.pathsep) if raw else ([default] if default else [])
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(os.path.expanduser(os.path.expandvars(p)))
    return out


# Folders watched for DJI SRT (Video Subtitles) telemetry files. Default is a
# local folder next to the app; point MATANGLAWIN_SRT_DIR (and optionally
# MATANGLAWIN_CAPTURE_DIR) at wherever your SRT files land - e.g.
#   set MATANGLAWIN_SRT_DIR=%USERPROFILE%\Videos\DJI
SRT_DIRS = _configured_dirs("MATANGLAWIN_SRT_DIR", str(DATA_DIR / "srt")) + \
    _configured_dirs("MATANGLAWIN_CAPTURE_DIR", "")

# Local offline map tiles served at /maps/<z>/<x>/<y>.png. Configurable so the
# tile dataset can be swapped without touching the frontend. Default is
# static/maps (the operator drops a real offline tile pack there - see README;
# no tiles are fabricated). Override with MATANGLAWIN_MAP_TILES_DIR.
MAP_TILES_DIR = Path(
    os.environ.get("MATANGLAWIN_MAP_TILES_DIR", str(RESOURCE_DIR / "static" / "maps"))
)
try:
    MAP_TILES_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # read-only (e.g. bundled) - operator sets MATANGLAWIN_MAP_TILES_DIR

# Where photos captured from the live MediaMTX stream are written. Default is
# the DJI-style pictures folder in the user's profile; override with
# MATANGLAWIN_PICTURES_DIR. Created lazily on first capture (never crashes if
# it can't be created up front).
PICTURES_DIR = default_pictures_dir()

WEIGHTS = os.environ.get("WEIGHTS", str(RESOURCE_DIR / "best.pt"))
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Generous limit - a DJI Neo 2 full-resolution JPEG is only a few MB, but
# leave headroom so an original is never rejected for size.
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

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
# Inference-time detection knobs (env-configurable; NO retraining needed).
# These tune recall on thin/hairline cracks in high-resolution DJI stills.
# They only affect what the model looks at - never the stored/displayed
# original, the live POV, or the (metric-free) result UI.
#
#   Recall (find faint/thin cracks):
#     MATANGLAWIN_CONF          detection threshold        (default 0.20)
#     MATANGLAWIN_IMGSZ         whole-image inference size (default 1280)
#     MATANGLAWIN_TILED         tiled/sliced inference 1/0 (default 1)
#     MATANGLAWIN_TILE          tile size in px            (default 1024)
#     MATANGLAWIN_TILE_OVERLAP  tile overlap fraction      (default 0.2)
#     MATANGLAWIN_ENHANCE       enhance input 1/0          (default 1)
#     MATANGLAWIN_CLAHE_CLIP    CLAHE clip limit           (default 1.5)
#     MATANGLAWIN_UNSHARP       unsharp mask 1/0           (default 0)
#     MATANGLAWIN_AUGMENT       test-time augmentation 1/0 (default 0)
#   Precision (don't paint texture/stains/beams/sills as cracks):
#     MATANGLAWIN_MAX_THICKNESS_FRAC strip regions wider than this frac of
#                                    the short side  (default 0.03; 0=off)
#     MATANGLAWIN_MIN_AREA_PX        noise floor in mask px       (default 60)
#     MATANGLAWIN_MIN_THINNESS       reject compact blobs         (default 3.0)
#     MATANGLAWIN_MIN_LENGTH_FRAC    drop short fragments (frac)  (default 0.05)
#   Overlay fidelity:
#     MATANGLAWIN_REFINE             tighten red mask to the crack 1/0 (default 1)
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def detector_config() -> dict:
    """Assemble the detector knobs from the environment (falling back to defaults)."""
    return {
        "conf": _env_float("MATANGLAWIN_CONF", DEFAULT_CONF),
        "imgsz": _env_int("MATANGLAWIN_IMGSZ", DEFAULT_IMGSZ),
        "min_area_px": _env_int("MATANGLAWIN_MIN_AREA_PX", DEFAULT_MIN_AREA_PX),
        "tiled": _env_bool("MATANGLAWIN_TILED", DEFAULT_TILED),
        "tile": _env_int("MATANGLAWIN_TILE", DEFAULT_TILE),
        "tile_overlap": _env_float("MATANGLAWIN_TILE_OVERLAP", DEFAULT_TILE_OVERLAP),
        "enhance": _env_bool("MATANGLAWIN_ENHANCE", DEFAULT_ENHANCE),
        "augment": _env_bool("MATANGLAWIN_AUGMENT", DEFAULT_AUGMENT),
        "clahe_clip": _env_float("MATANGLAWIN_CLAHE_CLIP", DEFAULT_CLAHE_CLIP),
        "unsharp": _env_bool("MATANGLAWIN_UNSHARP", DEFAULT_UNSHARP),
        "max_thickness_frac": _env_float("MATANGLAWIN_MAX_THICKNESS_FRAC", DEFAULT_MAX_THICKNESS_FRAC),
        "min_thinness": _env_float("MATANGLAWIN_MIN_THINNESS", DEFAULT_MIN_THINNESS),
        "min_length_frac": _env_float("MATANGLAWIN_MIN_LENGTH_FRAC", DEFAULT_MIN_LENGTH_FRAC),
        "refine": _env_bool("MATANGLAWIN_REFINE", DEFAULT_REFINE),
    }


# ---------------------------------------------------------------------------
# Lazily-created singletons (DB, analysis service, import watcher). Lazy so
# importing app.py (e.g. in tests) never loads YOLO weights or starts threads.
# ---------------------------------------------------------------------------
_db: InspectionDB = None
_service: InspectionService = None
_watcher: PhotoImportWatcher = None
_import_ledger: ImportLedger = None
_capture_service: CaptureService = None
_lock_obj = None

# Liveness of the automatic photo-transfer bridge (companion uploader).
# Cheap to construct, so it is a plain module-level singleton.
bridge_status = BridgeStatus()

# Latest GPS position (offline geotagging). Samples arrive at POST
# /api/telemetry (the phone GPS collector, e.g. Colota) and captures are
# stamped with the sample nearest in time. The latest sample is persisted
# locally so it survives a restart.
telemetry_store = TelemetryStore(
    persist_path=str(DATA_DIR / "telemetry_latest.json"),
    max_age_ms=_telemetry_module.phone_gps_max_age_ms(),
    stale_ms=_telemetry_module.phone_gps_max_age_ms(),
)
_srt_watcher: SrtWatcher = None


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
                import telemetry_store as _ts
                _service = InspectionService(
                    get_db(), str(INSPECTIONS_DIR), weights=WEIGHTS,
                    telemetry_store=telemetry_store,
                    radius_m=_ts.phone_gps_radius_m(), **detector_config()
                )
    return _service


def get_watcher() -> PhotoImportWatcher:
    global _watcher
    if _watcher is None:
        with _get_lock():
            if _watcher is None:
                _watcher = PhotoImportWatcher(str(IMPORT_DIR), get_service())
    return _watcher


def _live_rtsp_url() -> str:
    """Current MediaMTX RTSP URL (localhost) for on-demand frame capture."""
    return network_config.get_network_info().rtsp_url


def get_capture_service() -> CaptureService:
    global _capture_service
    if _capture_service is None:
        with _get_lock():
            if _capture_service is None:
                _capture_service = CaptureService(
                    get_service(),
                    rtsp_url_provider=_live_rtsp_url,
                    pictures_dir=str(PICTURES_DIR),
                    tmp_dir=str(IMPORT_TMP_DIR / "capture"),
                )
    return _capture_service


def get_srt_watcher() -> SrtWatcher:
    global _srt_watcher
    if _srt_watcher is None:
        with _get_lock():
            if _srt_watcher is None:
                _srt_watcher = SrtWatcher(SRT_DIRS, telemetry_store, db=get_db())
    return _srt_watcher


def get_import_ledger() -> ImportLedger:
    global _import_ledger
    if _import_ledger is None:
        with _get_lock():
            if _import_ledger is None:
                _import_ledger = ImportLedger(str(IMPORT_LEDGER_PATH))
    return _import_ledger


# ---------------------------------------------------------------------------
# URL helper for serving a record's result images. ONLY the untouched
# original and the red-highlighted image exist - never a crack-only image.
# ---------------------------------------------------------------------------
def _inspection_urls(record) -> dict:
    return {
        "original": url_for("inspection_original", inspection_id=record.id),
        "highlighted": url_for("inspection_highlighted", inspection_id=record.id),
    }


def _record_payload(record) -> dict:
    payload = record.to_dict()
    payload["urls"] = _inspection_urls(record)
    return payload


# ===========================================================================
# Pages
# ===========================================================================
@app.route("/", methods=["GET"])
def index():
    """Main operator view: the wide live drone POV + latest inspection."""
    return render_template("dashboard.html")


# Backwards-compatible alias for the dashboard.
@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/upload", methods=["GET"])
def upload_page():
    """Manual-upload FALLBACK page (testing / no-drone use only)."""
    return render_template("upload.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Manual upload (fallback) -> central analysis -> result page."""
    file = request.files.get("image")
    if file is None or file.filename == "":
        flash("Please choose an image file to upload.")
        return redirect(url_for("upload_page"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG, PNG, BMP, TIF, or WEBP image.")
        return redirect(url_for("upload_page"))

    ext = Path(secure_filename(file.filename)).suffix.lower()
    tmp_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(tmp_path)

    try:
        record = get_service().analyze_file(str(tmp_path), source="upload")
    except InvalidImageError:
        flash("That file could not be read as an image. Please try another photo.")
        return redirect(url_for("upload_page"))
    except Exception as exc:  # noqa: BLE001
        flash(f"Inspection failed: {exc}")
        return redirect(url_for("upload_page"))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    return redirect(url_for("inspection_result", inspection_id=record.id))


@app.route("/inspection/<int:inspection_id>", methods=["GET"])
def inspection_result(inspection_id):
    """Result page for a single inspection (upload or DJI import)."""
    record = get_db().get_inspection(inspection_id)
    if record is None:
        abort(404)
    urls = _inspection_urls(record)
    return render_template(
        "result.html",
        status=record.status,
        has_crack=record.has_crack,
        timestamp=record.timestamp,
        source_label=record.source_label,
        original_url=urls["original"],
        highlighted_url=urls["highlighted"],
        inspection_id=record.id,
        latitude=record.latitude,
        longitude=record.longitude,
        gps_available=record.gps_available,
    )


@app.route("/inspections", methods=["GET"])
def inspections_page():
    return render_template("inspections.html")


@app.route("/map", methods=["GET"])
def map_page():
    """Offline inspection map (red = crack, green = clear, blue = current drone)."""
    return render_template("map.html")


@app.route("/maps/<int:z>/<int:x>/<int:y>.png", methods=["GET"])
def map_tile(z, x, y):
    """
    Serve a single offline map tile from the local tile store
    (MAP_TILES_DIR/<z>/<x>/<y>.png). z/x/y are ints (Flask converter), so no
    path traversal is possible. Missing tiles return 404 - the frontend then
    shows a clean "Map data unavailable" fallback rather than a blank map.
    No external tile server is ever contacted (fully offline).
    """
    tile = MAP_TILES_DIR / str(z) / str(x) / f"{y}.png"
    if not tile.is_file():
        abort(404)
    return send_file(str(tile), mimetype="image/png")


@app.route("/api/map/available", methods=["GET"])
def api_map_available():
    """Whether a local offline tile pack is present (drives the map fallback)."""
    available = False
    try:
        for z in MAP_TILES_DIR.iterdir():
            if z.is_dir() and next(z.rglob("*.png"), None) is not None:
                available = True
                break
    except OSError:
        available = False
    return jsonify({"available": available, "tiles_dir": str(MAP_TILES_DIR)})


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
    """
    Analyze one image (multipart 'image'); returns JSON. Used by the
    manual fallback and by any external client. The automatic DJI path
    does NOT go through here - it uses the watch folder - but both share
    the same InspectionService underneath.
    """
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "no image provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    captured_at = request.form.get("captured_at") or request.headers.get("X-Captured-At")
    ext = Path(secure_filename(file.filename)).suffix.lower()
    tmp_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(tmp_path)
    try:
        record = get_service().analyze_file(str(tmp_path), source="upload", captured_at=captured_at)
    except InvalidImageError:
        return jsonify({"error": "invalid or corrupt image"}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"analysis failed: {exc}"}), 500
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    return jsonify(_record_payload(record))


# ---------------------------------------------------------------------------
# Live capture service: the production photo path.
#
# The dashboard's "Capture Photo" button calls this. It grabs the CURRENT
# MediaMTX frame (no VLC, no desktop screenshot, no continuous video decode),
# saves it as a clean JPEG in the operator's DJI pictures folder, associates
# the latest usable phone GPS, runs best.pt once, and creates one inspection -
# all with no manual upload and no separate Analyze step.
# ---------------------------------------------------------------------------
@app.route("/api/capture", methods=["POST"])
def api_capture():
    """Capture the current live frame -> GPS -> best.pt -> inspection record."""
    captured_at = request.form.get("captured_at") or request.headers.get("X-Captured-At")
    try:
        result = get_capture_service().capture(captured_at=captured_at)
    except CaptureError as exc:
        # A temporary stream/network failure is a normal condition, not a
        # crash: return a clean, human-readable error the UI can show without
        # exposing a stack trace. MediaMTX and inspection history are untouched.
        log.warning("[CAPTURE] %s", exc)
        return jsonify({"ok": False, "error": "capture_failed", "detail": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001 - last-resort guard; never 500 the whole app
        log.error("[CAPTURE] unexpected error: %s", exc)
        return jsonify({"ok": False, "error": "capture_failed", "detail": str(exc)}), 503

    record = result["record"]
    payload = _record_payload(record)
    payload["ok"] = True
    payload["image_path"] = result["image_path"]
    payload["captured_at"] = result["captured_at"]
    log.info("[MATANGLAWIN] Capture -> %s (inspection #%s)", record.status, record.id)
    return jsonify(payload), 201


@app.route("/api/capture/status", methods=["GET"])
def api_capture_status():
    """Internal capture/stream-health snapshot (debug only; not an operator panel)."""
    try:
        return jsonify(get_capture_service().status())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 200


# ---------------------------------------------------------------------------
# Automatic DJI photo-transfer bridge endpoints.
#
# The bridge (dji_photo_bridge.py running on the phone/PC, or any folder-sync
# tool paired with a re-poster) delivers the ACTUAL original DJI still here.
# This is NOT a manual upload: no browser, no file picker, no Analyze button.
# It converges on the exact same InspectionService as everything else, and
# is de-duplicated by content hash so a retried upload is analyzed only once.
# ---------------------------------------------------------------------------
@app.route("/api/import", methods=["POST"])
def api_import():
    """Receive one original DJI photo from the transfer bridge and analyze it."""
    bridge_status.record_contact()

    # Accept either multipart ('image') or a raw image body (simplest for a
    # lightweight uploader). Either way we keep the ORIGINAL bytes untouched.
    filename = request.headers.get("X-Filename", "")
    file = request.files.get("image")
    if file is not None and file.filename != "":
        filename = filename or file.filename
        data = file.read()
    else:
        data = request.get_data(cache=False, as_text=False)

    if not data:
        return jsonify({"error": "no image data received"}), 400

    digest = hash_bytes(data)
    ledger = get_import_ledger()

    # De-dup: a retried/duplicate transfer of the SAME photo is accepted
    # (HTTP 200) but never analyzed a second time.
    if ledger.seen(digest):
        log.info("[PHOTO BRIDGE] Duplicate photo ignored (already analyzed): %s", filename or digest[:12])
        return jsonify({"status": "duplicate", "message": "photo already analyzed", "sha256": digest}), 200

    log.info("[PHOTO BRIDGE] New DJI photo received (%d bytes): %s", len(data), filename or "(unnamed)")

    ext = Path(secure_filename(filename)).suffix.lower() if filename else ""
    if ext not in ALLOWED_EXTS:
        ext = ".jpg"
    captured_at = request.headers.get("X-Captured-At") or request.args.get("captured_at")
    tmp_path = IMPORT_TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        tmp_path.write_bytes(data)
        log.info("[MATANGLAWIN] Inspection started")
        # The content hash is this capture's unique identity. Passing it as
        # capture_id makes "one capture -> one inspection" hold at the DB
        # level, and it is the SAME key the watch-folder transport uses, so
        # the same photo arriving by both routes can never create two records.
        record = get_service().analyze_file(
            str(tmp_path), source="import", captured_at=captured_at, capture_id=digest
        )
    except InvalidImageError:
        log.warning("[PHOTO BRIDGE] Rejected: not a readable image: %s", filename or digest[:12])
        return jsonify({"error": "invalid or corrupt image"}), 400
    except Exception as exc:  # noqa: BLE001
        log.error("[PHOTO BRIDGE] Analysis failed: %s", exc)
        return jsonify({"error": f"analysis failed: {exc}"}), 500
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    # Only mark as processed AFTER a successful analysis, so a transient
    # failure doesn't permanently blacklist a photo the bridge will retry.
    ledger.add(digest)
    bridge_status.record_photo(status=record.status)
    log.info("[MATANGLAWIN] %s (inspection #%s)", record.status, record.id)

    payload = _record_payload(record)
    payload["duplicate"] = False
    # SHA-256 of the EXACT bytes received and fed to analysis. Not a model
    # metric - a content-integrity/de-dup identity, and not shown in the UI.
    # It lets a caller prove the analyzed image is the original photo it sent.
    payload["sha256"] = digest
    return jsonify(payload), 201


@app.route("/api/bridge/ping", methods=["POST", "GET"])
def api_bridge_ping():
    """Lightweight heartbeat from the transfer bridge (keeps status READY)."""
    bridge_status.record_contact()
    return jsonify({"ok": True})


@app.route("/api/bridge/status", methods=["GET"])
def api_bridge_status():
    """Current PHOTO BRIDGE liveness for the dashboard indicator."""
    snap = bridge_status.snapshot()
    snap["import_dir"] = str(IMPORT_DIR)
    return jsonify(snap)


@app.route("/api/inspection/<int:inspection_id>", methods=["GET"])
def api_inspection(inspection_id):
    record = get_db().get_inspection(inspection_id)
    if record is None:
        abort(404)
    return jsonify(_record_payload(record))


@app.route("/api/inspection/latest", methods=["GET"])
@app.route("/api/inspections/latest", methods=["GET"])  # alias
def api_inspection_latest():
    record = get_db().get_latest()
    if record is None:
        return jsonify(None)
    return jsonify(_record_payload(record))


@app.route("/api/inspections", methods=["GET"])
def api_inspections():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        limit = 50
    db = get_db()
    records = db.list_inspections(limit=limit)
    items = [_record_payload(r) for r in records]
    return jsonify({"count": db.count(), "inspections": items})


@app.route("/api/inspections/geo", methods=["GET"])
def api_inspections_geo():
    """
    Points for the offline inspection map: geolocated inspections only.
    Returns a compact point list plus image URLs for the marker popups.
    This endpoint keeps working even when the live stream is down, so map
    markers (and their history) survive a video/network drop.
    """
    db = get_db()
    records = db.list_inspections(limit=1000)
    points = []
    for r in records:
        if r.latitude is None or r.longitude is None or not r.gps_available:
            continue
        m = r.to_map()
        if m.get("radius_m") is None:
            m["radius_m"] = _telemetry_module.phone_gps_radius_m()  # fallback for older rows
        m["urls"] = _inspection_urls(r)
        points.append(m)
    return jsonify({"count": len(points), "points": points})


@app.route("/api/inspections/summary", methods=["GET"])
def api_inspections_summary():
    """
    Dashboard summary counters from the actual database:
        {"total": N, "cracks": C, "clear": N - C}
    Updates automatically as new inspections are created (the dashboard
    polls this). No model/debug metrics are exposed.
    """
    return jsonify(get_db().counts())


# ===========================================================================
# Aircraft telemetry (offline DJI GPS geotagging)
# ===========================================================================
def _extract_gps(fields) -> tuple:
    """
    Pull (lat, lon, timestamp) out of a phone-GPS payload, ignoring every
    other field. Accepts the real Colota Google Play payload
    (lat, lon, acc, alt, vel, batt, bs, tst, bear, ...) as well as plain
    lat/lon/timestamp. `fields` is a mapping (JSON object or request.args).
    Colota uses `tst` for the timestamp; only lat/lon/timestamp are used.
    """
    def _get(*names):
        for n in names:
            if n in fields and fields.get(n) not in (None, ""):
                return fields.get(n)
        return None

    lat = _get("lat", "latitude")
    lon = _get("lon", "longitude")
    ts = _get("tst", "timestamp", "time", "ts")
    return lat, lon, ts


@app.route("/api/telemetry", methods=["POST", "GET"])
def api_telemetry():
    """
    Ingest one phone-GPS sample from the local collector (LAN only, e.g.
    Colota on the operator's phone). Accepts:

      * POST JSON object  - the full Colota payload
        {lat, lon, acc, alt, vel, batt, bs, tst, bear, ...};
      * POST JSON array   - a batch of such objects (the latest is used);
      * GET query params  - ?lat=..&lon=..&tst=.. .

    Only latitude/longitude/timestamp are extracted and stored - all other
    Colota fields (acc/alt/vel/batt/bs/bear) are ignored, never stored or
    exposed. Colota's `tst` maps to the timestamp; a missing/invalid
    timestamp is NOT an error (the arrival time is used). HTTP 400 is
    returned ONLY when the required location (lat/lon) is missing or invalid.
    Never crashes on bad input.
    """
    payload = request.get_json(silent=True)
    if isinstance(payload, list):                 # batch -> use the newest entry
        payload = payload[-1] if payload else {}
    if not isinstance(payload, dict) or not payload:
        payload = request.args                     # fall back to GET query params

    lat, lon, ts = _extract_gps(payload)
    sample = telemetry_store.add(
        latitude=lat, longitude=lon, timestamp=ts, source="phone_gps",
    )
    if sample is None:
        # Only genuinely invalid/missing LOCATION is a 400.
        return jsonify({"ok": False, "error": "missing or invalid lat/lon"}), 400

    ts_s = int(round(sample.timestamp_ms / 1000.0))
    log.info("[GPS] received lat=%s lon=%s timestamp=%s",
             sample.latitude, sample.longitude, ts_s)
    return jsonify({"ok": True, "buffered": telemetry_store.stats()["buffered"]})


@app.route("/api/telemetry/latest", methods=["GET"])
def api_telemetry_latest():
    """
    Latest GPS position (debug/verification). Returns the simple
    {lat, lon, timestamp} view Colota testing expects, plus the
    availability/staleness fields. Never exposes
    accuracy/altitude/speed/battery/heading.
    """
    latest, stale = telemetry_store.latest()
    stats = telemetry_store.stats()
    srt_mode = _srt_watcher.mode if _srt_watcher is not None else "idle"
    return jsonify({
        # simple debug view (Colota verification)
        "lat": latest.latitude if latest else None,
        "lon": latest.longitude if latest else None,
        "timestamp": int(round(latest.timestamp_ms / 1000.0)) if latest else None,
        # map/internal view
        "available": latest is not None,
        "stale": stale,
        "latest": latest.to_dict() if latest else None,
        "buffered": stats["buffered"],
        "received": stats["received"],
        "rejected": stats["rejected"],
        "srt_mode": srt_mode,
    })


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


@app.route("/api/mediamtx/config", methods=["GET"])
def api_mediamtx_config():
    """
    Download a ready-to-use mediamtx.yml generated from the CURRENT host
    IP / ports / stream key, so the operator never edits config by hand
    and it always matches whatever network the PC is on.
    """
    try:
        text = network_config.render_mediamtx_yaml()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    return Response(
        text,
        mimetype="text/yaml",
        headers={"Content-Disposition": "attachment; filename=mediamtx.yml"},
    )


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

    # Start watching for DJI SRT (Video Subtitles) telemetry. Independent of
    # the photo path and the live POV - it only feeds aircraft coordinates
    # and backfills inspections' locations when SRT files appear.
    try:
        get_srt_watcher().start()
        print(f"  Watching for DJI SRT telemetry in: {', '.join(SRT_DIRS) or '(none configured)'}")
    except Exception as exc:  # noqa: BLE001 - never block the app on telemetry
        print(f"  WARNING: SRT telemetry watcher failed to start: {exc}")

    cfg = detector_config()
    print(
        "  Detection config: "
        f"conf={cfg['conf']} imgsz={cfg['imgsz']} tiled={cfg['tiled']} tile={cfg['tile']} "
        f"overlap={cfg['tile_overlap']} enhance={cfg['enhance']} "
        f"clahe_clip={cfg['clahe_clip']} unsharp={cfg['unsharp']} augment={cfg['augment']} | "
        f"precision: max_thickness_frac={cfg['max_thickness_frac']} "
        f"min_area_px={cfg['min_area_px']} min_thinness={cfg['min_thinness']} "
        f"min_length_frac={cfg['min_length_frac']} refine={cfg['refine']}"
    )

    print("=" * 60)
    print("  MatanglaWIN - UAV Crack Inspection")
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
