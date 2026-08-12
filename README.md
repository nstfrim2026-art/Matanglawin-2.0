# Matanglawin-2.0

Photo-based crack inspection with a display-only live drone POV.

## What it does

MatanglaWIN analyzes **still photos** for surface cracks using a trained
YOLO11-seg model (`best.pt`). For each photo it reports **CRACK
DETECTED** or **NO CRACK DETECTED**, produces a red-highlighted copy of
the photo, and isolates each crack using its segmentation mask.

There are two ways a photo reaches the analyzer, and both go through the
**one** central pipeline (`inspection_service.py`):

1. **Manual upload** (`/`) - upload a photo in the browser (or POST to
   `/api/inspect`).
2. **DJI photo import** - the DJI Neo 2 controller takes a photo; the
   photo is transferred to this PC's watch folder; MatanglaWIN detects
   it and analyzes it automatically.

Separately, the dashboard (`/dashboard`) shows the **live drone POV** -
purely for viewing. That video is the raw MediaMTX WebRTC feed embedded
in the browser. **No AI runs on the live video**: no continuous
inference, no red masks, no boxes, no automatic frame capture. The user
watches the feed and decides when to take a photo with the controller.

```
                 DJI Neo 2
                    |
              live video (RTMP)
                    v
                 MediaMTX ------> WebRTC ------> Dashboard "Live Drone POV"
                                                 (display only, NO AI)

   controller shutter button
            |
      DJI photo file
            |
      transfer to PC  (USB / SD / DJI Assistant / phone sync)
            |
            v
     import watch folder  ---\
                              >--> inspection_service (YOLO11-seg, once)
     manual browser upload --/            |
                                 CRACK DETECTED / NO CRACK DETECTED
                                          |
                          original + red-highlighted + crack-only images
                                          |
                              SQLite database + PDF report
```

The live-video path and the photo-analysis path are **completely
independent**: MediaMTX being down never blocks uploads or imports, and
analysis never touches the RTSP stream. (This is a deliberate change
from an earlier design that ran YOLO continuously on RTSP frames and
produced `DESCRIBE 404` / OpenCV RTSP-timeout errors - that live
inference loop has been removed entirely.)

## How a captured DJI photo reaches MatanglaWIN

DJI Fly / the RC controller do **not** expose a documented, reliable
local API on Windows for pushing a freshly captured still directly into
a third-party app. So MatanglaWIN uses a **watch folder**: point it at a
local Windows directory and drop DJI photos there by any means. The app
notices new files, waits until each finishes copying, de-duplicates by
content hash, and analyzes each new photo exactly once.

The DJI-to-Windows transfer step (done by you, once per set of photos)
can be any of:

- Connect the controller/phone by USB and copy the JPGs into the watch
  folder, or
- Pop the microSD card into the PC and copy the photos, or
- Use DJI Assistant / the phone's file sync, then copy into the folder,
  or
- Any tool that lands the `.jpg` in the watch folder.

Set the folder with `MATANGLAWIN_IMPORT_DIR` (default:
`matanglawin_data/import/`). If a fully-automatic controller-to-PC push
is available in your specific setup, point it at that same folder and
imports become hands-free. **This app does not claim to talk to the
drone directly** - it only reacts to files appearing on disk.

## Result output (intentionally simple)

The result page (`/inspection/<id>`) and the dashboard's "Latest
Inspection" panel show:

- **Identification**: `CRACK DETECTED` or `NO CRACK DETECTED`
- **Original image** - the photo as captured
- **Crack highlighted** - the same photo with every detected crack
  highlighted in **red** (only shown when a crack is found)
- **Crack only** - each detected crack isolated using its segmentation
  mask, background suppressed (not a rectangular slab of surface)

No confidence percentages, probability scores, FPS, or model statistics
are shown anywhere in the UI. (A confidence value may exist internally
for model filtering and inside `metadata.json`, but it is never
surfaced.)

## Storage layout

Everything written at runtime lives under `matanglawin_data/` (created
next to `app.py`, or next to the executable in a packaged build; not
committed):

```
matanglawin_data/
    import/                     <- watch folder for DJI photos (configurable)
    upload_tmp/                 <- transient manual-upload staging
    inspections.db              <- SQLite inspection records
    reports/                    <- generated PDF reports
    inspections/
        <inspection-id>/
            original/photo.jpg
            highlighted/crack_highlighted.jpg   (copy of original if no crack)
            crack/crack_001.png, crack_002.png, ...
            metadata.json
```

## Dynamic network configuration - no hardcoded LAN IPs

The live POV needs the PC's current LAN IP so you know what RTMP address
to type into DJI Fly, and that IP changes with every network. **Nothing
in this codebase hardcodes a LAN IP.** `network_config.py` computes
everything at request time:

- `host_ip` - priority: (1) `MATANGLAWIN_HOST_IP` env override, else
  (2) auto-detect the active outbound interface (Windows/macOS/Linux, no
  admin rights, no extra deps), else (3) `null` (app keeps running).
- `rtmp_address` / `rtmp_url` - `rtmp://<host_ip>:1935[/<key>]`, what you
  type into DJI Fly. `null` when no LAN IP.
- `webrtc_url` - `http://localhost:8889/<key>`, the POV the browser
  plays. Always `localhost` (MediaMTX runs on the same PC).

`GET /api/network` returns these live; the dashboard polls it, so
switching Wi-Fi updates the displayed DJI RTMP address without a
restart or any code change.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `WEIGHTS` | Path to the `.pt` weights | `best.pt` |
| `HOST` / `PORT` | Web server bind host / preferred port | `127.0.0.1` / `5000` |
| `MATANGLAWIN_HOST_IP` | Force a specific LAN IP (e.g. `172.20.10.3`) | auto-detect |
| `MATANGLAWIN_STREAM_KEY` | MediaMTX stream path/key | `matanglawin` |
| `MATANGLAWIN_RTMP_PORT` / `RTSP_PORT` / `WEBRTC_PORT` | MediaMTX ports | `1935` / `8554` / `8889` |
| `MATANGLAWIN_API_PORT` | MediaMTX HTTP API port (for POV status) | `9997` |
| `MATANGLAWIN_IMPORT_DIR` | DJI photo watch folder | `<data>/import` |
| `MATANGLAWIN_GPS_LAT` / `LON` / `ALT` | Manual GPS fix (see `gps_provider.py`) | none |

## Live POV (MediaMTX) - optional, display only

The POV is optional; uploads and imports work without it. To use it,
download MediaMTX from
[bluenviron/mediamtx](https://github.com/bluenviron/mediamtx) and run it
(its default config accepts any RTMP path and mirrors it to RTSP/WebRTC
under the same path). Point DJI Fly at the RTMP address shown on the
dashboard, stream key `matanglawin`, and the browser will play the feed.

- MatanglaWIN never starts/stops/reconfigures MediaMTX; if it isn't
  running the dashboard just shows "MediaMTX: Not reachable" / POV
  `UNKNOWN`, and everything else keeps working.
- The dashboard's LIVE/OFFLINE/UNKNOWN badge comes from MediaMTX's own
  HTTP API (`/v3/paths/get/<key>`), a lightweight publisher check - it
  does **not** run YOLO or read frames.

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /api/inspect` | Upload one image (`image` field); returns the inspection result JSON. |
| `GET /api/inspection/<id>` | Inspection metadata (status, source, GPS, crack image names, image URLs). |
| `GET /api/inspection/latest` | Most recent inspection, or `null`. |
| `GET /api/inspections?limit=N` | Recent inspections, newest first. |
| `GET /api/inspection/<id>/original` | The original photo. |
| `GET /api/inspection/<id>/highlighted` | The red-highlighted photo. |
| `GET /api/inspection/<id>/crack/<filename>` | One crack-only image (path-traversal guarded). |
| `GET /api/network` | Dynamic host IP / DJI RTMP address / stream key / URLs. |
| `GET /api/mediamtx/status` | MediaMTX RTMP/RTSP/WebRTC port reachability. |
| `GET /api/stream/status` | Display-only POV publisher state: `LIVE`/`OFFLINE`/`UNKNOWN`. No YOLO. |
| `GET /api/gps` | Current GPS fix, or `available: false`. |
| `GET /reports/inspection/<id>.pdf` | PDF report for one inspection. |
| `GET /reports/full.pdf` | PDF report covering all inspections. |
| `GET /` , `/dashboard`, `/inspections`, `/inspection/<id>` | Pages. |

No API response exposes a confidence value.

## Running

### Option A: One-click executable (no Python needed to run it)

Bundles Python, the website, and `best.pt` into a single executable.

```bash
# Build once (needs Python + deps):
./build_exe.sh      # Linux/macOS
build_exe.bat       # Windows
```

Produces `dist/Matanglawin(.exe)` (~300-350 MB, embeds CPU PyTorch).
Double-click it; a console shows startup and your browser opens to the
app. Runtime data is written to `matanglawin_data/` next to the
executable. Build separately per OS (PyInstaller does not
cross-compile).

### Option B: Run from source

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# CPU-only torch is enough for single-image inference and much smaller:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Linux only, if "libGL.so.1: cannot open shared object file":
#   Debian/Ubuntu: sudo apt-get install -y libgl1
#   Fedora/Amazon Linux: sudo dnf install -y mesa-libGL

python app.py                       # set AUTO_OPEN=1 to open the browser automatically
```

Open http://localhost:5000, upload a photo, and view the result.

## Files

- `app.py` - Flask app: pages, `/api/*`, image serving, reports; starts the photo-import watcher in `main()`.
- `inspection_service.py` - **the** central analyze-one-photo pipeline (shared by upload and import).
- `photo_import.py` - watch-folder importer (stability + hash dedup + restart-safe), calls the service.
- `detector.py` - YOLO11-seg wrapper (`process_frame`), `build_union_mask`, `extract_crack_only`, `crack_overlay_crop`.
- `inference_core.py` - model loading + mask-overlay drawing (shared with `infer_overlay.py`).
- `infer_overlay.py` - original standalone CLI script (unchanged).
- `network_config.py` - LAN IP detection, DJI RTMP address, stream key, localhost RTSP/WebRTC URLs, MediaMTX reachability + POV publisher status.
- `gps_provider.py` - optional GPS fix (env override or pluggable source); never fabricates coordinates.
- `inspection_db.py` - SQLite inspection records (status/source/paths/GPS); auto-migrates an older DB.
- `report_generator.py` - status-based PDF reports (original + highlighted + crack images; no confidence).
- `templates/`, `static/` - upload page, result page, dashboard, inspections page, CSS/JS.
- `matanglawin.spec`, `build_exe.sh`, `build_exe.bat` - packaging.
- `tests/` - unit tests (see below).
- `matanglawin_data/` - runtime data (not committed).

## Hardware Test Procedure (DJI Neo 2)

Steps that require a real DJI Neo 2 / MediaMTX / network **have not been
executed by an AI agent** - see "What was verified" below.

1. Connect the Windows PC to Wi-Fi.
2. (Optional, for POV) Start MediaMTX.
3. Start MatanglaWIN (`python app.py` or the executable).
4. Open `/dashboard`.
5. Confirm **Live Drone POV** displays the drone feed (after pointing
   DJI Fly at the RTMP address shown in the Network panel, stream key
   `matanglawin`).
6. Confirm the live feed has **no** red boxes/masks/overlays.
7. Point the drone at a surface.
8. Press the DJI controller camera/shutter button to capture a photo.
9. Transfer that photo into the import folder (see "How a captured DJI
   photo reaches MatanglaWIN"); confirm it lands in
   `MATANGLAWIN_IMPORT_DIR`.
10. Confirm MatanglaWIN detects the new photo (it appears on
    `/inspections` / "Latest Inspection").
11. Confirm YOLO ran once on that photo (one new inspection record, not
    a stream of them).
12. Confirm the result says `CRACK DETECTED` or `NO CRACK DETECTED`.
13. Confirm the original image is available.
14. Confirm the red-highlighted image is available (when a crack was
    found).
15. Confirm the crack-only image(s) are available (when a crack was
    found).
16. Confirm the live POV is still clean (no AI overlay).
17. Capture and import another photo.
18. Confirm the previous photo is not re-processed (content-hash dedup).
19. Change Wi-Fi networks.
20. Confirm the dashboard's PC IP / DJI RTMP address update
    automatically, with no code change.

### What was verified in this environment vs. what still needs hardware

**Verified** (no DJI Neo 2, no MediaMTX, no real LAN in this sandbox;
YOLO stubbed with synthetic masks since the heavy model isn't installed
here):

- Photo analysis end-to-end (`inspection_service.py`): no-crack and
  crack cases, multiple cracks, mask-based crack-only extraction
  (background suppressed, not a rectangle), original preserved,
  red-highlighted image generated, `metadata.json` written, GPS attached
  when available / "Unavailable" (never fabricated) otherwise.
- Upload and import produce identical results via the same pipeline.
- Watch-folder importer (`photo_import.py`): partial-copy safety
  (size-stability), SHA-256 content dedup, restart persistence, corrupt
  files marked invalid without crashing, `source="import"` recorded.
- No live-video inference remains: `live_pipeline.py` / `video_source.py`
  / `capture_manager.py` are deleted, nothing imports them, and no
  runtime module uses `cv2.VideoCapture` (guarded by a test).
- Dynamic LAN IP: auto-detect, `MATANGLAWIN_HOST_IP` override (valid and
  invalid), no-network fallback; RTSP/WebRTC always `localhost`;
  `MATANGLAWIN_STREAM_KEY` honored; no hardcoded LAN IP anywhere
  (repo-wide audit + test).
- MediaMTX port reachability and POV publisher status degrade to
  "not reachable" / `UNKNOWN` without crashing when MediaMTX is absent.
- PDF reports (single/full, crack/no-crack, missing image, empty set),
  status-based, with no confidence shown.
- All Flask routes via the test client, including image serving,
  crack-image path-traversal protection, and that removed routes
  (`/stream/preview.jpg`, `/data/captures/*`) are gone.
- Full suite: **93 tests passing**.

**Still requires real hardware/software** (cannot be verified here):

- Real WebRTC POV playback from a running MediaMTX fed by DJI Fly.
- Real YOLO11-seg accuracy/latency on actual DJI photos from `best.pt`.
- The physical DJI-to-Windows photo transfer for your specific
  controller/OS setup, and whether any hands-free push is available.
- Real GPS/telemetry from the DJI Neo 2 (wire a real source into
  `gps_provider.set_gps_source()` if/when you have one).
- Behavior across real Wi-Fi network switches on the target Windows PC.

> Photo automatic transfer from the DJI controller is **not** claimed to
> work out of the box: MatanglaWIN analyzes whatever photo lands in the
> import folder. The transfer of the file from the controller to that
> folder is a documented manual step unless your setup provides an
> automatic local sync.
