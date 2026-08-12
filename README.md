# Matanglawin-2.0
Second Version

## Web App

A Flask website (`app.py`) with two complementary workflows:

1. **Upload Photo** (`/`) - upload a single photo of a surface (road,
   wall, pipe, etc.) and see the trained YOLO11-seg model (`best.pt`)
   highlight every crack it detects, using the same overlay logic as
   `infer_overlay.py` (exact mask contour, no bounding boxes), shared via
   `inference_core.py`.
2. **Live Drone Dashboard** (`/dashboard`) - the DJI Neo 2 workflow:
   DJI Fly publishes an RTMP stream to a locally-running **MediaMTX**
   instance, which serves it back out as RTSP (for the AI pipeline) and
   WebRTC (for the website's live "Drone POV"). The backend continuously
   pulls frames from the RTSP stream, runs the same YOLO11-seg model on
   each frame, and automatically captures a cropped image + GPS (when
   available) + timestamp for every valid crack detection into a local
   database, from which PDF inspection reports can be generated
   (`/inspections`).

Both workflows share the same trained model and the same underlying
mask-overlay drawing code - nothing about the model or its weights was
changed to add the live pipeline.

### Live drone architecture

```
DJI Neo 2
    |  DJI Fly
    |  RTMP  (rtmp://<this PC's LAN IP>:1935/matanglawin)
    v
MediaMTX
    |-----------------------------|
    v                             v
RTSP (rtsp://localhost:8554/...)  WebRTC (http://localhost:8889/...)
    |                             |
    v                             v
YOLO11-seg (detector.py)     MatanglaWIN website <dashboard>
    |                        (live drone POV, in-browser,
    v                         no OBS/VLC/LetsView/screen-mirroring)
Crack detection (detector.py)
    |
    v
Automatic capture (capture_manager.py)
    |-- cropped crack image
    |-- detection info (confidence, bbox, instance count)
    |-- timestamp
    |-- GPS (when available - gps_provider.py; never fabricated)
    v
Inspection database (inspection_db.py, SQLite)
    v
PDF inspection report (report_generator.py)
```

MediaMTX itself is **not** started, stopped, or reconfigured by this
app - it must already be running (see "Running MediaMTX" below). The
app only *reads* from it (RTSP frames for AI, and a reachability probe
for the dashboard's "MediaMTX reachable" indicator) and *tells you*
what RTMP address to type into DJI Fly.

### Dynamic network configuration - no hardcoded LAN IPs

This PC's LAN/Wi-Fi IP address changes every time it joins a different
network (home Wi-Fi, a different router, a mobile hotspot, etc.).
**Nothing in this codebase hardcodes a specific LAN IP.** All
network-facing values are computed at request time by
`network_config.py`:

- `host_ip` - this machine's current LAN IPv4 address. Priority order:
  1. `MATANGLAWIN_HOST_IP` environment variable, if set (manual override)
  2. Automatic detection of the OS's active outbound interface (works
     identically on Windows/macOS/Linux, no admin rights or extra
     dependencies required)
  3. If neither yields a usable address (no network connected),
     `host_ip` is `null` - the app keeps running, it just can't show a
     DJI RTMP address yet.
- `rtmp_address` / `rtmp_url` - `rtmp://<host_ip>:1935` and
  `rtmp://<host_ip>:1935/<stream_key>` - what you type into DJI Fly.
  `null` when `host_ip` is `null`.
- `rtsp_url` / `webrtc_url` - **always** `rtsp://localhost:8554/<stream_key>`
  and `http://localhost:8889/<stream_key>`, because MediaMTX and this
  app run on the same PC - the AI pipeline never needs to cross the
  network, only DJI Fly does.

Query the live values at any time via `GET /api/network` (see below),
or watch them update automatically on the dashboard - the frontend
polls this endpoint every few seconds, so switching Wi-Fi networks
updates the displayed DJI RTMP address without restarting the app.

Optional environment variables:

- `MATANGLAWIN_HOST_IP` - force a specific LAN IP instead of
  auto-detecting one (e.g. `172.20.10.3`). Useful when auto-detection
  picks the wrong interface on a multi-homed machine.
- `MATANGLAWIN_STREAM_KEY` - MediaMTX stream path/key (default:
  `matanglawin`).
- `MATANGLAWIN_RTMP_PORT` / `MATANGLAWIN_RTSP_PORT` /
  `MATANGLAWIN_WEBRTC_PORT` - override MediaMTX's default ports
  (1935 / 8554 / 8889) if your MediaMTX config uses different ones.
- `MATANGLAWIN_GPS_LAT` / `MATANGLAWIN_GPS_LON` / `MATANGLAWIN_GPS_ALT` -
  manually supply a fixed GPS fix (e.g. for a stationary inspection
  site, or to test the GPS-present code path without hardware). See
  `gps_provider.py` for how a future real GPS/telemetry source can be
  plugged in instead (`set_gps_source()`).

### Running MediaMTX

Download MediaMTX for your OS from
[bluenviron/mediamtx](https://github.com/bluenviron/mediamtx) and run
it with a config exposing the stream path `matanglawin` (the default
MediaMTX config works out of the box - it accepts any RTMP path and
mirrors it to RTSP/WebRTC under the same path). Start it **once**;
this app never starts a second instance, and will simply report
"MediaMTX: Not reachable" on the dashboard instead of erroring if it
isn't running.

### API endpoints (live pipeline)

| Endpoint | Description |
|---|---|
| `GET /api/network` | Current `host_ip`, `rtmp_address`, `rtmp_url`, `stream_key`, `rtsp_url`, `webrtc_url`. `host_ip: null` if no network is detected - never crashes. |
| `GET /api/mediamtx/status` | Best-effort TCP reachability of MediaMTX's RTMP/RTSP/WebRTC ports (does not start/stop MediaMTX). |
| `GET /api/stream/status` | `stream_state` (`OFFLINE`/`CONNECTING`/`LIVE`), detector readiness, frame/capture counters. |
| `GET /api/gps` | Current GPS fix, or `available: false` if unavailable. |
| `GET /api/inspections?limit=N` | Recent inspection records, newest first. |
| `GET /api/inspections/latest` | The most recent inspection record, or `null`. |
| `GET /reports/inspection/<id>.pdf` | PDF report for one inspection. |
| `GET /reports/full.pdf` | PDF report covering every recorded inspection. |
| `GET /dashboard` | Live drone inspection dashboard (POV, detection status, latest capture, network panel). |
| `GET /inspections` | Inspection history table with per-row PDF download links. |

### Option A: One-click executable (no Python required to *run* it)

This bundles Python, the website, and `best.pt` into a single
executable. Anyone can double-click it and use the site &mdash; they
don't need Python, pip, or any dependencies installed.

**Build it once** (this step does need Python + the packages below):

```bash
# Linux/macOS
./build_exe.sh

# Windows
build_exe.bat
```

This creates `dist/Matanglawin` (or `dist/Matanglawin.exe` on Windows),
a single file around 300-350&nbsp;MB (it embeds a CPU build of PyTorch).

**Run it:** double-click `dist/Matanglawin(.exe)`. A console window
opens showing server startup, and your default browser opens
automatically to the site. Uploaded images and results are saved next
to the executable in a `matanglawin_data/` folder. Closing the console
window stops the app.

> Note: build the executable separately on each OS you want to support
> (a Linux build only runs on Linux, a Windows build only runs on
> Windows, etc.) &mdash; PyInstaller does not cross-compile.

### Option B: Run from source with Python

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Recommended: install the CPU-only build of torch first (much smaller
# than the default CUDA build, and all that's needed for single-image
# inference):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# Linux only, if you hit "libGL.so.1: cannot open shared object file":
#   Debian/Ubuntu: sudo apt-get install -y libgl1
#   Fedora/Amazon Linux: sudo dnf install -y mesa-libGL

python app.py
```

Then open http://localhost:5000 in your browser (set `AUTO_OPEN=1` to
have it open automatically), upload an image, and click **Detect
Cracks**. You can tweak confidence threshold, mask opacity, and outline
thickness under "Advanced options" before detecting.

Optional environment variables:

- `WEIGHTS` - path to a different `.pt` weights file (default: `best.pt`)
- `HOST` - host to bind to (default: `127.0.0.1`)
- `PORT` - preferred port to listen on; if busy, a free one is chosen automatically (default: `5000`)
- `AUTO_OPEN` - set to `1` to auto-open the browser in dev mode too

### Files

- `app.py` - Flask web app: upload form + `/detect` (single-image), `/dashboard` + `/inspections` (live pipeline), `/api/*` endpoints, `/health` check, auto-opens browser when run as the packaged executable
- `inference_core.py` - shared inference + overlay-drawing logic (used by both the upload flow and the live-frame detector)
- `infer_overlay.py` - original standalone CLI script (unchanged)
- `network_config.py` - single source of truth for LAN IP detection, DJI RTMP address, stream key, localhost RTSP/WebRTC URLs, MediaMTX reachability
- `gps_provider.py` - optional GPS fix lookup (env override or pluggable future source); never fabricates coordinates
- `detector.py` - per-frame YOLO11-seg wrapper + crack validation/cropping for the live pipeline
- `video_source.py` - background RTSP reader with OFFLINE/CONNECTING/LIVE state, resilient to MediaMTX/drone disconnects
- `capture_manager.py` - automatic cropped-capture orchestration with cooldown/debounce + GPS attachment
- `inspection_db.py` - SQLite-backed inspection record storage
- `report_generator.py` - PDF inspection report generation (single crack / full report), GPS-optional
- `live_pipeline.py` - wires video_source + detector + capture_manager together into one background pipeline
- `templates/`, `static/` - website HTML/CSS/JS (upload page, result page, live dashboard, inspections page)
- `matanglawin.spec` - PyInstaller build configuration for the one-click executable
- `build_exe.sh` / `build_exe.bat` - one-command build scripts (Linux/macOS and Windows)
- `tests/` - unit tests for network config, GPS provider, capture cooldown/dedupe, and PDF generation
- `matanglawin_data/` (created at runtime, not committed) - uploaded images, detection results, live captures, inspection database, generated PDF reports

## Hardware Test Procedure (DJI Neo 2)

This is the exact sequence to validate the live pipeline against real
hardware. Steps 1-9 and 13 require a real DJI Neo 2 + MediaMTX +
network switch and **have not been executed by an AI agent** - only
the software listed in "What was actually verified" below has been.

1. Connect the PC to Wi-Fi (or Ethernet).
2. Start MediaMTX (`mediamtx` binary, default config is sufficient).
3. Start MatanglaWIN (`python app.py`, or the packaged executable) and
   open `/dashboard` in a browser.
4. Check the displayed **PC IP** in the Network panel (or `GET
   /api/network`). Example: `172.20.10.3`.
5. In DJI Fly, configure:
   - RTMP Address: `rtmp://<PC IP from step 4>:1935`
   - Stream Key: `matanglawin` (or your `MATANGLAWIN_STREAM_KEY`)
6. Start DJI live transmission.
7. Verify MediaMTX's own logs report the stream as available/online,
   and the dashboard's "MediaMTX reachable" indicator is green.
8. Verify the dashboard's **Drone Stream** badge changes from `OFFLINE`
   to `LIVE`, and the live drone POV video appears (served directly by
   MediaMTX's WebRTC endpoint in an iframe - no OBS/VLC/LetsView).
9. Confirm (e.g. via MediaMTX logs or a `ffprobe`/VLC test connection)
   that the AI pipeline is reading `rtsp://localhost:8554/matanglawin`.
10. Point the drone camera at a crack (or any surface with a similar
    known-good test image from the training set).
11. Verify the dashboard's "Crack Detection" panel shows a
    non-`OFFLINE` detector status and rising "Frames processed".
12. Verify "Latest Capture" updates with a cropped crack image and
    confidence percentage shortly after step 10.
13. Verify a new row appears on `/inspections` with a timestamp and
    (if the drone/telemetry provides it, or `MATANGLAWIN_GPS_LAT/LON`
    is set) GPS coordinates; otherwise it correctly shows "Unavailable".
14. Confirm GPS shows "Unavailable" (not a crash, not fabricated
    coordinates) when no GPS source is configured.
15. Click "Download PDF report" (single) and "Download full PDF
    report" on `/inspections`, and confirm both open as valid PDFs
    containing the timestamp, confidence, image, and GPS line.

### Network-switch test

After the above, disconnect from the current network and connect to a
different one (different Wi-Fi, mobile hotspot, or Ethernet). Confirm,
**without restarting MatanglaWIN**, that the dashboard's PC IP and DJI
RTMP address update to the new network's address within a few seconds
(the frontend polls `GET /api/network` periodically). Only the RTMP
address typed into DJI Fly needs to change on the new network - the
stream key, RTSP URL, and WebRTC URL stay the same.

### What was actually verified (this change) vs. what still needs real hardware

**Verified in this environment** (no DJI Neo 2, no MediaMTX instance,
no real LAN available in this sandbox):

- LAN IP auto-detection logic, `MATANGLAWIN_HOST_IP` override (valid
  and invalid), and the no-network fallback (`network_config.py`)
- Dynamic RTMP/RTSP/WebRTC URL generation, and that RTSP/WebRTC always
  use `localhost` regardless of the detected/overridden LAN IP
- `MATANGLAWIN_STREAM_KEY` override
- MediaMTX port-reachability probing against a real closed port
  (correctly reports `Not reachable` without crashing or retrying
  forever)
- RTSP video source state machine (OFFLINE / CONNECTING / LIVE) against
  both a real (absent) RTSP endpoint and a mocked `cv2.VideoCapture`
  simulating a live stream and a mid-stream disconnect
- Per-frame crack detection, minimum-area filtering, and crop-with-margin
  logic (`detector.py`) using a mocked YOLO result (no real YOLO
  installed in the test environment)
- Capture cooldown/debounce, low-confidence rejection, and GPS
  attachment (present and unavailable) end-to-end into a real SQLite
  database (`capture_manager.py`, `inspection_db.py`)
- PDF generation for single/full reports, with and without GPS, with a
  missing image, and with zero inspections (`report_generator.py`)
- All new Flask routes (`/api/network`, `/api/mediamtx/status`,
  `/api/stream/status`, `/api/gps`, `/api/inspections*`, `/dashboard`,
  `/inspections`, `/reports/*`) via Flask's test client
- The original single-image upload flow (`/`, `/detect`, `/health`)
  still compiles and its route wiring is unchanged

**Still requires real hardware testing** (cannot be verified without a
physical DJI Neo 2, a running MediaMTX instance, and a real network):

- Actual DJI Fly -> MediaMTX RTMP publish and MediaMTX's real
  "stream is available and online" / disconnect log lines
- Real WebRTC playback in a browser from MediaMTX's `/matanglawin`
  WebRTC endpoint
- Real YOLO11-seg inference on live drone frames from `best.pt`
  (correctness/latency on real crack imagery, not synthetic masks)
- End-to-end automatic capture while physically flying over a real
  crack
- Real GPS/telemetry availability from the DJI Neo 2 (this app does
  not assume any particular DJI telemetry format is available; if you
  have a way to feed real telemetry, wire it in via
  `gps_provider.set_gps_source()`)
- Behavior when switching real Wi-Fi networks/hotspots on the actual
  target Windows PC
