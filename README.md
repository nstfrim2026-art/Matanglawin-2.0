# MatanglaWIN 2.0

Professional UAV structural crack-inspection system for the DJI Neo 2.

MatanglaWIN keeps a **clean live drone point-of-view** for monitoring and
runs **automatic crack segmentation on each captured still photo** — press
the shutter on the controller and the photo is transferred, analyzed with
the trained YOLO11-seg model (`best.pt`), and shown on the dashboard with
no manual upload and no Analyze button.

The web UI is a wide, desktop-first dark dashboard that preserves the
established MatanglaWIN look (dark navy surface, red accent, panelled
cards) rather than a narrow mobile stack.

---

## Two independent pipelines

```
Pipeline A — LIVE DRONE POV (display only)
    DJI Neo 2 → DJI Fly → RTMP → MediaMTX → MatanglaWIN → clean live POV
    (no YOLO, no masks, no boxes, no auto-capture — monitoring only)

Pipeline B — DJI STILL PHOTO INSPECTION (automatic)
    DJI Neo 2 → press PHOTO → automatic transfer → watch folder
    → InspectionService → best.pt / YOLO segmentation
    → CRACK DETECTED / NO CRACK DETECTED + original + red-highlighted
    → automatic website update → inspection history
```

The two are fully independent: the live stream being down never blocks
photo inspection, and running an inspection never interrupts the live POV.

---

## Logo

Place the official logo at:

```
static/img/matanglawin-logo.png
```

The header, favicon, and branding load that PNG automatically and fall
back to the bundled `static/img/matanglawin-logo.svg` placeholder only if
the PNG is missing. The PNG's aspect ratio is preserved (it is scaled by
height in the header). No other logo is generated or substituted.

---

## Live drone POV (Pipeline A) — MediaMTX

1. **Run MediaMTX** (a standalone binary, download from its releases page)
   on this PC. From the running app you can download a ready-made config:

   ```
   http://localhost:5000/api/mediamtx/config    →  mediamtx.yml
   ```

   Then start it with `mediamtx mediamtx.yml`. The config is generated from
   the current network settings, so it always matches your ports/stream key.

2. **Point DJI Fly at this PC.** Open the dashboard — the *Live Feed
   Connection* panel shows the exact values to enter in DJI Fly's
   live-stream (RTMP) setting:

   - **DJI RTMP address**, e.g. `rtmp://<your-PC-IP>:1935`
   - **Stream key**, e.g. `matanglawin`

   These follow the PC's current network automatically (see *Dynamic IP*).

3. The dashboard embeds the MediaMTX **WebRTC** feed directly in the
   browser. The backend never reads or annotates the video — the POV is
   always clean.

---

## Automatic DJI photo inspection (Pipeline B)

When the operator presses the shutter, the DJI Neo 2 saves a still photo.
Transfer that photo to this PC's **import folder** and MatanglaWIN analyzes
it automatically. There is no browse/select/drag/Upload/Analyze step.

Set the import folder (defaults to `matanglawin_data/import` next to the
app):

```
MATANGLAWIN_IMPORT_DIR = C:\path\to\your\import\folder
```

Point your DJI-to-PC transfer at that folder. Any of these work — the
watch folder reacts to files arriving regardless of the tool:

- DJI Fly / DJI Assistant download destination,
- a phone auto-sync (e.g. the album folder) synced to the PC,
- USB/SD copy into the folder,
- a cloud-drive folder that syncs to the PC.

The importer is **partial-copy safe** (waits until a file finishes
copying), **de-duplicates** by content hash (the same photo is never
analyzed twice, even after a restart), and **tolerates bad files** (a
corrupt image is skipped without stopping the watcher).

The dashboard's *Latest Inspection* panel updates on its own when a new
photo is analyzed — no page refresh needed.

---

## What a result shows (and never shows)

Each inspection shows only:

- **CRACK DETECTED** or **NO CRACK DETECTED**
- the **original** untouched photo
- a **red-highlighted** photo (segmentation masks painted red) — only when
  a crack is found

It never shows confidence/score, FPS, IoU, precision/recall, frame counts,
thresholds, bounding boxes, or any crack-only / cropped / separate-mask
image.

---

## Dynamic IP (works on any network)

The PC's LAN IP is never hardcoded. On every request the app either uses
your override or auto-detects the active LAN IPv4:

```
MATANGLAWIN_HOST_IP = 172.20.10.3     # optional manual override; wins if set
```

If unset, the current Wi-Fi/Ethernet/hotspot IP is detected automatically,
so everything keeps working when you switch networks. Consumer-side URLs
(RTSP/WebRTC) always use `localhost`; only the DJI-facing RTMP address uses
the LAN IP.

Other optional variables: `MATANGLAWIN_STREAM_KEY`, `MATANGLAWIN_RTMP_PORT`,
`MATANGLAWIN_RTSP_PORT`, `MATANGLAWIN_WEBRTC_PORT`, `MATANGLAWIN_API_PORT`.

---

## Manual upload (fallback / testing only)

The **Manual Upload** page (`/upload`) is a fallback for testing or when no
drone transfer is available. It runs the exact same `InspectionService` as
the automatic path. DJI photos never need it.

---

## Run from source

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Recommended: CPU-only torch (much smaller than the CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Linux only, if you hit "libGL.so.1: cannot open shared object file":
#   Debian/Ubuntu: sudo apt-get install -y libgl1

python app.py                       # open http://localhost:5000
```

Optional: `HOST`, `PORT`, `WEIGHTS`, `AUTO_OPEN=1` (auto-open browser).

## One-click executable

```bash
./build_exe.sh        # Linux/macOS
build_exe.bat         # Windows
```

Produces `dist/MatanglaWIN(.exe)`. It bundles the site and `best.pt`;
runtime data (photos, results, DB) is written to a `matanglawin_data/`
folder next to the executable. Build separately per OS (PyInstaller does
not cross-compile).

---

## Tests

```bash
pip install pytest
pytest -q
```

Covers dynamic IP config + override, no hardcoded IPs, no live-video
inference / no live YOLO, automatic photo ingestion, partial-copy safety,
duplicate protection, invalid-file handling, original preservation,
red-highlighted output, CRACK/NO-CRACK status, no confidence/box/crack-only
in the user-facing output, manual-upload fallback, and the app routes.

---

## Files

- `app.py` — Flask app: dashboard, result, inspections, manual-upload
  fallback, and JSON/image APIs.
- `inspection_service.py` — the single central analysis pipeline (manual +
  automatic converge here).
- `photo_import.py` — automatic DJI photo watch-folder bridge.
- `detector.py` / `inference_core.py` — YOLO11-seg segmentation + red
  overlay drawing (shared, no duplication).
- `inspection_db.py` — SQLite inspection history.
- `network_config.py` — dynamic IP + MediaMTX config generation.
- `infer_overlay.py` — original standalone CLI (unchanged behavior).
- `templates/`, `static/` — the desktop-first dark UI.
- `tests/` — the full test suite.
- `matanglawin.spec`, `build_exe.sh`, `build_exe.bat` — one-click build.
