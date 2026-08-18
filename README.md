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

When the operator presses the shutter, the DJI Neo 2 saves the original
still. MatanglaWIN then receives and analyzes that **exact original JPEG**
automatically — no browse/select/drag/Upload/Analyze step, ever.

### Honest note on what DJI provides

DJI Fly / the DJI Neo 2 do **not** expose a webhook or a documented local
API that pushes a freshly captured photo into a third-party app. After the
shutter, the original ends up in a phone/PC **album folder**. So the
"phone → Windows" hop is completed by one of the two bridges below. Both
deliver the original bytes and feed the **same** `InspectionService`; both
are de-duplicated so a photo is analyzed only once.

### Option 1 — Companion uploader (`dji_photo_bridge.py`) — direct LAN push

A small stdlib-only watcher that monitors the DJI album folder and POSTs
each new original to MatanglaWIN's `POST /api/import`. Run it on Android
via **Termux** (watch the DJI album, target the PC's LAN address), or on
the PC itself watching a folder that a sync app mirrors from the phone.

```
python dji_photo_bridge.py --server http://<PC-LAN-IP>:5000 --watch-dir "<DJI album folder>"
```

Environment fallbacks: `MATANGLAWIN_BRIDGE_SERVER`,
`MATANGLAWIN_BRIDGE_WATCH_DIR`, `MATANGLAWIN_BRIDGE_INTERVAL`. The PC address
is **configured here, never hardcoded** — change Wi-Fi and just point
`--server` at the PC's current LAN IP (shown on the dashboard). The uploader:

- waits until each file finishes downloading/copying (size stable),
- uploads the **raw original bytes** (no quality loss, no re-encode),
- **retries with backoff** when the PC is temporarily unavailable,
- **never uploads the same photo twice** (SHA-256, persisted across restarts),
- sends a heartbeat so the dashboard shows the bridge as connected,
- requires no manual action after setup — you only press the shutter.

### Option 2 — Folder sync into the watch folder — zero extra code

Point any sync tool at the import folder MatanglaWIN already watches:

```
MATANGLAWIN_IMPORT_DIR = C:\path\to\your\import\folder   (default: matanglawin_data\import)
```

Then mirror the phone's DJI album into that folder with, e.g.:

- **Syncthing** (LAN-only, no cloud — recommended), or
- a cloud client (Google Photos/Drive, OneDrive, Dropbox), or
- DJI Assistant / QuickTransfer download destination, or USB/SD copy.

The watch-folder importer is **partial-copy safe**, **de-duplicates** by
content hash (survives restarts), and **tolerates corrupt files** (skips
them without stopping).

> Prerequisite for both options: in DJI Fly, enable saving/downloading the
> full-resolution **original** photo to the phone album (not just a cache
> thumbnail), so a real original reaches the folder.

Either way, the dashboard's *Latest Inspection* panel updates on its own —
no page refresh, no Analyze button.

### Photo bridge status + logging

The dashboard shows a small **PHOTO BRIDGE** indicator (`READY` /
`WAITING FOR PHOTO` / `OFFLINE`) driven by `GET /api/bridge/status` from the
companion uploader's heartbeat. Both the app and the uploader log the flow:

```
[PHOTO BRIDGE] New DJI photo detected: DJI_0001.JPG
[PHOTO BRIDGE] Upload complete
[MATANGLAWIN] Inspection started
[MATANGLAWIN] CRACK DETECTED (inspection #1)
[PHOTO BRIDGE] PC unavailable - will retry     (on failure)
```

No technical metrics appear in the user-facing inspection result.

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

## Tuning detection sensitivity (thin cracks — no retraining)

High-resolution DJI stills shrink ~6× when analyzed at a small inference
size, so a hairline crack can drop below one pixel and vanish before the
model ever sees it. To recover thin/faint cracks **without retraining**,
analysis runs with recall-tuned defaults and **tiled (sliced) inference**:
the photo is cut into overlapping tiles, each analyzed at near-native
resolution, and the masks are stitched back together. A light,
detection-only contrast boost (CLAHE + unsharp) helps faint cracks stand
out — applied only to the model's input, never to the stored/displayed
original, and never to the live POV.

Detection is balanced between **recall** (find the faint/thin crack) and
**precision** (don't paint wall texture, stains, or shadows). A real crack
is thin, long, and connected; typical false positives are short dashes or
compact blobs — so a shape filter keeps crack-like shapes and rejects the
rest. All knobs are environment variables (no code edits, nothing shown in
the UI):

```
# recall (find faint/thin cracks)
MATANGLAWIN_CONF=0.20           # detection threshold (lower = more sensitive)
MATANGLAWIN_IMGSZ=1280          # whole-image inference size
MATANGLAWIN_TILED=1             # tiled/sliced inference on/off (biggest recall win)
MATANGLAWIN_TILE=1024           # tile size in px (try 640 for the thinnest cracks)
MATANGLAWIN_TILE_OVERLAP=0.2    # tile overlap fraction
MATANGLAWIN_ENHANCE=1           # contrast-boost the analysis copy only
MATANGLAWIN_CLAHE_CLIP=1.5      # enhancement strength (higher = stronger, more texture)
MATANGLAWIN_UNSHARP=0           # add unsharp mask (more recall, more false edges)
MATANGLAWIN_AUGMENT=0           # test-time augmentation (extra recall, slower)

# precision (reject non-crack detections)
MATANGLAWIN_MAX_THICKNESS_FRAC=0.03  # strip regions wider than 3% of the short side
MATANGLAWIN_MIN_AREA_PX=60           # drop specks below this many mask pixels
MATANGLAWIN_MIN_THINNESS=3.0         # reject compact blobs (disk ~= 1.0; crack >> 1)
MATANGLAWIN_MIN_LENGTH_FRAC=0.05     # drop fragments shorter than 5% of the long side

# overlay fidelity
MATANGLAWIN_REFINE=1                  # tighten the red mask to the actual dark crack line
```

**`MATANGLAWIN_REFINE`** tightens the red overlay so it hugs the actual
crack instead of a fat band. Within each detection it keeps only the dark
thin structure (a black-hat / Otsu step), giving a clean line that follows
the crack. It is safeguarded: if a detection contains no clear dark
structure (e.g. a bright/low-contrast crack) the original mask is kept
unchanged, so a real detection is never erased. It only ever shrinks the
mask *within* an already-detected region and never touches the stored
original photo. Set to 0 to paint the raw segmentation masks instead.

The most important precision lever is **`MATANGLAWIN_MAX_THICKNESS_FRAC`**.
A crack is thin *everywhere*; the worst false positives are wide filled
regions the model paints over a dark beam, a shadowed sill, or a stain — and
these often connect to the real crack. A morphological opening removes
anything whose local thickness exceeds `max_thickness_frac * min(H, W)`, so
those wide regions are stripped **even when a thin crack is connected to
them** (the crack survives, the blob does not). Set it to 0 to disable, or
raise it to keep genuinely wide cracks / spalled areas.

Measured on the real `best.pt`:

- **Recall** — a faint ~1px crack in a 1600×2400 image: the old whole-image
  path (conf 0.25, imgsz 640) detected **0** crack pixels; tiling recovered
  it. Tiling is the decisive factor; `MATANGLAWIN_TILE=640` recovers the
  thinnest cracks best (more tiles/time).
- **Precision** — a textured wall with one real crack plus 9 stains/dashes:
  without the shape filter the model returned **10** detections (crack + 9
  false positives); with the filter it returned **1** — only the real crack.
- **Thickness suppression** — verified at the unit level: a thin crack drawn
  through a wide "beam" band keeps the crack while the band is stripped
  (>80% of the wide area removed). It targets exactly the filled beam/sill/
  stain regions seen in field photos.

Tuning guidance (use the **Manual Upload** page to A/B one photo quickly):

- Over-detecting (beams / sills / stains / texture marked red)? Lower
  `MATANGLAWIN_MAX_THICKNESS_FRAC` (e.g. 0.02) to strip wider regions, raise
  `MATANGLAWIN_CONF` (e.g. 0.30), raise `MATANGLAWIN_MIN_THINNESS` (4–5) and
  `MATANGLAWIN_MIN_LENGTH_FRAC` (0.08), lower `MATANGLAWIN_CLAHE_CLIP`.
- Missing faint/short cracks? Lower `MATANGLAWIN_CONF` (e.g. 0.12), lower
  `MATANGLAWIN_MIN_LENGTH_FRAC` (0.02) and `MATANGLAWIN_MIN_THINNESS` (2),
  raise `MATANGLAWIN_MAX_THICKNESS_FRAC` (e.g. 0.06) or set it to 0, set
  `MATANGLAWIN_TILE=640`, enable `MATANGLAWIN_UNSHARP=1`. Setting the three
  shape/thickness knobs to 0 disables that filtering entirely.

Note the default `MIN_LENGTH_FRAC=0.05` also drops genuine *short* secondary
cracks, and a very wide crack/spall may be trimmed by thickness suppression;
adjust those knobs if you need such cases. Tiling costs seconds per photo,
which is fine for **still-photo** analysis and never touches the live POV.
If cracks are still missed after tuning, that is when more training data
(native-resolution hairline crops) becomes the real answer.

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
inference / no live YOLO, automatic photo ingestion (watch folder **and**
`/api/import` bridge), partial-copy safety, duplicate protection across
retries, invalid-file handling, original preservation, red-highlighted
output, CRACK/NO-CRACK status, no confidence/box/crack-only in the
user-facing output, bridge status transitions, manual-upload fallback, and
the app routes.

---

## Real DJI Neo 2 hardware test

Prove that pressing PHOTO makes the actual original arrive and appear in
MatanglaWIN with **no manual upload**.

1. Start MediaMTX (for the live POV) and `python app.py`. Note the PC's LAN
   IP from the dashboard's *Live Feed Connection* panel.
2. In DJI Fly, enable saving the full-resolution **original** photo to the
   phone album, and configure the live-stream (RTMP) values shown on the
   dashboard.
3. Start the bridge:
   - Companion uploader: `python dji_photo_bridge.py --server http://<PC-LAN-IP>:5000 --watch-dir "<DJI album>"` (Termux on the phone, or on the PC watching a synced folder), **or**
   - Folder sync: point Syncthing/cloud at `MATANGLAWIN_IMPORT_DIR`.
   Confirm the dashboard shows **PHOTO BRIDGE: READY / WAITING FOR PHOTO**.
4. Connect the DJI Neo 2, open DJI Fly, confirm the **live POV** shows clean
   video (no boxes/masks).
5. **Isolation check:** stop MediaMTX so there is no live stream, aim at a
   known crack, and press the **PHOTO/shutter button once**. Do not upload
   or copy anything.
6. Confirm, hands-off: the original lands in the watch folder / is POSTed,
   the log shows `[MATANGLAWIN] Inspection started` then `CRACK DETECTED` /
   `NO CRACK DETECTED`, and the dashboard's *Latest Inspection* updates by
   itself with the original + red-highlighted image.
7. **Prove it's the actual DJI still (not a stream frame):**
   - The companion uploader's `POST /api/import` response includes a
     `sha256` of the exact bytes sent. Compare it to the file on the phone/PC:
     `certutil -hashfile "DJI_0001.JPG" SHA256` (Windows) — they must match.
   - Confirm the stored original's resolution equals the DJI capture's
     resolution, and its timestamp/filename correspond to your capture.
   - Because MediaMTX was stopped in step 5, the result provably cannot have
     come from the live stream.
8. Restart MediaMTX; confirm the POV plays again, that merely streaming
   creates **no** inspection, and that pressing PHOTO still produces one
   while the POV keeps playing.
9. Press PHOTO twice, and also re-copy an already-processed file: confirm
   duplicates are **not** re-analyzed. Briefly stop the app mid-capture:
   confirm the uploader retries and the photo still arrives once the app is
   back.

Automated stand-in for this flow (no drone needed): the test suite exercises
the uploader → `/api/import` → analysis → status path, and a byte-identity
check confirms the exact original bytes reach the analysis input.

---

## Drone crack geotagging & spatial mapping (optional module)

An offline/Colab layer that turns pixel-space crack segmentations into
geographic **GeoJSON** so cracks can be mapped and measured. It reuses the
same `inference_core` model and does **not** change the live web UI, the POV
pipeline, or the automatic photo-import workflow. Spec:
`.kiro/specs/drone-crack-geotagging/`.

- `exif_metadata.py` — reads drone GPS (EXIF, DMS→decimal degrees), DJI XMP
  gimbal yaw/pitch and relative/absolute altitude, and capture timestamp.
  Degrades gracefully (never raises) when metadata is absent.
- `geotag_engine.py` — converts each crack polygon to **WGS84 (EPSG:4326)**
  via either an **orthomosaic GeoTIFF** transform (`rasterio`, most accurate)
  or a **Ground Sample Distance** transform from altitude + camera specs
  (nadir/flat-ground approximation), then exports a GeoJSON
  `FeatureCollection` with per-crack `crack_area_m2`, `length_m`,
  `confidence`, `image_name`, and `timestamp`.
- `geotagging_colab_demo.ipynb` — mount Google Drive, run segmentation +
  geotagging, and render an interactive **Folium/Leaflet** map inline.

```bash
pip install -r requirements-geo.txt     # exifread, pyproj, folium (+ rasterio for orthomosaics)
python -c "from geotag_engine import geotag_batch, write_geojson; \
           write_geojson(geotag_batch(['DJI_0001.JPG']), 'cracks.geojson')"
```

The geospatial extras are **optional**; the base app installs and runs
without them, and the geotagging core math is unit-tested with only
NumPy/OpenCV present. Confidence/area are exported in the GeoJSON (a GIS
artifact for engineers) while remaining hidden in the operator web UI.
Accuracy note: the GSD method assumes near-nadir capture over flat ground;
use an orthomosaic for survey-grade results.

## Files

- `app.py` — Flask app: dashboard, result, inspections, `/api/import`
  bridge ingest, bridge status, manual-upload fallback, JSON/image APIs.
- `inspection_service.py` — the single central analysis pipeline (manual +
  automatic converge here).
- `photo_import.py` — automatic DJI photo watch-folder bridge (Option 2).
- `dji_photo_bridge.py` — companion uploader (Option 1): watch DJI album →
  POST original to `/api/import`. Stdlib only; runs on Termux or the PC.
- `import_ledger.py` — SHA-256 de-dup for `/api/import` (retry-safe).
- `bridge_status.py` — PHOTO BRIDGE liveness (READY/WAITING/OFFLINE).
- `detector.py` / `inference_core.py` — YOLO11-seg segmentation + red
  overlay drawing (shared, no duplication). Includes tiled/sliced inference
  and detection-only contrast enhancement for thin-crack recall (see
  "Tuning detection sensitivity").
- `inspection_db.py` — SQLite inspection history.
- `network_config.py` — dynamic IP + MediaMTX config generation.
- `infer_overlay.py` — original standalone CLI (unchanged behavior).
- `templates/`, `static/` — the desktop-first dark UI.
- `tests/` — the full test suite.
- `matanglawin.spec`, `build_exe.sh`, `build_exe.bat` — one-click build.
