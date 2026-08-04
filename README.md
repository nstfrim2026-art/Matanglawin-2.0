# Matanglawin-2.0
Second Version

## Web App

A small Flask website (`app.py`) that lets you upload a photo of a
surface (road, wall, pipe, etc.) in the browser and see the trained
YOLO11-seg model (`best.pt`) highlight every crack it detects, using the
same overlay logic as `infer_overlay.py` (exact mask contour, no
bounding boxes), refactored into `inference_core.py` so both the CLI
script and the website share it.

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

### Get a live public link (hosted website)

To put the site online with a shareable URL, deploy the included
`Dockerfile` to **Hugging Face Spaces** (free, no credit card). Full
step-by-step instructions are in [DEPLOY.md](DEPLOY.md). In short: create a
free Hugging Face account, create a **Docker** Space, upload the project
files, and it goes live at `https://<your-username>-matanglawin.hf.space`.

You can also run the container anywhere Docker runs:

```bash
docker build -t matanglawin .
docker run -p 7860:7860 matanglawin
# open http://localhost:7860
```

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

- `app.py` - Flask web app (upload form, `/detect` route, `/health` check, auto-opens browser when run as the packaged executable)
- `inference_core.py` - shared inference + overlay-drawing logic
- `infer_overlay.py` - original standalone CLI script (unchanged)
- `templates/`, `static/` - website HTML/CSS
- `matanglawin.spec` - PyInstaller build configuration for the one-click executable
- `build_exe.sh` / `build_exe.bat` - one-command build scripts (Linux/macOS and Windows)
- `matanglawin_data/` (created at runtime, not committed) - uploaded images and detection results
