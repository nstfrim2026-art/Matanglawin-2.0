# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build spec for the Matanglawin crack-detection website.
#
# Build with:
#   pyinstaller matanglawin.spec
#
# The resulting one-file executable is written to dist/Matanglawin
# (or dist/Matanglawin.exe on Windows). Double-clicking it starts the
# local web server and opens your default browser to the app.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
ROOT = Path(".").resolve()

datas = [
    (str(ROOT / "templates"), "templates"),
    (str(ROOT / "static" / "style.css"), "static"),
    (str(ROOT / "best.pt"), "."),
]
# static/uploads and static/results are write targets at runtime, not
# bundled read-only resources - app.py creates a sibling writable
# "matanglawin_data" folder next to the executable instead.

datas += collect_data_files("ultralytics")
datas += collect_data_files("reportlab")

hiddenimports = []
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("reportlab")
# First-party modules - PyInstaller's static analysis normally finds
# these via app.py's imports, but they're listed explicitly since some
# import cv2/ultralytics lazily inside functions, which can confuse
# dependency analysis.
hiddenimports += [
    "network_config",
    "gps_provider",
    "detector",
    "inference_core",
    "inspection_db",
    "inspection_service",
    "photo_import",
    "report_generator",
]

a = Analysis(
    ["app.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "notebook",
        "IPython",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Matanglawin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
