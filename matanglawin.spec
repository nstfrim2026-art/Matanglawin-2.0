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

hiddenimports = []
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("cv2")

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
