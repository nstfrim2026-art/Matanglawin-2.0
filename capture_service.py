"""
capture_service.py - On-demand single-frame capture from the MediaMTX stream.

This is the production photo-capture path for MatanglaWIN. When the operator
presses **Capture Photo**, the browser calls ``POST /api/capture`` and this
service:

    1. Grabs the CURRENT frame from the MediaMTX live stream (RTSP, on
       localhost) at the exact trigger time - a single snapshot, NOT a
       continuous video decode and NOT any YOLO-on-video pipeline.
    2. Saves it as a clean JPEG into the operator's DJI pictures folder
       (default ``%USERPROFILE%\\Pictures\\DJI\\MatanglaWIN``).
    3. Hands the saved JPEG to the one central InspectionService, which
       associates the latest usable phone GPS and runs best.pt once.
    4. Returns the saved image path + capture timestamp + the inspection.

Design guarantees (see the finalization requirements):

  * VLC is NOT part of this path. The authoritative video source is MediaMTX;
    VLC may be used by a human for debugging, but a VLC crash cannot affect
    capture, GPS, the website, or inspection state.
  * The Windows desktop / browser chrome is never captured - only the clean
    drone frame that MediaMTX is serving.
  * No continuous OpenCV video reader: the default frame grabber shells out to
    ``ffmpeg`` for exactly one frame, so there is no long-lived decode loop and
    no live-video inference.
  * A temporary stream/network failure is a NORMAL condition, not a crash: the
    grab is retried with a short bounded backoff, and if it still fails a clean
    ``CaptureError`` is raised (the route turns it into a tidy JSON error).
    MediaMTX is never started, stopped, or reconfigured here.

The frame grabber is injectable so the whole flow can be unit-tested without a
real stream or ffmpeg installed.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("matanglawin.capture")

DEFAULT_GRAB_TIMEOUT_S = float(os.environ.get("MATANGLAWIN_CAPTURE_TIMEOUT", "8"))
DEFAULT_RETRIES = int(os.environ.get("MATANGLAWIN_CAPTURE_RETRIES", "3"))
DEFAULT_BACKOFF_S = float(os.environ.get("MATANGLAWIN_CAPTURE_BACKOFF", "0.4"))


def resolve_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    """
    Locate a usable ffmpeg executable, returning its absolute path or None.

    Resolution order (real testing showed ffmpeg is often installed but not on
    the MatanglaWIN process PATH - e.g. a WinGet package):

      1. ``explicit`` argument, else the ``MATANGLAWIN_FFMPEG`` env var (highest
         priority). Accepts a full path or a bare name resolved via PATH.
      2. ``shutil.which("ffmpeg")`` (on PATH).
      3. Sensible Windows locations, discovered by globbing (NO hardcoded
         username, NO hardcoded exact path):
           * WinGet packages under %LOCALAPPDATA% and %ProgramFiles%
             (e.g. Gyan.FFmpeg, BtbN builds),
           * common install dirs (C:\\ffmpeg\\bin, Program Files\\ffmpeg).

    The result is meant to be cached by the caller.
    """
    # 1) explicit / env override
    candidate = (explicit if explicit is not None
                 else os.environ.get("MATANGLAWIN_FFMPEG", "")).strip()
    if candidate:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
        found = shutil.which(candidate)
        if found:
            return found
        # An explicit-but-missing override should not silently fall through to
        # a different ffmpeg; treat it as unconfigured.
        return None

    # 2) on PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 3) Windows discovery (glob; expand env vars; never hardcode a username)
    patterns = []
    for base_env in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(base_env)
        if base:
            patterns.append(os.path.join(base, "Microsoft", "WinGet", "Packages",
                                         "*FFmpeg*", "**", "ffmpeg.exe"))
            patterns.append(os.path.join(base, "**", "ffmpeg.exe"))
    patterns += [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\ffmpeg.exe"]
    for pat in patterns:
        try:
            for hit in glob.glob(pat, recursive=True):
                if os.path.isfile(hit):
                    return os.path.abspath(hit)
        except OSError:
            continue
    return None


def default_pictures_dir() -> Path:
    """
    Where captured photos are written. Default is the DJI-style pictures
    folder in the user's profile (``~/Pictures/DJI/MatanglaWIN`` - on Windows
    ``~`` expands to ``%USERPROFILE%``). Override with
    ``MATANGLAWIN_PICTURES_DIR``. Never hardcodes a username.
    """
    raw = os.environ.get("MATANGLAWIN_PICTURES_DIR", "").strip()
    if raw:
        return Path(os.path.expanduser(os.path.expandvars(raw)))
    return Path(os.path.expanduser(os.path.join("~", "Pictures", "DJI", "MatanglaWIN")))


class CaptureError(Exception):
    """Raised when a live frame cannot be captured (clean, non-fatal)."""


class FfmpegNotConfiguredError(CaptureError):
    """Raised when no usable ffmpeg executable can be found (a config problem,
    distinct from a transient 'could not read the live stream')."""


def ffmpeg_grab(rtsp_url: str, out_path: str,
                ffmpeg_bin: Optional[str] = None,
                timeout_s: float = DEFAULT_GRAB_TIMEOUT_S) -> bool:
    """
    Grab a single current frame from ``rtsp_url`` into ``out_path`` (JPEG)
    using ffmpeg. Returns True on success. Never raises for the "ffmpeg is
    missing / stream unreachable" case - it returns False so the caller can
    retry and then surface a clean error. Does NOT touch MediaMTX.
    """
    ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg() or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-q:v", "2",
        "-y", str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, timeout=timeout_s,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning("[CAPTURE] ffmpeg not found (%s) - cannot grab frame", ffmpeg_bin)
        return False
    except subprocess.TimeoutExpired:
        log.warning("[CAPTURE] ffmpeg timed out reading %s", rtsp_url)
        return False
    except Exception as exc:  # noqa: BLE001 - a grab failure must never crash the app
        log.warning("[CAPTURE] ffmpeg grab error: %s", exc)
        return False
    ok = proc.returncode == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 0
    if not ok:
        log.warning("[CAPTURE] ffmpeg could not read a frame (rc=%s)", proc.returncode)
    return ok


class CaptureService:
    """
    On-demand capture of the current MediaMTX frame -> saved JPEG -> GPS ->
    best.pt -> inspection record. Thread-safe; keeps lightweight internal
    stream-health figures for debugging (never a big operator panel).
    """

    def __init__(
        self,
        inspection_service,
        rtsp_url_provider: Callable[[], Optional[str]],
        pictures_dir: Optional[str] = None,
        tmp_dir: Optional[str] = None,
        grabber: Callable[[str, str], bool] = None,
        ffmpeg_path: Optional[str] = None,
        retries: int = DEFAULT_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
    ):
        self.inspection_service = inspection_service
        self.rtsp_url_provider = rtsp_url_provider
        self.pictures_dir = Path(pictures_dir) if pictures_dir else default_pictures_dir()
        self.tmp_dir = Path(tmp_dir) if tmp_dir else (self.pictures_dir / ".tmp")
        # Resolve and cache the ffmpeg executable once (see resolve_ffmpeg).
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path is not None else resolve_ffmpeg()
        # A real (ffmpeg) capture requires a resolved executable; an injected
        # grabber (tests) does not.
        self._custom_grabber = grabber is not None
        self._grabber = grabber or (lambda url, out: ffmpeg_grab(url, out, ffmpeg_bin=self.ffmpeg_path))
        self.retries = max(1, int(retries))
        self.backoff_s = max(0.0, float(backoff_s))

        self._lock = threading.Lock()
        # Internal stream-health monitoring (requirement 4C). These exist for
        # debugging/verification; the operator UI only ever shows a clean state.
        self._last_success_ms: Optional[float] = None
        self._last_error: Optional[str] = None
        self._last_attempts = 0
        self._total_captures = 0

    # -- diagnostics / health snapshot (internal only) -----------------

    def ffmpeg_available(self) -> bool:
        return self._custom_grabber or bool(self.ffmpeg_path)

    def status(self) -> dict:
        """
        Internal capture/stream diagnostic (kept out of the operator UI):
        ffmpeg availability + resolved executable + the RTSP URL, plus the
        last capture health figures.
        """
        try:
            rtsp_url = self.rtsp_url_provider()
        except Exception:  # noqa: BLE001
            rtsp_url = None
        with self._lock:
            return {
                "ffmpeg_available": self.ffmpeg_available(),
                "ffmpeg_executable": self.ffmpeg_path,
                "rtsp_url": rtsp_url,
                "last_frame_ts": (self._last_success_ms / 1000.0) if self._last_success_ms else None,
                "last_error": self._last_error,
                "last_attempts": self._last_attempts,
                "total_captures": self._total_captures,
                "pictures_dir": str(self.pictures_dir),
            }

    # -- the capture operation ----------------------------------------

    def capture(self, captured_at: Optional[str] = None) -> dict:
        """
        Capture one current frame and turn it into an inspection.

        Returns a dict:
            {"image_path", "captured_at", "timestamp", "record"}
        where ``record`` is the created InspectionRecord.

        Raises CaptureError (clean, non-fatal) if the live stream cannot be
        read after the bounded retries. The website/server keep running and
        MediaMTX is left untouched.
        """
        # ffmpeg must be configured for a real capture (an injected grabber in
        # tests bypasses this). Reported as a clear CONFIG error, distinct from
        # a transient stream-read failure.
        if not self.ffmpeg_available():
            raise FfmpegNotConfiguredError(
                "FFmpeg not configured - set MATANGLAWIN_FFMPEG or install ffmpeg on PATH"
            )

        rtsp_url = None
        try:
            rtsp_url = self.rtsp_url_provider()
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"could not resolve live stream URL: {exc}")
        if not rtsp_url:
            raise CaptureError("live stream is not available (MediaMTX URL unknown)")

        # UTC, timezone-aware capture time so it is DIRECTLY comparable to the
        # phone-GPS (Colota) timestamps, which are UTC. A naive local time here
        # was the cause of the UTC+8 "Not recorded" bug: a +8h skew pushed the
        # capture outside every GPS sample's match/age window.
        trigger_dt = datetime.now(timezone.utc)
        trigger_ms = trigger_dt.timestamp() * 1000.0
        captured_iso = captured_at or trigger_dt.isoformat()

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.tmp_dir / f"grab_{uuid.uuid4().hex}.jpg"

        # Retry the grab with a short bounded backoff: a temporary stream/
        # network hiccup is a normal condition, not a fatal error.
        attempts = 0
        grabbed = False
        last_err = "stream unreachable"
        while attempts < self.retries and not grabbed:
            attempts += 1
            try:
                grabbed = bool(self._grabber(rtsp_url, str(tmp_path)))
            except Exception as exc:  # noqa: BLE001 - never let a grabber bug crash capture
                last_err = str(exc)
                grabbed = False
            if not grabbed and attempts < self.retries:
                time.sleep(self.backoff_s * attempts)

        if not grabbed:
            with self._lock:
                self._last_error = last_err
                self._last_attempts = attempts
            self._safe_unlink(tmp_path)
            raise CaptureError(
                "could not read the live stream - is the drone publishing to MediaMTX?"
            )

        # Persist the clean drone frame into the operator's DJI pictures folder
        # with a timestamped, collision-free name.
        self.pictures_dir.mkdir(parents=True, exist_ok=True)
        stamp = trigger_dt.strftime("%Y%m%d_%H%M%S")
        saved_path = self.pictures_dir / f"capture_{stamp}_{uuid.uuid4().hex[:6]}.jpg"
        try:
            shutil.move(str(tmp_path), str(saved_path))
        except Exception as exc:  # noqa: BLE001
            self._safe_unlink(tmp_path)
            raise CaptureError(f"failed to save captured photo: {exc}")

        # Each Capture Photo press is its OWN physical capture, so it gets a
        # unique capture identity (trigger time + random suffix) - NOT a content
        # hash. This keeps "one capture -> one inspection" (a retried request
        # for the same press is still deduped) without ever collapsing two
        # distinct presses that happened to grab a byte-identical frame (e.g. a
        # momentarily frozen stream).
        capture_id = f"capture:{int(trigger_ms)}:{uuid.uuid4().hex}"

        record = self.inspection_service.analyze_file(
            str(saved_path), source="capture", captured_at=captured_iso, capture_id=capture_id,
        )

        with self._lock:
            self._last_success_ms = trigger_ms
            self._last_error = None
            self._last_attempts = attempts
            self._total_captures += 1

        return {
            "image_path": str(saved_path),
            "captured_at": captured_iso,
            "timestamp": int(round(trigger_ms / 1000.0)),
            "record": record,
        }

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            Path(path).unlink()
        except OSError:
            pass
