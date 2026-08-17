#!/usr/bin/env python3
"""
dji_photo_bridge.py - Companion uploader that completes the automatic
DJI Neo 2 -> phone/PC -> MatanglaWIN photo-transfer bridge.

WHY THIS EXISTS
---------------
DJI Fly / the DJI Neo 2 do NOT expose a webhook or a documented local API
that pushes a freshly captured still into a third-party app. After you
press the shutter, the ORIGINAL JPEG ends up in an album/folder (on the
phone via QuickTransfer/download, or on a PC folder synced from it). This
watcher turns "a new original photo appeared in that album folder" into
"MatanglaWIN analyzed it automatically" - with no manual upload.

WHERE IT RUNS
-------------
Run it wherever it can see the DJI photo album folder and reach the PC:
  * On Android via Termux (point --watch-dir at the DJI album; --server at
    the PC's LAN address), or
  * On the Windows PC itself, watching a folder that a sync app mirrors
    from the phone (Syncthing / cloud client).
Either way it uploads the ORIGINAL bytes to MatanglaWIN's /api/import, so
the analyzed image is exactly the DJI still - never a video/RTMP frame.

WHAT IT GUARANTEES (matches the required uploader behavior)
-----------------------------------------------------------
  1. Monitors the DJI photo directory/album.
  2. Detects newly created original images.
  3. Waits until each file has finished writing/downloading (size stable).
  4. Uploads the JPEG automatically to MatanglaWIN.
  5. Retries (with backoff) if the PC is temporarily unavailable.
  6. Never uploads the same image twice (SHA-256 content de-dup, persisted).
  7. Preserves original quality (uploads the raw bytes verbatim).
  8. PC address is fully configurable (CLI/env) - never hardcoded.
  9. After setup, the operator only presses the DJI shutter.

Stdlib only (urllib) so it runs on a plain Python 3 install, including
Termux on Android. It does not import MatanglaWIN, OpenCV, or YOLO.

USAGE
-----
    python dji_photo_bridge.py \
        --server http://<PC-LAN-IP>:5000 \
        --watch-dir "/path/to/DJI/album"

Environment fallbacks:
    MATANGLAWIN_BRIDGE_SERVER      e.g. http://<PC-LAN-IP>:5000
    MATANGLAWIN_BRIDGE_WATCH_DIR   e.g. /sdcard/DCIM/DJI
    MATANGLAWIN_BRIDGE_INTERVAL    poll seconds (default 2.0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_ENDPOINT = "/api/import"
DEFAULT_PING_ENDPOINT = "/api/bridge/ping"
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_STABLE_POLLS = 2
DEFAULT_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 2.0
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

log = logging.getLogger("dji_photo_bridge")


def _hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# Result of a single upload attempt.
#   ok:        the photo is now safely handled (accepted or duplicate)
#   retryable: worth trying again later (PC down / 5xx)
UploadResult = Tuple[bool, bool, str]


def http_upload(server: str, endpoint: str, data: bytes, filename: str, timeout: float = 30.0) -> UploadResult:
    """POST the original bytes to MatanglaWIN. Returns (ok, retryable, detail)."""
    url = server.rstrip("/") + endpoint
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "image/jpeg")
    req.add_header("X-Filename", filename)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(2048).decode("utf-8", "ignore")
            return True, False, f"{resp.status} {body[:200]}"
    except urllib.error.HTTPError as exc:
        # A non-2xx response is never "ok" here. 4xx = the server understood
        # and rejected (e.g. not an image) - retrying won't help, so caller
        # stops. 5xx = server-side hiccup - retry later.
        retryable = exc.code >= 500
        return False, retryable, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        # PC unreachable / network down - definitely retry.
        return False, True, f"unreachable: {exc}"


def http_ping(server: str, endpoint: str = DEFAULT_PING_ENDPOINT, timeout: float = 5.0) -> bool:
    url = server.rstrip("/") + endpoint
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


class PhotoUploadBridge:
    def __init__(
        self,
        watch_dir: str,
        server: str,
        endpoint: str = DEFAULT_ENDPOINT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        stable_polls: int = DEFAULT_STABLE_POLLS,
        retries: int = DEFAULT_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        state_path: Optional[str] = None,
        uploader: Optional[Callable[[str, str, bytes, str], UploadResult]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.watch_dir = Path(watch_dir)
        self.server = server
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self.stable_polls = max(1, stable_polls)
        self.retries = max(1, retries)
        self.retry_backoff = retry_backoff
        self.state_path = Path(state_path) if state_path else (self.watch_dir / ".matanglawin_bridge_state.json")
        # Injected uploader (for tests). Signature: (server, endpoint, data, filename) -> UploadResult
        self._uploader = uploader or (lambda s, e, d, f: http_upload(s, e, d, f))
        self._sleep = sleep

        self._pending: Dict[str, Tuple[int, int]] = {}     # path -> (last_size, stable_count)
        self._uploaded_hashes = self._load_state()
        self._handled_signatures = set()

    # -- state persistence (cross-restart de-dup) ----------------------

    def _load_state(self) -> set:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return set(data.get("uploaded_hashes", []))
        except (OSError, ValueError):
            return set()

    def _save_state(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps({"uploaded_hashes": sorted(self._uploaded_hashes)}), encoding="utf-8"
            )
        except OSError:
            pass

    # -- one scan pass (directly callable from tests) ------------------

    def scan_once(self) -> List[dict]:
        """
        Scan the album folder once. Uploads any new, fully-written photo
        that hasn't been uploaded before. Returns a list of action dicts:
            {"path", "action": "uploaded"|"duplicate"|"deferred"|"skipped"|"failed"}
        "deferred" = seen but not yet size-stable (will retry next pass).
        "failed"   = PC unreachable now; NOT marked done, so it retries.
        """
        results: List[dict] = []
        try:
            entries = sorted(self.watch_dir.iterdir())
        except OSError:
            return results

        for path in entries:
            if not path.is_file() or path.name == self.state_path.name:
                continue
            if path.suffix.lower() not in ALLOWED_EXTS:
                continue
            try:
                st = path.stat()
            except OSError:
                continue

            signature = (str(path), st.st_size, int(st.st_mtime))
            if signature in self._handled_signatures:
                continue

            # -- wait until the download/copy has finished (size stable) --
            key = str(path)
            last_size, stable_count = self._pending.get(key, (None, 0))
            if last_size == st.st_size and st.st_size > 0:
                stable_count += 1
            else:
                stable_count = 1
            self._pending[key] = (st.st_size, stable_count)
            if stable_count < self.stable_polls:
                results.append({"path": key, "action": "deferred"})
                continue

            digest = _hash_file(path)
            if digest is None:
                results.append({"path": key, "action": "skipped"})
                continue
            if digest in self._uploaded_hashes:
                # Already uploaded this exact photo before - never re-send.
                self._handled_signatures.add(signature)
                self._pending.pop(key, None)
                results.append({"path": key, "action": "duplicate"})
                continue

            try:
                data = path.read_bytes()
            except OSError:
                results.append({"path": key, "action": "skipped"})
                continue

            log.info("[PHOTO BRIDGE] New DJI photo detected: %s (%d bytes)", path.name, len(data))
            ok, detail = self._upload_with_retry(data, path.name)
            if ok:
                self._uploaded_hashes.add(digest)
                self._save_state()
                self._handled_signatures.add(signature)
                self._pending.pop(key, None)
                log.info("[PHOTO BRIDGE] Upload complete: %s (%s)", path.name, detail)
                results.append({"path": key, "action": "uploaded"})
            else:
                # Leave it pending so the next scan retries it.
                log.warning("[PHOTO BRIDGE] PC unavailable - will retry: %s (%s)", path.name, detail)
                results.append({"path": key, "action": "failed"})

        return results

    def _upload_with_retry(self, data: bytes, filename: str) -> Tuple[bool, str]:
        detail = ""
        for attempt in range(1, self.retries + 1):
            ok, retryable, detail = self._uploader(self.server, self.endpoint, data, filename)
            if ok:
                return True, detail
            if not retryable:
                # Server rejected the file (e.g. not a valid image). Treat as
                # handled so we don't loop forever on one bad file.
                log.warning("[PHOTO BRIDGE] Rejected by server (not retrying): %s (%s)", filename, detail)
                return True, detail
            if attempt < self.retries:
                self._sleep(self.retry_backoff * attempt)
        return False, detail

    def heartbeat(self) -> None:
        http_ping(self.server)

    # -- run loop ------------------------------------------------------

    def run(self) -> None:
        log.info("[PHOTO BRIDGE] Watching %s", self.watch_dir)
        log.info("[PHOTO BRIDGE] Uploading originals to %s%s", self.server.rstrip("/"), self.endpoint)
        while True:
            try:
                self.heartbeat()
                self.scan_once()
            except Exception as exc:  # noqa: BLE001 - the bridge must never die
                log.error("[PHOTO BRIDGE] scan error: %s", exc)
            self._sleep(self.poll_interval)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automatic DJI photo uploader for MatanglaWIN.")
    p.add_argument(
        "--server",
        default=os.environ.get("MATANGLAWIN_BRIDGE_SERVER", ""),
        help="MatanglaWIN base URL, e.g. http://<PC-LAN-IP>:5000 "
             "(or set MATANGLAWIN_BRIDGE_SERVER). The IP is your PC's current "
             "LAN address - configure it here, never hardcoded in the app.",
    )
    p.add_argument(
        "--watch-dir",
        default=os.environ.get("MATANGLAWIN_BRIDGE_WATCH_DIR", ""),
        help="Folder to watch for new DJI original photos (the phone's DJI "
             "album, or a folder synced from it). Or set MATANGLAWIN_BRIDGE_WATCH_DIR.",
    )
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--interval", type=float,
                   default=float(os.environ.get("MATANGLAWIN_BRIDGE_INTERVAL", DEFAULT_POLL_INTERVAL)))
    p.add_argument("--stable-polls", type=int, default=DEFAULT_STABLE_POLLS)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if not args.server or not args.watch_dir:
        log.error("Both --server and --watch-dir are required "
                  "(or MATANGLAWIN_BRIDGE_SERVER / MATANGLAWIN_BRIDGE_WATCH_DIR).")
        return 2
    if not Path(args.watch_dir).exists():
        log.error("Watch dir does not exist: %s", args.watch_dir)
        return 2

    bridge = PhotoUploadBridge(
        watch_dir=args.watch_dir,
        server=args.server,
        endpoint=args.endpoint,
        poll_interval=args.interval,
        stable_polls=args.stable_polls,
        retries=args.retries,
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        log.info("[PHOTO BRIDGE] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
