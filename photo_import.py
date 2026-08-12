"""
photo_import.py - Watch-folder importer for DJI-captured inspection photos.

This is the practical, reliable bridge between "the DJI controller took a
photo" and "MatanglaWIN analyzed it". The application watches a local
Windows folder; when a new image file appears and has finished copying,
it is analyzed exactly once through the SAME central pipeline as a manual
upload (inspection_service.InspectionService).

Why a watch folder (and not a direct DJI API call): DJI Fly / the RC
controller do not expose a documented, reliable local API on Windows for
pushing a freshly captured still into a third-party app. The photos land
on the controller/app storage and are transferred to the PC by the user
(USB/SD copy, DJI Assistant, phone sync, etc.). A monitored import folder
is the cleanest mechanism that works regardless of how the file arrives -
see README.md for the exact DJI-to-Windows transfer step. This module
never claims to talk to the drone directly; it only reacts to files
appearing on disk.

Safety properties:
    - Partial-copy safe: a file is only analyzed after its size has been
      stable across `stable_polls` consecutive scans, so a half-copied
      photo is never fed to the model.
    - Dedup: content is hashed (SHA-256); a hash already processed is
      skipped, so the same photo is never analyzed twice - even across
      app restarts (the processed-hash set is persisted to a small JSON
      state file), and even if copied in again under a different name.
    - Fault-tolerant: an unreadable/corrupt file is recorded as handled
      and skipped; a single bad file never stops the watcher or crashes
      the app.

MediaMTX/RTSP/WebRTC are irrelevant here: photo import is completely
independent of the live-video POV, so the drone stream being offline
never affects inspection of imported (or uploaded) photos.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from inspection_service import InspectionService, InvalidImageError

DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_STABLE_POLLS = 2  # size must be unchanged this many consecutive scans
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class PhotoImportWatcher:
    def __init__(
        self,
        import_dir: str,
        service: InspectionService,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        stable_polls: int = DEFAULT_STABLE_POLLS,
        state_path: Optional[str] = None,
    ):
        self.import_dir = Path(import_dir)
        self.import_dir.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.poll_interval = poll_interval
        self.stable_polls = max(1, stable_polls)
        self.state_path = Path(state_path) if state_path else (self.import_dir / ".matanglawin_import_state.json")

        # path -> (last_size, consecutive_stable_count)
        self._pending: Dict[str, Tuple[int, int]] = {}
        # content hashes we've already analyzed (persisted)
        self._processed_hashes = self._load_state()
        # file signatures (path,size,mtime) we've already handled this run
        self._handled_signatures = set()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # -- state persistence ---------------------------------------------

    def _load_state(self) -> set:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return set(data.get("processed_hashes", []))
        except (OSError, ValueError):
            return set()

    def _save_state(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps({"processed_hashes": sorted(self._processed_hashes)}), encoding="utf-8"
            )
        except OSError:
            pass

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="PhotoImportWatcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - the watcher must never die
                pass
            self._stop_event.wait(self.poll_interval)

    # -- one scan pass (also directly callable, e.g. from tests) -------

    def poll_once(self) -> List[dict]:
        """
        Scan the import folder once. Returns a list of result dicts for
        files acted on in this pass:
            {"path": str, "action": "analyzed"|"skipped_duplicate"|"invalid",
             "inspection_id": Optional[int]}
        Files that are new-but-not-yet-stable produce no entry (they'll be
        picked up on a later pass).
        """
        results: List[dict] = []
        try:
            entries = sorted(self.import_dir.iterdir())
        except OSError:
            return results

        for path in entries:
            if not path.is_file():
                continue
            if path.name == self.state_path.name:
                continue
            if path.suffix.lower() not in ALLOWED_EXTS:
                continue

            try:
                st = path.stat()
            except OSError:
                continue

            signature = (str(path), st.st_size, int(st.st_mtime))
            if signature in self._handled_signatures:
                continue  # already dealt with this exact file state

            # -- stability check (partial-copy safety) --
            key = str(path)
            last_size, stable_count = self._pending.get(key, (None, 0))
            if last_size == st.st_size and st.st_size > 0:
                stable_count += 1
            else:
                stable_count = 1
            self._pending[key] = (st.st_size, stable_count)
            if stable_count < self.stable_polls:
                continue  # not stable yet; wait for another pass

            # -- stable: dedup by content hash --
            file_hash = _hash_file(path)
            if file_hash is None:
                self._handled_signatures.add(signature)
                results.append({"path": str(path), "action": "invalid", "inspection_id": None})
                continue

            with self._lock:
                already = file_hash in self._processed_hashes
            if already:
                self._handled_signatures.add(signature)
                self._pending.pop(key, None)
                results.append({"path": str(path), "action": "skipped_duplicate", "inspection_id": None})
                continue

            # -- analyze once --
            try:
                record = self.service.analyze_file(str(path), source="import")
            except InvalidImageError:
                self._handled_signatures.add(signature)
                self._pending.pop(key, None)
                results.append({"path": str(path), "action": "invalid", "inspection_id": None})
                continue
            except Exception:  # noqa: BLE001 - never let one file kill the watcher
                self._handled_signatures.add(signature)
                results.append({"path": str(path), "action": "invalid", "inspection_id": None})
                continue

            with self._lock:
                self._processed_hashes.add(file_hash)
                self._save_state()
            self._handled_signatures.add(signature)
            self._pending.pop(key, None)
            results.append(
                {"path": str(path), "action": "analyzed", "inspection_id": record.id}
            )

        return results
