"""
import_ledger.py - Content-hash de-duplication for the automatic photo
transfer bridge's HTTP ingest endpoint (/api/import).

The DJI photo-transfer bridge (dji_photo_bridge.py, or any folder-sync
tool + a re-post) may legitimately retry an upload when the PC is briefly
unavailable, or the same original photo may reach the PC by more than one
route. To honor "the same DJI photo must never be analyzed repeatedly",
this ledger records the SHA-256 of every photo that has already been
accepted for analysis and rejects repeats.

It is intentionally tiny, dependency-free (stdlib only), thread-safe, and
persisted to a small JSON file so de-duplication survives an app restart -
mirroring the behavior photo_import.PhotoImportWatcher already provides for
the watch-folder path.

This never touches the analysis pipeline or its output; it only decides
whether a set of image bytes has been seen before.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def hash_bytes(data: bytes) -> str:
    """SHA-256 of raw image bytes - the identity used for de-duplication."""
    return hashlib.sha256(data).hexdigest()


class ImportLedger:
    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hashes = self._load()

    def _load(self) -> set:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return set(data.get("processed_hashes", []))
        except (OSError, ValueError):
            return set()

    def _save(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps({"processed_hashes": sorted(self._hashes)}), encoding="utf-8"
            )
        except OSError:
            # The ledger is a convenience for cross-restart de-dup; failing
            # to persist must not break ingestion.
            pass

    def seen(self, digest: str) -> bool:
        with self._lock:
            return digest in self._hashes

    def add(self, digest: str) -> None:
        with self._lock:
            self._hashes.add(digest)
            self._save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._hashes)
