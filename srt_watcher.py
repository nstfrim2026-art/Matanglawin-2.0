"""
srt_watcher.py - Automatic, in-process watcher for DJI SRT telemetry.

Runs inside MatanglaWIN (like photo_import.PhotoImportWatcher) so the
operator NEVER has to run a parse command or upload SRT files. It monitors
configurable local folders for *.srt / *.SRT, parses new telemetry with
srt_telemetry, and:

    * feeds each new (lat, lon, timestamp) sample into the shared
      telemetry_store (so a capture can be matched to the nearest position);
    * backfills any inspection that was captured WITHOUT a location once a
      matching SRT sample becomes available (Mode B, post-recording);
    * tracks whether SRT is arriving incrementally during recording
      (Mode A, "live") or only after recording (Mode B) - reported honestly,
      never assumed.

Robustness: tolerant of missing/locked/partial/half-written files, malformed
blocks, and files that are rewritten; a single bad file never stops the
watcher or the web app. It never modifies or deletes the SRT files.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import srt_telemetry

DEFAULT_POLL_INTERVAL = 2.0
# A file touched within this window is considered part of an active recording.
LIVE_RECENCY_MS = 6000.0

MODE_IDLE = "idle"
MODE_LIVE = "AUTOMATED LIVE AIRCRAFT POSITION"          # SRT grows during recording
MODE_POST = "AUTOMATED POST-CAPTURE GEOTAGGING"          # SRT only after recording


class SrtWatcher:
    def __init__(
        self,
        srt_dirs: List[str],
        telemetry_store,
        db=None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self.srt_dirs = [Path(d) for d in srt_dirs if d]
        self.telemetry_store = telemetry_store
        self.db = db
        self.poll_interval = poll_interval

        # path -> (size, mtime, added_count)
        self._files: Dict[str, Tuple[int, float, int]] = {}
        self.mode = MODE_IDLE
        self.samples_ingested = 0
        self.backfilled = 0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        for d in self.srt_dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SrtWatcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - the watcher must never die
                pass
            self._stop.wait(self.poll_interval)

    # -- discovery + ingest -------------------------------------------

    def _iter_srt_files(self) -> List[Path]:
        found: List[Path] = []
        for d in self.srt_dirs:
            try:
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() == ".srt":
                        found.append(p)
            except OSError:
                continue
        return found

    def poll_once(self, now_ms: Optional[float] = None) -> dict:
        """
        One scan pass. Ingests new SRT samples and backfills pending
        inspections. Returns a summary dict; never raises.
        """
        now_ms = time.time() * 1000.0 if now_ms is None else now_ms
        ingested = 0
        saw_live = False

        for path in self._iter_srt_files():
            try:
                st = path.stat()
            except OSError:
                continue
            key = str(path)
            prev = self._files.get(key)
            if prev is not None and prev[0] == st.st_size and abs(prev[1] - st.st_mtime) < 1e-6:
                continue  # unchanged since last poll

            samples = srt_telemetry.parse_file(key)
            prev_count = prev[2] if prev else 0
            if len(samples) < prev_count:
                prev_count = 0  # file was rewritten/rotated -> re-ingest all
            new_samples = samples[prev_count:]

            for s in new_samples:
                if self.telemetry_store is not None:
                    self.telemetry_store.add(s.latitude, s.longitude,
                                             timestamp=s.timestamp_ms, source="dji_srt")
                ingested += 1

            # Mode detection: file changed AND was touched very recently AND
            # it grew => SRT is arriving during recording (live).
            grew = prev is not None and st.st_size > prev[0]
            recent = (now_ms - st.st_mtime * 1000.0) <= LIVE_RECENCY_MS
            if grew and recent and new_samples:
                saw_live = True

            self._files[key] = (st.st_size, st.st_mtime, len(samples))

        self.samples_ingested += ingested
        if ingested:
            self.mode = MODE_LIVE if saw_live else MODE_POST

        updated = self._backfill(now_ms=now_ms) if ingested else 0
        return {"ingested": ingested, "backfilled": updated, "mode": self.mode}

    # -- backfill pending inspections (Mode B) ------------------------

    def _backfill(self, now_ms: Optional[float] = None) -> int:
        if self.db is None or self.telemetry_store is None:
            return 0
        try:
            pending = self.db.list_pending_geo(limit=500)
        except Exception:  # noqa: BLE001
            return 0
        updated = 0
        for rec in pending:
            capture_ms = srt_telemetry_parse_ts(rec.captured_at)
            if capture_ms is None:
                continue
            match = self.telemetry_store.match_for_capture(capture_ms)
            if match.get("gps_available"):
                try:
                    if self.db.update_geo(
                        rec.id, match["latitude"], match["longitude"],
                        gps_time_delta_ms=match["gps_time_delta_ms"],
                        gps_source=match.get("gps_source") or "dji_srt",
                        gps_available=True,
                    ):
                        updated += 1
                except Exception:  # noqa: BLE001
                    continue
        self.backfilled += updated
        return updated

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "watched_dirs": [str(d) for d in self.srt_dirs],
            "files_tracked": len(self._files),
            "samples_ingested": self.samples_ingested,
            "backfilled": self.backfilled,
        }


def srt_telemetry_parse_ts(value) -> Optional[float]:
    """Parse a capture timestamp (ISO/epoch) to epoch-ms, via telemetry_store."""
    try:
        import telemetry_store
        return telemetry_store.parse_timestamp_ms(value)
    except Exception:  # noqa: BLE001
        return None
