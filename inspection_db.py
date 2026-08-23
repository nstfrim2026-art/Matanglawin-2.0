"""
inspection_db.py - SQLite-backed storage for crack inspection records.

Each row represents ONE photo-based inspection - either a manually
uploaded photo (fallback path) or a photo imported automatically from
the DJI controller via the watch folder. There is no "live video /
continuous detection" writer; every record corresponds to a single still
image analyzed once.

Columns (exactly the inspection-history fields the spec calls for):
    id           - autoincrement inspection ID
    timestamp    - ISO-ish local timestamp string
    status       - "CRACK DETECTED" / "NO CRACK DETECTED"
    source       - "upload" (manual fallback) / "import" (automatic DJI)
    original_image_path      - the untouched photo
    highlighted_image_path   - the red-highlighted photo
    num_instances - internal-only crack count (NEVER surfaced to the UI)
    latitude/longitude/altitude_m - aircraft position at capture (nullable)
    gps_available / gps_time_delta_ms / gps_source / captured_at
                  - geotag quality metadata (nullable)

``to_dict()`` returns what the OPERATOR UI is allowed to show
(status/source/timestamp plus the capture latitude/longitude, which are
displayed in the inspection details) and still omits confidence, crack
counts, and every other model statistic.

This module has no Flask/YOLO dependency; it only needs the standard
library (sqlite3), so it's trivial to unit test in isolation. A tiny
migration in __init__ upgrades any pre-existing DB by adding new columns
in place.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

STATUS_CRACK = "CRACK DETECTED"
STATUS_NO_CRACK = "NO CRACK DETECTED"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'NO CRACK DETECTED',
    source TEXT NOT NULL DEFAULT 'upload',
    original_image_path TEXT,
    highlighted_image_path TEXT,
    num_instances INTEGER NOT NULL DEFAULT 0,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    gps_available INTEGER NOT NULL DEFAULT 0,
    gps_time_delta_ms REAL,
    gps_source TEXT,
    captured_at TEXT,
    radius_m REAL,
    capture_id TEXT
);
"""

# capture_id is the single identity of one physical capture, threaded through
# the whole automatic pipeline (image save -> import -> service -> DB). A
# partial UNIQUE index enforces "one capture -> one inspection" at the storage
# layer: any second insert with the same capture_id fails, so duplicates are
# impossible even under a race or two import transports. Manual uploads pass
# capture_id = NULL (SQLite allows many NULLs in a UNIQUE index), so each
# manual upload remains its own record.
CAPTURE_ID_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_inspections_capture_id "
    "ON inspections(capture_id) WHERE capture_id IS NOT NULL"
)

# Columns that may be missing from an older on-disk DB and need adding.
_EXPECTED_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'NO CRACK DETECTED'",
    "source": "TEXT NOT NULL DEFAULT 'upload'",
    "original_image_path": "TEXT",
    "highlighted_image_path": "TEXT",
    "num_instances": "INTEGER NOT NULL DEFAULT 0",
    "latitude": "REAL",
    "longitude": "REAL",
    "altitude_m": "REAL",
    "gps_available": "INTEGER NOT NULL DEFAULT 0",
    "gps_time_delta_ms": "REAL",
    "gps_source": "TEXT",
    "captured_at": "TEXT",
    "radius_m": "REAL",
    "capture_id": "TEXT",
}


@dataclass
class InspectionRecord:
    id: Optional[int]
    timestamp: str
    status: str
    source: str = "upload"
    original_image_path: Optional[str] = None
    highlighted_image_path: Optional[str] = None
    num_instances: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    gps_available: bool = False
    gps_time_delta_ms: Optional[float] = None
    gps_source: Optional[str] = None
    captured_at: Optional[str] = None
    radius_m: Optional[float] = None
    capture_id: Optional[str] = None

    @property
    def has_crack(self) -> bool:
        return self.status == STATUS_CRACK

    @property
    def source_label(self) -> str:
        return "DJI import" if self.source == "import" else "Manual upload"

    def to_dict(self) -> dict:
        """
        User-facing/serializable view. Deliberately omits confidence,
        crack counts, bounding boxes, and every other model statistic -
        only the inspection result and the information shown in the
        inspection details: status + source + timestamp plus the capture
        location (latitude/longitude, when GPS was available). Image URLs
        are added by app.py.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "status": self.status,
            "has_crack": self.has_crack,
            "source": self.source,
            "source_label": self.source_label,
            "gps_available": bool(self.gps_available),
            "latitude": self.latitude if self.gps_available else None,
            "longitude": self.longitude if self.gps_available else None,
        }


class InspectionDB:
    """
    Thin, thread-safe SQLite wrapper. One connection per DB instance,
    guarded by a lock, since the photo-import watcher thread and Flask
    request threads may read/write concurrently.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(SCHEMA)
        self._migrate()
        # Enforce one-capture-one-inspection at the DB level (after migrate so
        # the capture_id column exists on upgraded databases too).
        with self._conn:
            self._conn.execute(CAPTURE_ID_INDEX)

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA table_info(inspections)")
            existing = {row["name"] for row in cur.fetchall()}
            for col, decl in _EXPECTED_COLUMNS.items():
                if col not in existing:
                    # SQLite can't add a NOT NULL column without a default;
                    # all our declarations include one, so this is safe.
                    cur.execute(f"ALTER TABLE inspections ADD COLUMN {col} {decl}")
            self._conn.commit()
            cur.close()

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def add_inspection(
        self,
        timestamp: str,
        status: str,
        source: str = "upload",
        original_image_path: Optional[str] = None,
        highlighted_image_path: Optional[str] = None,
        num_instances: int = 0,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude_m: Optional[float] = None,
        gps_available: bool = False,
        gps_time_delta_ms: Optional[float] = None,
        gps_source: Optional[str] = None,
        captured_at: Optional[str] = None,
        radius_m: Optional[float] = None,
        capture_id: Optional[str] = None,
    ) -> int:
        """
        Insert one inspection and return its id. If ``capture_id`` is given
        and a row with that capture_id already exists, this raises
        ``sqlite3.IntegrityError`` (the UNIQUE index) - callers use that to
        keep "one capture -> one inspection".
        """
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO inspections
                    (timestamp, status, source, original_image_path,
                     highlighted_image_path, num_instances,
                     latitude, longitude, altitude_m, gps_available,
                     gps_time_delta_ms, gps_source, captured_at, radius_m,
                     capture_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    status,
                    source,
                    original_image_path,
                    highlighted_image_path,
                    num_instances,
                    latitude,
                    longitude,
                    altitude_m,
                    1 if gps_available else 0,
                    gps_time_delta_ms,
                    gps_source,
                    captured_at,
                    radius_m,
                    capture_id,
                ),
            )
            return cur.lastrowid

    def get_inspection(self, inspection_id: int) -> Optional[InspectionRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
            row = cur.fetchone()
        return _row_to_record(row) if row else None

    def get_by_capture_id(self, capture_id: str) -> Optional[InspectionRecord]:
        """The inspection for a given capture_id, or None. Used for dedup."""
        if not capture_id:
            return None
        with self._cursor() as cur:
            cur.execute("SELECT * FROM inspections WHERE capture_id = ?", (capture_id,))
            row = cur.fetchone()
        return _row_to_record(row) if row else None

    def list_inspections(
        self, limit: int = 100, offset: int = 0, newest_first: bool = True
    ) -> List[InspectionRecord]:
        order = "DESC" if newest_first else "ASC"
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM inspections ORDER BY id {order} LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()
        return [_row_to_record(r) for r in rows]

    def get_latest(self) -> Optional[InspectionRecord]:
        results = self.list_inspections(limit=1, newest_first=True)
        return results[0] if results else None

    def update_geo(
        self,
        inspection_id: int,
        latitude: float,
        longitude: float,
        gps_time_delta_ms: Optional[float] = None,
        gps_source: Optional[str] = None,
        gps_available: bool = True,
        radius_m: Optional[float] = None,
    ) -> bool:
        """
        Backfill / update an inspection's location (e.g. once telemetry
        appears after capture). Returns True if a row was updated. When
        `radius_m` is given, the inspection-area radius is set too.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE inspections
                   SET latitude = ?, longitude = ?, gps_available = ?,
                       gps_time_delta_ms = ?, gps_source = ?,
                       radius_m = COALESCE(?, radius_m)
                 WHERE id = ?
                """,
                (latitude, longitude, 1 if gps_available else 0,
                 gps_time_delta_ms, gps_source, radius_m, inspection_id),
            )
            return cur.rowcount > 0

    def list_pending_geo(self, limit: int = 500) -> List[InspectionRecord]:
        """
        Inspections that still need a location but have a capture time to
        match against (gps_available = 0 AND captured_at present). Used by
        the SRT watcher to backfill coordinates when telemetry arrives late.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM inspections
                 WHERE (gps_available = 0 OR gps_available IS NULL)
                   AND captured_at IS NOT NULL AND captured_at != ''
                   AND source = 'import'
                 ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM inspections")
            row = cur.fetchone()
        return int(row["c"]) if row else 0

    def counts(self) -> dict:
        """
        Dashboard summary straight from the DB:
            {"total": N, "cracks": C, "clear": N - C}
        `cracks` counts rows whose status is CRACK DETECTED; `clear` is
        everything else. No model/debug metrics are involved.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS cracks "
                "FROM inspections",
                (STATUS_CRACK,),
            )
            row = cur.fetchone()
        total = int(row["total"]) if row and row["total"] is not None else 0
        cracks = int(row["cracks"]) if row and row["cracks"] is not None else 0
        return {"total": total, "cracks": cracks, "clear": total - cracks}

    def delete_inspection(self, inspection_id: int) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_record(row: sqlite3.Row) -> InspectionRecord:
    keys = row.keys()
    return InspectionRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        status=row["status"] if "status" in keys and row["status"] else STATUS_NO_CRACK,
        source=row["source"] if "source" in keys and row["source"] else "upload",
        original_image_path=row["original_image_path"] if "original_image_path" in keys else None,
        highlighted_image_path=row["highlighted_image_path"] if "highlighted_image_path" in keys else None,
        num_instances=row["num_instances"] if "num_instances" in keys and row["num_instances"] is not None else 0,
        latitude=row["latitude"] if "latitude" in keys else None,
        longitude=row["longitude"] if "longitude" in keys else None,
        altitude_m=row["altitude_m"] if "altitude_m" in keys else None,
        gps_available=bool(row["gps_available"]) if "gps_available" in keys and row["gps_available"] is not None else False,
        gps_time_delta_ms=row["gps_time_delta_ms"] if "gps_time_delta_ms" in keys else None,
        gps_source=row["gps_source"] if "gps_source" in keys else None,
        captured_at=row["captured_at"] if "captured_at" in keys else None,
        radius_m=row["radius_m"] if "radius_m" in keys else None,
        capture_id=row["capture_id"] if "capture_id" in keys else None,
    )
