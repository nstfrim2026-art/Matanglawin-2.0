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

``to_dict()`` returns only what the OPERATOR UI is allowed to show
(status/source/timestamp) and still omits confidence, counts, and
coordinates. The offline map uses ``to_map()`` instead, which adds the
stored latitude/longitude so inspections can be placed as markers - the
coordinates never leak into the operator's result screen.

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
    captured_at TEXT
);
"""

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
        only the inspection result and the information needed to display
        it (status + source + timestamp; image URLs are added by app.py).
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "status": self.status,
            "has_crack": self.has_crack,
            "source": self.source,
            "source_label": self.source_label,
        }

    def to_map(self) -> dict:
        """
        View for the offline inspection MAP: the operator-safe fields plus
        the stored aircraft coordinates + geotag quality. Used only by the
        map/points API, never by the operator result screen.
        """
        d = self.to_dict()
        d.update(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude_m=self.altitude_m,
            gps_available=bool(self.gps_available),
            gps_time_delta_ms=self.gps_time_delta_ms,
            gps_source=self.gps_source,
            captured_at=self.captured_at,
        )
        return d


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
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO inspections
                    (timestamp, status, source, original_image_path,
                     highlighted_image_path, num_instances,
                     latitude, longitude, altitude_m, gps_available,
                     gps_time_delta_ms, gps_source, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            return cur.lastrowid

    def get_inspection(self, inspection_id: int) -> Optional[InspectionRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
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
    ) -> bool:
        """
        Backfill / update an inspection's aircraft location (e.g. once an SRT
        telemetry file appears after capture - Mode B). Returns True if a row
        was updated.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE inspections
                   SET latitude = ?, longitude = ?, gps_available = ?,
                       gps_time_delta_ms = ?, gps_source = ?
                 WHERE id = ?
                """,
                (latitude, longitude, 1 if gps_available else 0,
                 gps_time_delta_ms, gps_source, inspection_id),
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
    )
