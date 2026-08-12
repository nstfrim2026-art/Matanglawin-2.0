"""
inspection_db.py - SQLite-backed storage for crack inspection records.

Each row represents ONE photo-based inspection (see
inspection_service.py) - either a manually uploaded photo or a photo
imported from the DJI controller via the watch folder. There is no
longer any "live video / continuous detection" writer; every record
corresponds to a single still image analyzed once.

Columns:
    id, timestamp, status ("CRACK DETECTED" / "NO CRACK DETECTED"),
    source ("upload" / "import"), original_image_path,
    highlighted_image_path, crack_image_paths (JSON list),
    num_instances, latitude, longitude, altitude, gps_available,
    detection_info (free-form JSON - internal only; may hold
    confidences/bboxes, which are NOT surfaced to the UI).

GPS fields are nullable. We never fabricate coordinates - if GPS wasn't
available, latitude/longitude/altitude are stored as NULL and
gps_available is 0.

Confidence is intentionally NOT a user-facing field: it may live inside
`detection_info` for internal/debug use, but `to_dict()` never returns
it, so no confidence value can leak into the UI/API.

This module has no Flask/YOLO dependency; it only needs the standard
library (sqlite3), so it's trivial to unit test in isolation. A tiny
migration in __init__ upgrades any pre-existing DB (from an older
schema) by adding the new columns in place.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    crack_image_paths TEXT,          -- JSON list of crack-only image paths
    num_instances INTEGER NOT NULL DEFAULT 0,
    latitude REAL,
    longitude REAL,
    altitude REAL,
    gps_available INTEGER NOT NULL DEFAULT 0,
    detection_info TEXT              -- internal JSON (confidences/bboxes); never shown in UI
);
"""

# Columns that may be missing from an older on-disk DB and need adding.
_EXPECTED_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'NO CRACK DETECTED'",
    "source": "TEXT NOT NULL DEFAULT 'upload'",
    "original_image_path": "TEXT",
    "highlighted_image_path": "TEXT",
    "crack_image_paths": "TEXT",
    "num_instances": "INTEGER NOT NULL DEFAULT 0",
    "latitude": "REAL",
    "longitude": "REAL",
    "altitude": "REAL",
    "gps_available": "INTEGER NOT NULL DEFAULT 0",
    "detection_info": "TEXT",
}


@dataclass
class InspectionRecord:
    id: Optional[int]
    timestamp: str
    status: str
    source: str = "upload"
    original_image_path: Optional[str] = None
    highlighted_image_path: Optional[str] = None
    crack_image_paths: List[str] = field(default_factory=list)
    num_instances: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    gps_available: bool = False
    detection_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_crack(self) -> bool:
        return self.status == STATUS_CRACK

    def crack_image_names(self) -> List[str]:
        return [Path(p).name for p in self.crack_image_paths]

    def to_dict(self) -> dict:
        """
        User-facing/serializable view. Deliberately omits confidence and
        any raw model statistics - only the inspection result and the
        information needed to display it.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "status": self.status,
            "has_crack": self.has_crack,
            "source": self.source,
            "num_instances": self.num_instances,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "gps_available": self.gps_available,
            "crack_image_names": self.crack_image_names(),
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

    def _migrate(self) -> None:
        """
        Upgrade a pre-existing DB to the current schema.

        Two cases:
        - Legacy live-detection schema (has `cropped_image_path` /
          `confidence` columns, some NOT NULL and no longer populated):
          rebuild the table into the new shape, best-effort mapping old
          rows across (SQLite's standard rename+recreate+copy pattern,
          which also drops the old NOT NULL constraints).
        - Otherwise (fresh or already-current): just ADD any columns
          that happen to be missing.
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("PRAGMA table_info(inspections)")
                existing = {row["name"] for row in cur.fetchall()}
                legacy_markers = {"cropped_image_path", "confidence"}
                if existing & legacy_markers:
                    self._rebuild_from_legacy(cur, existing)
                else:
                    for col, decl in _EXPECTED_COLUMNS.items():
                        if col not in existing:
                            # All decls include a default, which SQLite
                            # requires when adding a NOT NULL column.
                            cur.execute(f"ALTER TABLE inspections ADD COLUMN {col} {decl}")
                self._conn.commit()
            finally:
                cur.close()

    def _rebuild_from_legacy(self, cur, existing_cols: set) -> None:
        """Rebuild an old live-detection table into the current schema."""
        legacy_rows = cur.execute("SELECT * FROM inspections").fetchall()

        cur.execute("ALTER TABLE inspections RENAME TO inspections_legacy")
        cur.execute(SCHEMA)  # fresh `inspections` in the current shape

        for r in legacy_rows:
            keys = r.keys()
            original = r["image_path"] if "image_path" in keys else None
            crack = r["cropped_image_path"] if "cropped_image_path" in keys else None
            status = STATUS_CRACK if crack else STATUS_NO_CRACK
            crack_list = [crack] if crack else []
            cur.execute(
                """
                INSERT INTO inspections
                    (id, timestamp, status, source, original_image_path,
                     highlighted_image_path, crack_image_paths, num_instances,
                     latitude, longitude, altitude, gps_available, detection_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["id"] if "id" in keys else None,
                    r["timestamp"] if "timestamp" in keys else "",
                    status,
                    "upload",
                    original,
                    original,  # best-effort: no separate highlighted image in legacy data
                    json.dumps(crack_list),
                    r["num_instances"] if "num_instances" in keys and r["num_instances"] is not None else (1 if crack else 0),
                    r["latitude"] if "latitude" in keys else None,
                    r["longitude"] if "longitude" in keys else None,
                    r["altitude"] if "altitude" in keys else None,
                    r["gps_available"] if "gps_available" in keys and r["gps_available"] is not None else 0,
                    r["detection_info"] if "detection_info" in keys else json.dumps({}),
                ),
            )

        cur.execute("DROP TABLE inspections_legacy")

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
        crack_image_paths: Optional[List[str]] = None,
        num_instances: int = 0,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
        detection_info: Optional[Dict[str, Any]] = None,
    ) -> int:
        gps_available = latitude is not None and longitude is not None
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO inspections
                    (timestamp, status, source, original_image_path,
                     highlighted_image_path, crack_image_paths, num_instances,
                     latitude, longitude, altitude, gps_available, detection_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    status,
                    source,
                    original_image_path,
                    highlighted_image_path,
                    json.dumps(crack_image_paths or []),
                    num_instances,
                    latitude,
                    longitude,
                    altitude,
                    1 if gps_available else 0,
                    json.dumps(detection_info or {}),
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


def _json_or_default(value, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def _row_to_record(row: sqlite3.Row) -> InspectionRecord:
    keys = row.keys()
    return InspectionRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        status=row["status"] if "status" in keys and row["status"] else STATUS_NO_CRACK,
        source=row["source"] if "source" in keys and row["source"] else "upload",
        original_image_path=row["original_image_path"] if "original_image_path" in keys else None,
        highlighted_image_path=row["highlighted_image_path"] if "highlighted_image_path" in keys else None,
        crack_image_paths=_json_or_default(row["crack_image_paths"] if "crack_image_paths" in keys else None, []),
        num_instances=row["num_instances"] if "num_instances" in keys and row["num_instances"] is not None else 0,
        latitude=row["latitude"] if "latitude" in keys else None,
        longitude=row["longitude"] if "longitude" in keys else None,
        altitude=row["altitude"] if "altitude" in keys else None,
        gps_available=bool(row["gps_available"]) if "gps_available" in keys else False,
        detection_info=_json_or_default(row["detection_info"] if "detection_info" in keys else None, {}),
    )
