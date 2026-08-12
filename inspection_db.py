"""
inspection_db.py - SQLite-backed storage for crack inspection records.

Each row represents one automatically-captured crack detection event
from the live drone pipeline (see capture_manager.py), holding exactly
the fields required by the PDF report and the "Inspections" page:

    id, timestamp, image_path, cropped_image_path, confidence,
    num_instances, latitude, longitude, altitude, gps_available,
    detection_info (free-form JSON string, e.g. per-instance bboxes)

GPS fields are nullable. We never fabricate coordinates - if GPS wasn't
available at capture time, `latitude`/`longitude`/`altitude` are stored
as NULL and `gps_available` is 0.

This module has no Flask/YOLO dependency; it only needs the standard
library (`sqlite3`), so it's trivial to unit test in isolation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    cropped_image_path TEXT NOT NULL,
    confidence REAL NOT NULL,
    num_instances INTEGER NOT NULL DEFAULT 1,
    latitude REAL,
    longitude REAL,
    altitude REAL,
    gps_available INTEGER NOT NULL DEFAULT 0,
    detection_info TEXT
);
"""


@dataclass
class InspectionRecord:
    id: Optional[int]
    timestamp: str
    cropped_image_path: str
    confidence: float
    image_path: Optional[str] = None
    num_instances: int = 1
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    gps_available: bool = False
    detection_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "cropped_image_path": self.cropped_image_path,
            "confidence": self.confidence,
            "num_instances": self.num_instances,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "gps_available": self.gps_available,
            "detection_info": self.detection_info,
        }


class InspectionDB:
    """
    Thin, thread-safe SQLite wrapper. One connection per DB instance,
    guarded by a lock, since the live-detection background thread and
    Flask request threads both read/write concurrently.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(SCHEMA)

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
        cropped_image_path: str,
        confidence: float,
        image_path: Optional[str] = None,
        num_instances: int = 1,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
        detection_info: Optional[Dict[str, Any]] = None,
    ) -> int:
        gps_available = latitude is not None and longitude is not None
        detection_info_json = json.dumps(detection_info or {})
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO inspections
                    (timestamp, image_path, cropped_image_path, confidence,
                     num_instances, latitude, longitude, altitude,
                     gps_available, detection_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    image_path,
                    cropped_image_path,
                    confidence,
                    num_instances,
                    latitude,
                    longitude,
                    altitude,
                    1 if gps_available else 0,
                    detection_info_json,
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


def _row_to_record(row: sqlite3.Row) -> InspectionRecord:
    try:
        detection_info = json.loads(row["detection_info"]) if row["detection_info"] else {}
    except (TypeError, ValueError):
        detection_info = {}
    return InspectionRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        image_path=row["image_path"],
        cropped_image_path=row["cropped_image_path"],
        confidence=row["confidence"],
        num_instances=row["num_instances"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        altitude=row["altitude"],
        gps_available=bool(row["gps_available"]),
        detection_info=detection_info,
    )
