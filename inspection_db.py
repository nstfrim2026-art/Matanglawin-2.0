"""
inspection_db.py - SQLite database for inspection records.

Provides persistent storage for all inspection metadata, enabling
querying, filtering, reporting, and dashboard statistics without
needing to scan the filesystem.

Database file: inspections.db (in project root, gitignored)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspections.db")


def _get_connection() -> sqlite3.Connection:
    """Create a new database connection with row factory enabled."""
    conn = sqlite3.connect(_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    """Initialize the database schema (idempotent)."""
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capture_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                crack_area INTEGER,
                estimated_length REAL,
                estimated_width REAL,
                bbox TEXT,
                camera_source TEXT,
                detection_threshold REAL,
                image_resolution TEXT,
                classification TEXT,
                image_path TEXT,
                overlay_path TEXT,
                metadata_path TEXT,
                report_path TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_timestamp
            ON inspections(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_capture_id
            ON inspections(capture_id)
        """)
        # Migrate: add 'notes' column if not present
        cursor = conn.execute("PRAGMA table_info(inspections)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "notes" not in columns:
            conn.execute(
                "ALTER TABLE inspections ADD COLUMN notes TEXT DEFAULT ''"
            )
        conn.commit()
    finally:
        conn.close()


def insert_inspection(data: Dict[str, Any]) -> int:
    """
    Insert a new inspection record.

    Args:
        data: dict with keys matching the table columns (capture_id required).

    Returns:
        The row id of the inserted record.
    """
    conn = _get_connection()
    try:
        bbox_value = data.get("bbox")
        if bbox_value is not None and not isinstance(bbox_value, str):
            bbox_value = json.dumps(bbox_value)

        cursor = conn.execute(
            """
            INSERT INTO inspections (
                capture_id, timestamp, confidence, crack_area,
                estimated_length, estimated_width, bbox,
                camera_source, detection_threshold, image_resolution,
                classification, image_path, overlay_path,
                metadata_path, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["capture_id"],
                data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                data.get("confidence"),
                data.get("crack_area"),
                data.get("estimated_length"),
                data.get("estimated_width"),
                bbox_value,
                data.get("camera_source"),
                data.get("detection_threshold"),
                data.get("image_resolution"),
                data.get("classification"),
                data.get("image_path"),
                data.get("overlay_path"),
                data.get("metadata_path"),
                data.get("report_path"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_inspections(
    page: int = 1,
    per_page: int = 20,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query inspections with pagination and optional filtering.

    Returns:
        dict with keys: items (list of dicts), total (int), page, per_page
    """
    conn = _get_connection()
    try:
        conditions = []
        params: List[Any] = []

        if search:
            conditions.append(
                "(capture_id LIKE ? OR classification LIKE ?)"
            )
            like_term = f"%{search}%"
            params.extend([like_term, like_term])

        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("timestamp <= ?")
            params.append(date_to)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # Get total count
        count_sql = f"SELECT COUNT(*) FROM inspections {where_clause}"
        total = conn.execute(count_sql, params).fetchone()[0]

        # Get paginated results
        offset = (page - 1) * per_page
        query_sql = f"""
            SELECT * FROM inspections {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(
            query_sql, params + [per_page, offset]
        ).fetchall()

        items = [dict(row) for row in rows]

        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
        }
    finally:
        conn.close()


def get_inspection_by_id(capture_id: str) -> Optional[Dict[str, Any]]:
    """Get a single inspection record by capture_id."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM inspections WHERE capture_id = ?",
            (capture_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_inspection(capture_id: str) -> bool:
    """
    Delete an inspection record by capture_id.

    Returns:
        True if a record was deleted, False if not found.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM inspections WHERE capture_id = ?",
            (capture_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_inspection_notes(capture_id: str, notes: str) -> bool:
    """
    Update the notes field for an inspection record.

    Returns:
        True if a record was updated, False if not found.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "UPDATE inspections SET notes = ? WHERE capture_id = ?",
            (notes, capture_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_summary_stats() -> Dict[str, Any]:
    """
    Get dashboard summary statistics.

    Returns:
        dict with: total_inspections, total_confirmed_cracks,
        average_confidence, largest_crack, average_crack_size,
        inspections_today
    """
    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_inspections,
                COUNT(CASE WHEN classification IS NOT NULL THEN 1 END) as total_confirmed_cracks,
                AVG(confidence) as average_confidence,
                MAX(crack_area) as largest_crack,
                AVG(crack_area) as average_crack_size
            FROM inspections
        """).fetchone()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_row = conn.execute(
            "SELECT COUNT(*) as count FROM inspections WHERE timestamp LIKE ?",
            (f"{today}%",),
        ).fetchone()

        return {
            "total_inspections": row["total_inspections"] or 0,
            "total_confirmed_cracks": row["total_confirmed_cracks"] or 0,
            "average_confidence": round(row["average_confidence"] or 0.0, 4),
            "largest_crack": row["largest_crack"] or 0,
            "average_crack_size": round(row["average_crack_size"] or 0.0, 1),
            "inspections_today": today_row["count"] or 0,
        }
    finally:
        conn.close()


def get_all_inspections() -> List[Dict[str, Any]]:
    """
    Fetch all inspection records without pagination (for export).

    Returns:
        List of all inspection records as dicts, ordered by timestamp descending.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM inspections ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
