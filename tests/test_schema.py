"""Verify the SQLite schema applies cleanly and exposes the expected tables."""

import sqlite3
from pathlib import Path

EXPECTED_TABLES = {"sessions", "shots", "swings", "events"}


def test_schema_creates_expected_tables(tmp_db: Path) -> None:
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    actual = {r[0] for r in rows}
    assert EXPECTED_TABLES.issubset(actual), f"Missing tables: {EXPECTED_TABLES - actual}"


def test_shots_unique_constraint_dedupes(tmp_db: Path) -> None:
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO shots (vtrack_shot_id, captured_at, raw_json) VALUES (?, ?, ?)",
            ("dup-1", "2026-04-20T00:00:00Z", "{}"),
        )
        cursor = conn.execute(
            "INSERT OR IGNORE INTO shots (vtrack_shot_id, captured_at, raw_json) VALUES (?, ?, ?)",
            ("dup-1", "2026-04-20T00:00:00Z", "{}"),
        )
        assert cursor.rowcount == 0
        count = conn.execute("SELECT COUNT(*) FROM shots WHERE vtrack_shot_id='dup-1'").fetchone()[0]
        assert count == 1
