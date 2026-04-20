"""Shared pytest fixtures for SwingSage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data" / "schema.sql"
FIXTURE_VTRACK_DIR = REPO_ROOT / "tests" / "fixtures" / "vtrack_shots"


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Empty SQLite db with the SwingSage schema applied."""
    db_path = tmp_path / "swingsage.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())
    return db_path


@pytest.fixture()
def fixture_vtrack_dir() -> Path:
    return FIXTURE_VTRACK_DIR
