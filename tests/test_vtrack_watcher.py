"""End-to-end smoke test for the VTrack watcher (one-shot scan path).

Doesn't exercise the long-running observer — that needs filesystem events
and is platform-flaky in CI. The `process_existing` path covers the same
parsing + dedupe logic.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from capture.vtrack_watcher import process_existing


def test_process_existing_ingests_all_fixture_shots(tmp_db: Path, fixture_vtrack_dir: Path) -> None:
    inserted = process_existing(fixture_vtrack_dir, tmp_db)
    expected_count = len(list(fixture_vtrack_dir.glob("*.json")))
    assert inserted == expected_count

    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT vtrack_shot_id, club, ball_speed_mps FROM shots ORDER BY id").fetchall()

    assert len(rows) == expected_count
    for shot_id, club, ball_speed in rows:
        assert shot_id.startswith("fixture-")
        # Sparse fixture has minimal fields; others should at least have a club.
        if not shot_id.endswith("-sparse"):
            assert club is not None
            assert ball_speed is not None


def test_process_existing_is_idempotent(tmp_db: Path, fixture_vtrack_dir: Path) -> None:
    first = process_existing(fixture_vtrack_dir, tmp_db)
    second = process_existing(fixture_vtrack_dir, tmp_db)
    assert first > 0
    assert second == 0


def test_unknown_fields_survive_in_raw_json(tmp_path: Path, tmp_db: Path) -> None:
    shot = {
        "shotID": "novel-shot",
        "captureTime": "2026-04-20T15:00:00Z",
        "club": "5i",
        "ballSpeedMps": 60.0,
        "weirdFutureField": {"nested": [1, 2, 3]},
    }
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "novel.json").write_text(json.dumps(shot))

    process_existing(drop, tmp_db)

    with sqlite3.connect(tmp_db) as conn:
        raw = conn.execute(
            "SELECT raw_json FROM shots WHERE vtrack_shot_id=?", ("novel-shot",)
        ).fetchone()[0]

    payload = json.loads(raw)
    assert payload["weirdFutureField"]["nested"] == [1, 2, 3]
