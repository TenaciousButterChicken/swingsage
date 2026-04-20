"""Watch the VTrack ShotData folder, parse new JSON files, write to SQLite.

Two entry points:
    process_existing(folder, db_path) — one-shot scan, ideal for tests and backfill.
    watch(folder, db_path)            — long-running observer (Ctrl-C to stop).

Idempotent: the (UNIQUE) vtrack_shot_id column dedupes re-ingested shots.
On Mac (or with SWINGSAGE_USE_MOCK_VTRACK=true), defaults to the fixture folder
so the pipeline runs end-to-end without a real launch monitor.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from capture.config import REPO_ROOT, Config, load_config

log = logging.getLogger(__name__)

SCHEMA_PATH = REPO_ROOT / "data" / "schema.sql"


# ─── DB plumbing ──────────────────────────────────────────────────────
def _ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# ─── Shot parsing ─────────────────────────────────────────────────────
# VTrack's exact JSON schema is proprietary and not publicly documented.
# These keys are best-guess based on common launch-monitor conventions and
# MUST be verified against a real ShotData file on Windows. Unknown keys
# survive in raw_json so nothing is lost.
_FIELD_MAP: dict[str, str] = {
    "shotID": "vtrack_shot_id",
    "shotId": "vtrack_shot_id",
    "captureTime": "captured_at",
    "club": "club",
    "ballSpeedMps": "ball_speed_mps",
    "launchAngleDeg": "launch_angle_deg",
    "launchDirectionDeg": "launch_direction_deg",
    "backSpinRpm": "back_spin_rpm",
    "sideSpinRpm": "side_spin_rpm",
    "spinAxisDeg": "spin_axis_deg",
    "carryDistanceM": "carry_distance_m",
    "totalDistanceM": "total_distance_m",
    "peakHeightM": "peak_height_m",
    "descentAngleDeg": "descent_angle_deg",
    "clubSpeedMps": "club_speed_mps",
    "smashFactor": "smash_factor",
    "clubPathDeg": "club_path_deg",
    "faceAngleDeg": "face_angle_deg",
    "attackAngleDeg": "attack_angle_deg",
    "impactLocationXmm": "impact_location_x_mm",
    "impactLocationYmm": "impact_location_y_mm",
}


def _parse_shot(raw: dict, json_text: str) -> dict:
    """Map a raw VTrack shot JSON into a dict aligned with the shots table."""
    row: dict = {col: None for col in _FIELD_MAP.values()}
    for src, dest in _FIELD_MAP.items():
        if src in raw and raw[src] is not None:
            row[dest] = raw[src]

    # Required fields with sensible fallbacks
    if not row["vtrack_shot_id"]:
        row["vtrack_shot_id"] = f"unknown-{uuid.uuid4()}"
    if not row["captured_at"]:
        row["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    row["raw_json"] = json_text
    return row


def insert_shot(conn: sqlite3.Connection, row: dict, session_id: int | None = None) -> bool:
    """Insert a single parsed shot. Returns True if inserted, False if duplicate."""
    columns = ["session_id"] + list(row.keys())
    values = [session_id] + list(row.values())
    placeholders = ",".join("?" * len(values))
    sql = f"INSERT OR IGNORE INTO shots ({','.join(columns)}) VALUES ({placeholders})"
    cursor = conn.execute(sql, values)
    conn.commit()
    return cursor.rowcount > 0


# ─── File processing ──────────────────────────────────────────────────
def _ingest_file(path: Path, db_path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Skipping unreadable shot file %s: %s", path, e)
        return False

    row = _parse_shot(data, text)
    with _connect(db_path) as conn:
        inserted = insert_shot(conn, row)
    if inserted:
        log.info("Ingested shot %s from %s", row["vtrack_shot_id"], path.name)
    else:
        log.debug("Duplicate shot %s, skipped", row["vtrack_shot_id"])
    return inserted


def process_existing(folder: Path, db_path: Path) -> int:
    """Scan a folder once, ingest every *.json. Returns number of new shots inserted."""
    _ensure_db(db_path)
    if not folder.exists():
        log.warning("VTrack folder does not exist yet: %s", folder)
        return 0

    inserted = 0
    for path in sorted(folder.glob("*.json")):
        if _ingest_file(path, db_path):
            inserted += 1
    return inserted


# ─── Long-running observer ────────────────────────────────────────────
class _ShotEventHandler(FileSystemEventHandler):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _maybe_ingest(self, src_path: str) -> None:
        path = Path(src_path)
        if path.suffix.lower() != ".json":
            return
        # Tiny delay lets the writer finish flushing.
        time.sleep(0.05)
        _ingest_file(path, self.db_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_ingest(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_ingest(event.src_path)


def watch(folder: Path, db_path: Path, *, ingest_existing: bool = True) -> None:
    """Block the calling thread, watching `folder` for new ShotData JSON files."""
    _ensure_db(db_path)
    folder.mkdir(parents=True, exist_ok=True)

    if ingest_existing:
        n = process_existing(folder, db_path)
        log.info("Backfilled %d existing shots from %s", n, folder)

    observer = Observer()
    observer.schedule(_ShotEventHandler(db_path), str(folder), recursive=False)
    observer.start()
    log.info("Watching %s — Ctrl-C to stop", folder)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping watcher")
    finally:
        observer.stop()
        observer.join()


def main() -> None:
    cfg: Config = load_config()
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not cfg.vtrack_shotdata_path or str(cfg.vtrack_shotdata_path) == ".":
        raise SystemExit(
            "VTRACK_SHOTDATA_PATH not set and no mock fallback resolved. "
            "Set it in .env or enable SWINGSAGE_USE_MOCK_VTRACK=true."
        )

    log.info("Mode: %s", "mock" if cfg.use_mock_vtrack else "live")
    watch(cfg.vtrack_shotdata_path, cfg.db_path)


if __name__ == "__main__":
    main()
