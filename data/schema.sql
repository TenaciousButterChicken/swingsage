-- SwingSage SQLite schema
-- Apply with: sqlite3 swingsage.db < data/schema.sql
-- Idempotent: safe to re-apply.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ─── sessions ─────────────────────────────────────────────────────────
-- One row per practice session.
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at      TEXT,
    label         TEXT,            -- e.g. "Sat morning irons"
    notes         TEXT
);

-- ─── shots ────────────────────────────────────────────────────────────
-- One row per VTrack shot. Mirrors the launch-monitor JSON.
-- Field names follow VTrack JSON conventions (camelCase preserved as snake_case).
-- raw_json holds the unparsed payload for forensic recovery.
CREATE TABLE IF NOT EXISTS shots (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    vtrack_shot_id          TEXT    UNIQUE,        -- VTrack's own UUID; dedupe key
    captured_at             TEXT    NOT NULL,
    club                    TEXT,                  -- "7i", "Driver", etc.

    -- Ball flight
    ball_speed_mps          REAL,
    launch_angle_deg        REAL,
    launch_direction_deg    REAL,
    back_spin_rpm           REAL,
    side_spin_rpm           REAL,
    spin_axis_deg           REAL,
    carry_distance_m        REAL,
    total_distance_m        REAL,
    peak_height_m           REAL,
    descent_angle_deg       REAL,

    -- Club delivery
    club_speed_mps          REAL,
    smash_factor            REAL,
    club_path_deg           REAL,
    face_angle_deg          REAL,
    attack_angle_deg        REAL,
    impact_location_x_mm    REAL,
    impact_location_y_mm    REAL,

    -- Provenance
    raw_json                TEXT    NOT NULL,
    ingested_at             TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_shots_session ON shots(session_id);
CREATE INDEX IF NOT EXISTS idx_shots_captured ON shots(captured_at);

-- ─── swings ───────────────────────────────────────────────────────────
-- One row per captured swing video associated with a shot.
-- Pose tensor lives on disk (parquet/npz) — only the path is stored.
CREATE TABLE IF NOT EXISTS swings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id         INTEGER REFERENCES shots(id) ON DELETE CASCADE,
    video_path      TEXT    NOT NULL,
    pose_path       TEXT,                          -- 2D keypoints (T, J, 3)
    mesh_path       TEXT,                          -- 3D SMPL params from 4D-Humans
    camera_id       TEXT,                          -- which camera if multi-cam
    fps             REAL,
    frame_count     INTEGER,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_swings_shot ON swings(shot_id);

-- ─── events ───────────────────────────────────────────────────────────
-- 8 swing events from SwingNet, plus any custom annotations.
-- event_type values: address, takeaway, mid_backswing, top, mid_downswing,
-- impact, mid_followthrough, finish.
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    swing_id        INTEGER NOT NULL REFERENCES swings(id) ON DELETE CASCADE,
    event_type      TEXT    NOT NULL,
    frame_number    INTEGER NOT NULL,
    confidence      REAL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_swing ON events(swing_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

-- ─── analyses ─────────────────────────────────────────────────────────
-- Persistent record of every swing video SwingSage has analyzed. The
-- full result payload is stored as JSON so this table survives schema
-- changes to the inner metrics/coaching shapes without migration. A
-- nullable shot_id links to the VTrack shot paired at analysis time.
CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT    NOT NULL UNIQUE,         -- upload's UUID
    shot_id         INTEGER REFERENCES shots(id) ON DELETE SET NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    video_path      TEXT    NOT NULL,                -- captures/uploads/<id>.mp4
    trim_dir        TEXT,                            -- captures/<stem>_trim
    result_json     TEXT    NOT NULL                 -- the full job.result dict
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_shot ON analyses(shot_id);
