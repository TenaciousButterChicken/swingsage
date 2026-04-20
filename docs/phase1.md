# Phase 1 — Repo, schema, VTrack watcher, smoke tests

## What shipped

| Layer | Component | File(s) |
|---|---|---|
| Repo plumbing | License (MIT), gitignore, pyproject (ruff + pytest config) | `LICENSE`, `.gitignore`, `pyproject.toml` |
| Folder structure | All Phase 2-5 packages stubbed | `capture/`, `inference/`, `analytics/`, `coach/`, `ui/` |
| Config | `.env.example` + platform-aware loader | `.env.example`, `capture/config.py` |
| Dependencies | Base, dev-mac, prod-windows, prod-windows-fallback | `requirements-*.txt` |
| Storage | SQLite schema (sessions, shots, swings, events) | `data/schema.sql` |
| Capture | Watchdog-based VTrack `ShotData` JSON ingestor | `capture/vtrack_watcher.py` |
| Fixtures | 4 fake VTrack JSON files + sample-video generator | `tests/fixtures/`, `scripts/generate_sample_video.py` |
| Tests | Imports, schema, watcher (one-shot, dedupe, raw-json passthrough) | `tests/test_*.py` |
| Setup scripts | Mac bootstrap (bash) + Windows bootstrap (PowerShell) | `scripts/setup_*.{sh,ps1}` |
| Docs | This file + `HANDOFF.md` | `docs/phase1.md`, `HANDOFF.md` |

## What did NOT ship (deferred to later phases)

- Camera capture (Phase 2)
- Any GPU inference (Phase 2/3)
- 4D-Humans / SwingNet / RTMPose integration (Phase 2/3)
- Ollama coaching loop (Phase 4)
- PyQt6 dashboard (Phase 5)

## How to verify on Mac

```bash
cd ~/swingsage
./scripts/setup_mac.sh          # creates .venv, installs deps, copies .env
source .venv/bin/activate
pytest                          # all green
python scripts/generate_sample_video.py   # creates the synthetic test video
python -m capture.vtrack_watcher          # watches the fixture folder; Ctrl-C to stop
```

The watcher will backfill the 4 fixture shots into `./data/runtime/swingsage.db`,
then sit idle waiting for new files. Add a JSON to `tests/fixtures/vtrack_shots/`
and confirm it gets ingested in real time.

Inspect the db with:

```bash
sqlite3 data/runtime/swingsage.db "SELECT vtrack_shot_id, club, ball_speed_mps FROM shots;"
```

## How to verify on Windows (when you pull)

```powershell
git pull
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
pytest
```

Then point `VTRACK_SHOTDATA_PATH` in `.env` at the real ShotData folder and:

```powershell
python -m capture.vtrack_watcher
```

Hit a shot on the simulator. Within ~1 second, you should see a log line and a
new row in the db.

## Known assumptions to verify in real-world Phase 2 use

1. **VTrack JSON field names** — the `_FIELD_MAP` in `capture/vtrack_watcher.py`
   is best-guess. On first real shot, dump the JSON and update the map.
2. **`shotID` uniqueness** — the dedupe relies on VTrack assigning a unique ID
   per shot. If it doesn't, switch the dedupe key to a hash of `raw_json`.
3. **File-write atomicity** — `on_modified` adds a 50ms grace before reading.
   If VTrack writes incrementally, may need a "wait until file size stable"
   check.
