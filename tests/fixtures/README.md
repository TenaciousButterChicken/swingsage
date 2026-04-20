# Test fixtures

## `vtrack_shots/`

Fake VTrack `ShotData` JSON files. Field names are best-guess based on common
launch-monitor conventions and **must be verified against a real VTrack output
on Windows** before Phase 2. Update `_FIELD_MAP` in `capture/vtrack_watcher.py`
if real field names differ.

Includes:

- `shot_001.json` — 7-iron, slight pull-draw
- `shot_002.json` — Driver, push with cut spin
- `shot_003.json` — PW pitch, high launch
- `shot_004_minimal.json` — sparse JSON to exercise the "missing fields" code path

## `videos/`

This folder is gitignored except for `sample_synthetic.mp4`, which is generated
on demand by `scripts/generate_sample_video.py`:

```bash
python scripts/generate_sample_video.py
```

The synthetic clip is a 2-second 30 fps colored test pattern — useful for
exercising the video-decode path without committing a real swing video.

For real swing footage, drop CC-licensed videos into this folder by hand. They
will not be committed. Suggested sources:

- USGA / R&A public-domain instructional footage
- Creative Commons golf tutorials (verify the specific license per video)
- Your own iPhone footage (best — no licensing concerns at all)
