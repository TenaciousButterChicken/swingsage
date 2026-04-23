# SwingSage server

FastAPI app that wraps the analysis pipeline (`scripts/analyze_swing.py`)
and exposes it to the web frontend in `../web`.

## Run

```powershell
# From repo root
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `POST /api/upload` — multipart, one `file` field. Returns `{job_id, filename}`.
- `WS  /api/ws/{job_id}` — streams stage/log/done/error events as JSON lines.
- `GET /api/health` — liveness probe.
- `GET /captures/...` — static serving of trimmed videos + keyframe JPGs
  written by the pipeline into `<repo>/captures/`.

## Pipeline stages emitted on the WebSocket

`decode` → `pose` → `trim` → `swingnet` → `metrics` → `coaching`

Each stage emits a `{type: "stage", name, label}` event when it starts.
Interstitial `{type: "log", message}` events carry timing + debug info.
The final `{type: "done", result}` carries the full analysis payload; any
exception emits `{type: "error", message}` and closes the socket.

## Dependencies

On top of the base repo requirements:

- `fastapi>=0.115`
- `uvicorn[standard]>=0.32`
- `python-multipart>=0.0.12`

`uvicorn[standard]` pulls in `websockets`, `httptools`, and `watchfiles`.
