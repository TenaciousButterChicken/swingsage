# SwingSage web

Vite + React + TypeScript frontend. Dark warm-black theme with champagne
gold + fairway green accents.

## Run

```powershell
# First time
npm install

# Dev server (hot reload). Proxies /api and /captures to 127.0.0.1:8000.
npm run dev

# Production bundle
npm run build
```

The FastAPI server (`../server`) must be running for the upload and
WebSocket calls to work.

## Structure

- `src/main.tsx` — React entry
- `src/App.tsx` — top-level phase machine (upload → processing → results)
- `src/components/Brand.tsx` — header with runtime pills
- `src/components/UploadView.tsx` — drag-drop upload card
- `src/components/ProcessingView.tsx` — WebSocket progress tracker
- `src/components/ResultsView.tsx` — video, keyframes, metrics, coaching
- `src/lib/api.ts` — fetch + WebSocket helpers
- `src/lib/types.ts` — shapes mirrored from `server/main.py`

## Design tokens

Colors live in `tailwind.config.js`:

- `ink-{100..950}` — warm near-black neutrals
- `champagne-{50..600}` — gold accent
- `fairway-{400..600}` — success / in-range green
- `ember-{400..600}` — out-of-range red

Fonts:

- Display — Fraunces (serif, italic supported)
- Sans — Geist
- Mono — Geist Mono

Loaded from Google Fonts in `index.html`.
