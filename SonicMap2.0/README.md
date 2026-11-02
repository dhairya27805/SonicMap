# SonicMap

Music discovery powered by unsupervised ML (KMeans + PCA + cosine similarity)
over a curated Spotify audio-features dataset, with Spotify account login,
a private per-user **My Library** and **My Music Map**, external discovery,
SonicMap Playlists, and a persistent in-browser player built on the official
**Spotify Web Playback SDK**.

## Features

| Area | What it does |
|---|---|
| Global Music Map | 20,000-song dataset, 8-cluster KMeans, PCA map, cosine-similarity recommendations (unchanged) |
| Spotify OAuth | Authorization Code + PKCE; sessions and tokens persisted server-side (refresh token never leaves the server) |
| My Library (Phase 2) | Imports liked songs, playlists, top and recently played tracks |
| Feature engine (Phase 3) | Acquires the 9 audio features with provenance + confidence; never fabricates values |
| My Music Map (Phase 4) | Personal KMeans/PCA trained only on your eligible library tracks |
| Discover (Phase 5) | Recommendations from outside your library |
| SonicMap Playlists (Phase 6) | Generated, stored and editable playlists |
| **Player (Phase 7)** | Persistent bottom player: play/pause, next, previous, seek, volume/mute, shuffle, repeat, "Open in Spotify". Play from My Library, My Music Map, Discover (dataset search + Discover for you), mood previews and SonicMap Playlists |

## Project structure

```
backend/    FastAPI app: ML engines, Spotify OAuth, library, discovery, playlists, playback
  routers/playback.py           Phase 7 playback API
  services/playback_service.py  Phase 7 URI validation, availability checks, Spotify error mapping
  test_phase1.py … test_phase7.py
frontend/   React + Vite app
  src/api.js                    shared fetch helper (relative /api, proxied by Vite)
  src/player/                   PlayerProvider (single SDK instance), PlayerBar, PlayButton
  src/components/               DiscoverForYou (Phase 5 UI), SonicPlaylists (Phase 6 UI)
```

## Run SonicMap locally

### 1. Register a Spotify app

1. Open the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and create an app
   (Development Mode apps require the **app owner to have Spotify Premium**).
2. Add the Redirect URI `http://127.0.0.1:8000/api/auth/callback` (Spotify no longer accepts `localhost`).
3. Enable **Web API** and **Web Playback SDK**.
4. **Settings → User Management**: add every Spotify account that will log in (max 5 in Development Mode).
5. Copy the **Client ID** and **Client Secret**.

### 2. Backend (terminal 1)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
# edit .env: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI,
#            FRONTEND_URL=http://127.0.0.1:5173, SESSION_SECRET
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Frontend (terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173** (use `127.0.0.1`, not `localhost`, so the session cookie set by the
backend applies). The Vite dev server proxies `/api` to `http://127.0.0.1:8000`; override with
`VITE_BACKEND_URL` in `frontend/.env.local` (see `frontend/.env.example`).

Click **Connect Spotify**, then **Sync Library**, **Acquire features** and press ▶ on any track.

### 4. Tests and production build

```bash
cd backend
python test_phase1.py && python test_phase2.py && python test_phase3.py && python test_phase4.py \
  && python test_phase5.py && python test_phase6.py && python test_phase7.py

cd ../frontend
npm test          # Node built-in test runner (player track-ID helpers)
npm run build     # production build into dist/
```

> The backend tests write test users to the local SQLite DB (`backend/sonicmap.db`).

## Playback (Phase 7)

- The browser loads Spotify's Web Playback SDK **once**; a single `SonicMap Web Player` device is kept
  on `window` so React StrictMode, hot reloads and navigation never create duplicates. The player lives
  above the whole page (`PlayerProvider` in `App.jsx`) and persists while you move between sections.
- Pressing ▶ on a track queues the surrounding list (library, map recommendations, discovery results,
  playlist, …) starting at that track. The backend checks the first 20 tracks with
  `GET /v1/tracks/{id}?market=from_token`, removes ones Spotify reports as unavailable/region-locked and
  tells you how many were skipped. Tracks without a Spotify ID (e.g. iTunes/MusicBrainz discovery
  metadata) show ⊘ and are never sent to Spotify.
- Play/pause, next, previous, seek and volume are executed by the SDK; shuffle and repeat use the Web API
  (`/api/playback/shuffle`, `/api/playback/repeat`). Controls disable themselves when Spotify disallows them.
- The SDK needs a short-lived access token in the browser (Spotify requirement). It is fetched from
  `/api/playback/token` with `Cache-Control: no-store`; the refresh token and client secret stay server-side.

## Spotify API limitations (read this)

- **Playback requires Spotify Premium.** Spotify only streams in the browser for Premium accounts. Since the
  February 2026 Web API changes, `GET /me` no longer returns `product` for Development Mode apps, so SonicMap
  cannot know in advance. It starts the SDK and, if Spotify replies with `account_error` or
  `PREMIUM_REQUIRED`, the player shows **"Spotify Premium required"** with an **Open in Spotify** link.
  No preview clips or substitute audio are played.
- **Development Mode limits.** The app owner must have Premium, and at most 5 allow-listed users can log in.
  Non-allow-listed users get 403 errors from Spotify.
- **Browser support.** The SDK needs Encrypted Media Extensions (DRM). Desktop Chrome, Edge and Firefox work;
  some mobile/embedded browsers don't. SonicMap then shows "Browser not supported".
- **Feb 2026 endpoint changes handled:** playlist creation uses `POST /me/playlists`, items are added via
  `/playlists/{id}/items`, and availability checks use single-track lookups (the batch `GET /tracks` endpoint
  was removed for Development Mode apps).
- **`/v1/audio-features` and `/v1/recommendations` are restricted** (Nov 2024). SonicMap never relies on them.
- **Track availability drifts.** Dataset IDs may stop resolving in your market; such tracks are skipped and reported.
- **Rate limits.** Spotify 429 responses surface as "Spotify is rate-limiting requests right now"; availability
  results are cached for 6 hours per user to reduce calls.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SPOTIFY_CLIENT_ID` | Yes (for auth) | From the Spotify Developer Dashboard |
| `SPOTIFY_CLIENT_SECRET` | Yes (for auth) | From the Spotify Developer Dashboard; server-side only |
| `SPOTIFY_REDIRECT_URI` | Yes | Must exactly match a Redirect URI registered in the dashboard |
| `FRONTEND_URL` | Yes | Where the frontend runs; used for post-login redirects and CORS |
| `SESSION_SECRET` | Recommended | Random string |
| `DATABASE_URL` | No | Defaults to `sqlite:///sonicmap.db` |
| `VITE_BACKEND_URL` (frontend) | No | Dev-proxy target, default `http://127.0.0.1:8000` |
| `VITE_API_BASE` (frontend) | No | API base for builds served without a `/api` proxy, default `/api` |

## Deployment notes (not performed — local development only)

- Set `secure=True` on the session cookie in `backend/routers/auth.py` once served over HTTPS, and update
  `SPOTIFY_REDIRECT_URI` / `FRONTEND_URL` (and the dashboard redirect URI).
- Serve the built frontend behind a reverse proxy that forwards `/api` to the backend, or build with
  `VITE_API_BASE` pointing at the backend.
