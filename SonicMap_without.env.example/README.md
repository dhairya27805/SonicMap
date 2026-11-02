# SonicMap

Music discovery powered by unsupervised ML (KMeans + PCA + cosine similarity)
over a curated Spotify audio-features dataset, with Spotify account login,
automatic mood-based playlist creation, and in-browser playback via the
Spotify Web Playback SDK.

## What changed in this version

- **Spotify OAuth (Authorization Code + PKCE)** — users log in with their
  Spotify account; the account is linked automatically, no manual token entry.
- **CSV upload removed.** The app now runs only on the internal, curated
  dataset (`backend/data/dataset.csv`). The ML engine (KMeans clustering, PCA
  projection, cosine-similarity recommendations) is unchanged and untouched.
- **Mood playlists are real Spotify playlists.** Picking a mood and clicking
  "Create & play on Spotify" has the ML engine select tracks, matches them to
  real Spotify track IDs, creates a private playlist on the user's account,
  and (for Premium accounts) starts playback immediately in the browser via
  the Web Playback SDK. No song titles are ever typed manually.

## Project structure

```
backend/   FastAPI app: ML engine, Spotify OAuth, playlist/playback routes
frontend/  React + Vite app: music map, search, mood playlists, player
```

## Setup

### 1. Register a Spotify app

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   and create an app.
2. Add a Redirect URI that matches exactly what you'll run locally, e.g.
   `http://127.0.0.1:8000/api/auth/callback`.
3. Under API/SDK usage, make sure **Web API** and **Web Playback SDK** are
   enabled for the app.
4. Copy the **Client ID** and **Client Secret**.

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
# SPOTIFY_REDIRECT_URI, FRONTEND_URL, SESSION_SECRET

uvicorn main:app --reload --port 8000
```

The dataset at `backend/data/dataset.csv` must exist (it's already included).
On startup the server trains the KMeans/PCA model and prints a warning if
Spotify credentials are missing — the music map and search will still work
without Spotify configured, but login/playlists/playback will not.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (default `http://127.0.0.1:5173`). Click
**Connect Spotify** to log in, then generate a mood playlist.

## Spotify API limitations (read this)

- **Playback requires Spotify Premium.** The Web Playback SDK only streams
  audio for Premium accounts. Free-tier users can still log in and have
  SonicMap create real playlists on their account, but "Create & play on
  Spotify" will create the playlist and open a link to it in the Spotify app
  instead of playing in-browser.
- **`/v1/audio-features` and `/v1/recommendations` are restricted.** Spotify
  tightened access to these endpoints for most developer apps in November
  2024. Because of this, SonicMap does **not** attempt to pull live audio
  features for a user's own library or use Spotify's own recommendation
  engine. All clustering, PCA projection, and mood scoring run against the
  bundled dataset, which already contains real Spotify `track_id`s captured
  at dataset-build time. If your Spotify Developer app has been approved for
  extended quota and these endpoints, the recommendation engine could be
  extended to blend in fresh per-user audio features — that's out of scope
  here and would need a quota-extension request from Spotify.
- **Track availability drifts over time.** A `track_id` in the dataset may
  no longer resolve (regional licensing changes, catalog removals). Before
  adding tracks to a generated playlist, the backend calls Spotify's
  `/v1/tracks` endpoint to confirm each ID is still resolvable, and falls
  back to a title/artist search for any that aren't. Tracks that still can't
  be resolved are skipped and reported back in the API response
  (`unresolved_tracks`).
- **Rate limits.** Spotify's Web API enforces per-app rate limits (standard
  apps get a modest shared quota). Creating large playlists or doing many
  track-existence checks in a short window can hit `429 Too Many Requests`;
  the UI will surface this as a generic playlist-creation error.
- **In-memory sessions.** This implementation stores Spotify session/token
  data in an in-memory Python dict (`backend/state.py: SESSIONS`) for
  simplicity. This means sessions do not survive a server restart and will
  not work correctly behind multiple server processes/workers. For a real
  deployment, replace this with a shared store (Redis, a database table,
  or signed/encrypted cookies holding the tokens directly).

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SPOTIFY_CLIENT_ID` | Yes (for auth) | From the Spotify Developer Dashboard |
| `SPOTIFY_CLIENT_SECRET` | Yes (for auth) | From the Spotify Developer Dashboard |
| `SPOTIFY_REDIRECT_URI` | Yes | Must exactly match a Redirect URI registered in the dashboard |
| `FRONTEND_URL` | Yes | Where the frontend runs; used for post-login redirects and CORS |
| `SESSION_SECRET` | Recommended | Random string; reserved for signing session data if you move off in-memory sessions |

## Deployment notes

- Set `secure=True` on the session cookie in `backend/routers/auth.py` once
  served over HTTPS, and update `SPOTIFY_REDIRECT_URI` / `FRONTEND_URL` to
  your real domains (and register the new redirect URI in the Spotify
  Dashboard).
- Replace the in-memory `SESSIONS` dict with a real session store before
  running more than one backend worker/process.
- The frontend's `API` constant in `frontend/src/App.jsx` is currently
  hardcoded to `http://127.0.0.1:8000/api` — change it (or read from an env
  var) for production builds.
