# SonicMap — Handover & Architectural Documentation

## Phase 1 Handover Checkpoint

### 1. Database Architecture & Schema
The project now uses SQLAlchemy 2.0 and Alembic to handle schema migrations and multi-database support.
- **Local Engine**: SQLite (`sqlite:///sonicmap.db` in `backend/`).
- **Production Engine**: PostgreSQL (Render provides `postgres://` which is automatically normalized to `postgresql://`).
- **Automatic Initialization**: On backend startup via FastAPI lifespan hook (`init_db()`), and version-controlled via Alembic (`alembic upgrade head`).

#### Database Models Overview (`backend/models.py`)
1. **`User`**:
   - `id` (PK, Int), `spotify_id` (Unique, Indexed), `display_name`, `email`, `product`, `avatar_url`, `created_at`, `updated_at`.
2. **`UserToken`**:
   - `id` (PK, Int), `user_id` (FK -> `users.id`, Unique, Cascade), `access_token` (Text), `refresh_token` (Text), `expires_at` (Float), `scope`, timestamps.
3. **`UserSession`**:
   - `session_id` (PK, String 128), `user_id` (FK -> `users.id`, Cascade), `expires_at` (Float), `created_at`.
4. **`Track`**:
   - `id` (PK, Int), `spotify_id` (Unique, Nullable), `title`, `artist`, `album`, `preview_url`, `image_url`, `duration_ms`.
   - 9 Audio Features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
5. **`AudioFeatureProvenance`**:
   - `id` (PK, Int), `track_id` (FK -> `tracks.id`), `feature_name`, `source` (`spotify_api`, `dataset_csv`, `librosa`, `ml_estimation`), `confidence` (0.0 - 1.0), `notes`.
6. **`UserLibraryTrack`**:
   - `id` (PK, Int), `user_id` (FK -> `users.id`), `track_id` (FK -> `tracks.id`), `source` (`liked`, `playlist`, `top_short`, `recently_played`), `added_at`.
   - UniqueConstraint on `(user_id, track_id, source)` guaranteeing user library isolation.
7. **`Playlist`**:
   - `id` (PK, Int), `user_id` (FK -> `users.id`), `name`, `description`, `playlist_type` (`cluster`, `similar`, `mood`, `energy`, `discovery`, `custom`), `is_public`, `spotify_playlist_id`.
8. **`PlaylistItem`**:
   - `id` (PK, Int), `playlist_id` (FK -> `playlists.id`), `track_id` (FK -> `tracks.id`), `position`, `added_at`.
   - UniqueConstraint on `(playlist_id, position)`.
9. **`PersonalModelMetadata`**:
   - `id` (PK, Int), `user_id` (FK -> `users.id`, Unique), `cluster_count`, `silhouette_score`, `track_count`, `variance_ratio_pca1`, `variance_ratio_pca2`, `model_version`, `last_trained_at`.

---

### 2. Environment Variables (`backend/.env`)

| Variable | Description | Default / Example |
|---|---|---|
| `DATABASE_URL` | Database connection string | `sqlite:///sonicmap.db` (local) or PostgreSQL URL |
| `SPOTIFY_CLIENT_ID` | Spotify developer client ID | Provided by Spotify Dashboard |
| `SPOTIFY_CLIENT_SECRET` | Spotify developer client secret | Kept server-side only |
| `SPOTIFY_REDIRECT_URI` | Spotify OAuth callback endpoint | `http://127.0.0.1:8000/api/auth/callback` |
| `FRONTEND_URL` | Frontend origin for redirect & CORS | `http://127.0.0.1:5173` |
| `SESSION_SECRET` | Secret key used to sign session cookies | Random hex string |

---

### 3. Spotify Scopes & API Endpoints

#### Required Spotify OAuth Scopes
- `user-read-email`: Retrieve user profile email and ID.
- `user-read-private`: Retrieve account subscription tier (`product`).
- `streaming`, `user-read-playback-state`, `user-modify-playback-state`: Web Playback SDK control.
- `playlist-modify-public`, `playlist-modify-private`: Generate playlists on user account.
- `user-library-read`: Import saved / liked tracks (`/v1/me/tracks`).
- `playlist-read-private`, `playlist-read-collaborative`: Import user's personal playlists (`/v1/me/playlists`).
- `user-top-read`: Import user's top affinity tracks (`/v1/me/top/tracks`).
- `user-read-recently-played`: Import user's recent listening history (`/v1/me/player/recently-played`).

#### API Caveats & Unavailable Endpoints
- **Audio Features & Recommendations**: Spotify deprecated public Web API access to `/v1/audio-features` and `/v1/recommendations` in Nov 2024 for non-whitelisted apps. SonicMap intentionally does not rely on these endpoints; audio features will be gathered via the modular feature acquisition engine in Phase 3.
- **Recently Played Limits**: `/v1/me/player/recently-played` returns at most the last 50 tracks played and requires active history on the account.
- **Top Tracks Limits**: `/v1/me/top/tracks` returns tracks computed over Spotify's affinity affinity window (short, medium, long term); it does not provide lifetime listening history.

---

### 4. Library API Endpoints Reference (`routers/library.py`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/library/sync` | Starts asynchronous background library import. |
| `GET` | `/api/library/status` | Polling endpoint for import status, progress %, and error details. |
| `GET` | `/api/library/tracks` | Paginated tracks with `q` search and `source` filter (`liked`, `playlist`, `recently_played`, `top_tracks`). |
| `GET` | `/api/library/stats` | Aggregated count of imported tracks by source and playlists. |
| `GET` | `/api/library/playlists` | List of imported Spotify playlists for the user. |

---

### 5. Phase 3: Audio Feature Acquisition Engine

Implemented in [backend/services/feature_engine_service.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/services/feature_engine_service.py):
- **Tier 1 (Reference Catalog)**: Exact Spotify ID match (confidence `1.0`) and fuzzy normalized title/artist/duration match (confidence `0.85` - `0.88`) against verified open reference catalog (`dataset-source/dataset.csv`). Ambiguous matches are strictly rejected.
- **Tier 2 (Librosa Signal Analysis)**: Direct physical acoustic measurement of `tempo` (autocorrelation BPM) and `loudness` (RMS dB) on legal 30s preview URLs (`preview_url`). No fabrication of subjective psychoacoustic features (`valence`, `danceability`).
- **Tier 3 (ML Estimation Interface)**: `MLEstimationService` interface reporting honest availability status when training datasets are absent.
- **Provenance Tracking**: In `audio_feature_provenance` table with `match_confidence`, `feature_confidence`, `acquisition_type` (`matched`, `measured`, `estimated`), and acquisition timestamp.
- **Clustering Gate**: Tracks must satisfy all 9 features with confidence $\ge 0.75$ and valid domain bounds before entering personal KMeans clustering (`check_clustering_eligibility()`). Ineligible tracks remain in the library with `null` features.

#### Feature Engine API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/library/feature-status` | Overview of library feature readiness, clustering eligibility counts, and ML estimator status. |
| `POST` | `/api/library/features/acquire` | Dispatches asynchronous background feature acquisition job for user's library. |

---

### 6. Phase 4: Personalized ML Engine & My Music Map

Implemented in [backend/services/personal_ml_service.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/services/personal_ml_service.py) and [backend/routers/personal.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/routers/personal.py):
- **User Dataset Isolation**: Models are fit exclusively on tracks from `UserLibraryTrack` belonging to the authenticated user.
- **Eligibility Filtering**: Reuses Phase 3 `check_clustering_eligibility` requiring 9 features, confidence >= 0.75, and domain bounds. Ineligible tracks remain in library with null features.
- **Adaptive KMeans**: `select_best_k` evaluates silhouette score across valid candidate K values (from 2 up to min(8, n_samples // 2)). Strongly rejects singleton clusters and degenerate empty partitions.
- **PCA 2D Projection**: Reduces standardized 9-dimensional audio features to 2D coordinates `(x, y)` independently for each user.
- **Semantic Cluster Labels**: Dynamically assigns musically descriptive names (e.g. "High Energy & Dance Beats", "Acoustic & Mellow", "Melancholic & Reflective") based on positive/negative feature z-score deviations.
- **Cosine Similarity Recommendations**: Computes normalized similarity percentages strictly within the user's private eligible collection, guaranteed to exclude the queried track.
- **Lifecycle & Model Caching**: In-memory caching with persistent database diagnostics in `PersonalModelMetadata` (`cluster_count`, `silhouette_score`, `track_count`, `variance_ratio_pca1`, `variance_ratio_pca2`, `model_version`, `last_trained_at`). Automatically detects library staleness when tracks or features change.

#### Personal ML API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/personal/status` | Model training status, eligible vs awaiting counts, cluster summary, and diagnostics. |
| `GET` | `/api/personal/map` | 2D coordinates `(x, y)` and cluster assignments for user's library map. |
| `GET` | `/api/personal/clusters` | List of clusters with names, counts, and top characteristic features. |
| `GET` | `/api/personal/recommendations/{track_id}` | Cosine similarity recommendations strictly within user's library. |
| `POST` | `/api/personal/train` | Explicitly triggers personal model retraining/refresh. |

---

### 7. Exact Commands to Run the Application

#### Backend
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
# (Optional) Run migrations manually
alembic upgrade head
# Start server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### Running Tests
```powershell
cd backend
1..7 | ForEach-Object { .\.venv\Scripts\python.exe "test_phase$_.py" }
cd ..\frontend
npm test
npm run build
```

#### Frontend
```powershell
cd frontend
npm install
npm run dev   # http://127.0.0.1:5173, proxies /api -> http://127.0.0.1:8000
```

---

### 4. Verification & Status of Components
- **Database Initialization**: Initialized and verified via Alembic and FastAPI lifespan.
- **Models**: Created and verified with ForeignKeys and Cascades.
- **Authentication**: Fully preserved. Sessions and tokens now persist in SQLite/PostgreSQL, protecting refresh tokens.
- **Global Music Map**: Untouched 20,000-song dataset, 8-cluster KMeans, PCA, and cosine similarity confirmed 100% operational (tested via HTTP and unit test).

---

### 8. Phase 7: Music Playback & Integration

#### Architecture
- **Browser = Spotify Connect device.** `PlayerProvider` (wraps the whole app in `App.jsx`) loads
  `https://sdk.scdn.co/spotify-player.js` once and creates one `Spotify.Player("SonicMap Web Player")`, stored in
  `window.__SONICMAP_PLAYER__`. SDK events are forwarded to React state. `usePlayer()` exposes
  `playTracks(tracks, startIndex, label)`, `togglePlay`, `next`, `previous`, `seek`, `setVolume`, `toggleShuffle`,
  `cycleRepeat`, `phase`, `state`, `message`.
- **Starting playback** requires the Web API: the frontend sends Spotify IDs to `PUT /api/playback/play`; the backend
  validates, verifies availability (first 20 tracks, cached 6 h per user), rotates the queue to the chosen track,
  and calls `PUT /v1/me/player/play?device_id=…`.
- **In-session controls** (pause/resume, next, previous, seek, volume) are SDK calls; shuffle and repeat use
  `/api/playback/shuffle` and `/api/playback/repeat`.
- **Track identity**: `getSpotifyId()` (`src/player/trackUtils.js`) accepts `spotify_id` (library, map, discovery,
  playlists), `track_id` (global dataset) or `spotify:track:` URIs. Anything else is shown as not on Spotify.

#### Playback API
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/playback/status` | `{authenticated, product, premium (true/false/null), missing_scopes, sdk_supported, reason}` |
| `GET` | `/api/playback/token` | Short-lived access token for the SDK (`no-store`); 401 not connected, 403 known non-Premium |
| `PUT` | `/api/playback/play` | Body `{device_id, uris[], offset_index, position_ms}` or `{device_id, context_uri}` → `{queued, start_uri, unavailable[], invalid[], unverified}` |
| `PUT` | `/api/playback/pause?device_id=` | Pause via Web API |
| `PUT` | `/api/playback/shuffle?state=&device_id=` | Toggle shuffle |
| `PUT` | `/api/playback/repeat?state=off\|track\|context&device_id=` | Repeat mode |
| `POST` | `/api/playback/create-mood-playlist` | Unchanged flow; now uses Feb 2026 playlist endpoints |

Errors use `detail: {code, message}` with codes `not_connected`, `premium_required`, `device_not_found`,
`device_required`, `no_tracks`, `no_playable_tracks`, `invalid_uri`, `rate_limited`, `spotify_auth`, `restricted`,
`spotify_error`, `spotify_unreachable`.

#### Known limitations
- Real audio playback was not exercised against a live Premium account in the build environment (no Spotify
  credentials there); the SDK lifecycle and every API path are covered by tests and by browser checks of the
  not-connected, session-expired and unsupported-browser states.
- Premium status is usually unknown before the first SDK/API response (Spotify removed `product` from `GET /me`).
- Tracks beyond the first 20 in a queue are not pre-verified; Spotify skips unplayable ones itself.
- The SDK requires DRM (EME); unsupported browsers get an explicit message.
- Discovery tracks saved into SonicMap Playlists are metadata-only `Track` rows (no audio features).

