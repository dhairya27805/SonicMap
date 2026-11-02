# SonicMap — Project Progress Tracker

## Phase Overview & Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Existing Project Audit and Architecture | **Complete** |
| **Phase 1** | Foundation and Database Architecture | **Complete** |
| **Phase 2** | Spotify Library Import | **Complete** |
| **Phase 3** | Nine-Feature Acquisition Engine | **Complete** |
| **Phase 4** | Personalized KMeans, PCA and My Music Map | **Complete** |
| **Phase 5** | External Music Discovery and Recommendations | Not Started |
| **Phase 6** | Playlist Generation and Storage | Not Started |
| **Phase 7** | Music Player and Playback-Provider Integration | Not Started |
| **Phase 8** | UI Integration, Testing, Security and Deployment | Not Started |

---

## Detailed Milestone Log

### Phase 0: Project Audit & Architecture
- Inspected existing codebase: FastAPI backend, 20,000-song dataset (`backend/data/dataset.csv`), 8-cluster KMeans & PCA visualization, Plotly frontend.
- Documented existing architecture, limitations (in-memory sessions, no persistence, global ML only), and requirements.

### Phase 1: Foundation and Database Architecture
- **Persistent Database Configuration**:
  - Integrated SQLAlchemy 2.0 with engine configuration in [backend/database.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/database.py).
  - Configured `DATABASE_URL` environment variable support (SQLite `sqlite:///sonicmap.db` for local dev; automatic `postgres://` to `postgresql://` conversion for Render production).
  - Setup Alembic database migrations (`backend/alembic`) and generated initial migration (`f68c67001563_initial_models_schema.py`).
- **Relational Data Models** in [backend/models.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/models.py):
  - `User`: Primary user identity, Spotify ID, profile metadata, timestamps.
  - `UserToken`: Isolated encrypted/secure Spotify OAuth tokens (access & refresh tokens, expiration epoch, scopes).
  - `UserSession`: Database-persisted session store with expiration timestamps.
  - `Track`: Canonical audio track catalog with the 9 core audio features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`), avoiding metadata duplication.
  - `AudioFeatureProvenance`: Feature provenance tracking with origin source (`spotify_api`, `dataset_csv`, `librosa`, `ml_estimation`) and confidence rating.
  - `UserLibraryTrack`: User's imported tracks (`liked`, `playlist`, `top_short`, `recently_played`) isolating private libraries with unique constraints per user.
  - `Playlist` & `PlaylistItem`: Native SonicMap playlists with positional ordering and multi-type categorization (`cluster`, `similar`, `mood`, `energy`, `discovery`, `custom`).
  - `PersonalModelMetadata`: Metadata for personalized KMeans and PCA models (clusters, silhouette score, explained variance ratios).
- **Authentication & Session Persistence**:
  - Refactored [backend/routers/auth.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/routers/auth.py) `get_session`, `get_valid_access_token`, `callback`, and `logout` to persist sessions and tokens to the database with safe fallback.
  - Refresh tokens and secrets remain protected server-side and are never sent to the client.
- **Preserved Existing Features**:
  - Global Music Map, 20,000-song dataset, 8-cluster KMeans, PCA, and cosine similarity remain 100% untouched and functional.
- **Automated Verification**:
  - Added unit test suite in [backend/test_phase1.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/test_phase1.py) (4 tests passed: DB init, user/token creation, user data isolation, global ML endpoints).

### Phase 2: Spotify Library Import
- **Modular Spotify Library Import Service**:
  - Implemented [backend/services/spotify_import_service.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/services/spotify_import_service.py) supporting multi-source data ingestion:
    - Saved / Liked tracks (`/v1/me/tracks`) with pagination.
    - User Playlists and their tracks (`/v1/me/playlists` and `/v1/playlists/{id}/tracks`).
    - Recently played tracks (`/v1/me/player/recently-played`).
    - Top tracks (`/v1/me/top/tracks` over medium_term).
  - Rate limiting handling (`429 Retry-After`) and automatic exponential backoff.
  - Partial failure resilience (individual endpoint permissions/deprecation errors logged without wiping or failing successful data).
  - Canonical track deduplication via `_upsert_track` avoiding redundant metadata storage.
  - User private library associations stored in `UserLibraryTrack` with isolated unique constraints.
- **Library API Endpoints**:
  - Implemented [backend/routers/library.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/routers/library.py):
    - `POST /api/library/sync`: Triggers asynchronous background library import job.
    - `GET /api/library/status`: Polls progress percentage, status (`idle` | `running` | `completed`), and statistics.
    - `GET /api/library/tracks`: Paginated, source-filtered (`liked`, `playlist`, `recently_played`, `top_tracks`), and search-indexed library tracks.
    - `GET /api/library/stats`: Aggregated summary statistics of user's personal collection.
    - `GET /api/library/playlists`: List of imported user playlists.
- **Frontend "My Library" UI**:
  - Added dedicated **My Library** section to [frontend/src/App.jsx](file:///c:/Users/l/Downloads/SonicMap_without.env.example/frontend/src/App.jsx) and sidebar navigation.
  - Features: real-time background sync progress bar, source filter buttons, title/artist search, album art display, empty and loading states.
- **Testing & Verification**:
  - Created [backend/test_phase2.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/test_phase2.py) with 6 comprehensive tests covering deduplication, pagination, playlist linking, recent/top tracks, partial error resilience, and user isolation.
  - Re-verified Phase 1 tests (`test_phase1.py`) and Global Music Map live endpoints with 0 regressions.

### Phase 3: Nine-Feature Acquisition Engine
- **Research & Strategy (Phase 3A)**:
  - Documented audit in [PHASE3_SOURCE_AUDIT.md](file:///c:/Users/l/Downloads/SonicMap_without.env.example/PHASE3_SOURCE_AUDIT.md).
- **Engine Implementation (Phase 3B)**:
  - Implemented [backend/services/feature_engine_service.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/services/feature_engine_service.py) with 3 modular acquisition tiers:
    - **Tier 1 (Reference Catalog)**: Exact Spotify ID matching and normalized metadata matching (with duration disambiguation and ambiguity rejection) against verified open reference catalog (`dataset-source/dataset.csv`).
    - **Tier 2 (Librosa Audio Analysis)**: Physical signal extraction for `tempo` (BPM) and `loudness` (RMS dB) on legal 30s audio previews (`preview_url`). No fabrication of psychoacoustic features.
    - **Tier 3 (ML Estimation Interface)**: `MLEstimationService` with honest availability reporting when verified training sets are absent.
  - **Feature Provenance & Confidence**:
    - Enhanced `AudioFeatureProvenance` model with `match_confidence`, `feature_confidence`, and `acquisition_type` (`matched`, `measured`, `estimated`).
    - Separated match accuracy from feature accuracy.
  - **Clustering Eligibility**:
    - Strict gating in `check_clustering_eligibility()`: requires all 9 features present, each with confidence $\ge 0.75$, and valid domain bounds. Ineligible tracks remain safely visible in user library.
  - **API Endpoints**:
    - `GET /api/library/feature-status`: Provides library clustering readiness metrics and ML model status.
    - `POST /api/library/features/acquire`: Asynchronous background feature acquisition job.
- **Testing & Verification**:
  - Implemented 7 tests in [backend/test_phase3.py](file:///c:/Users/l/Downloads/SonicMap_without.env.example/backend/test_phase3.py): 7/7 PASSED.
  - Re-verified Phase 1 (`test_phase1.py`: 4/4 PASSED) and Phase 2 (`test_phase2.py`: 7/7 PASSED) tests with zero regressions.

### Phase 4: Personalized KMeans, PCA and My Music Map
- **Personalized ML Service** (`backend/services/personal_ml_service.py`):
  - Strict isolation: Trains exclusively on the authenticated user's imported Spotify library tracks passing Phase 3 clustering eligibility (9 features, confidence >= 0.75, domain bounds).
  - Feature standardisation: `StandardScaler` applied to the 9 audio features.
  - Adaptive cluster count selection (`select_best_k`): Dynamically selects K based on sample size and silhouette score evaluation across valid partitions, strictly preventing degenerate singleton clusters.
  - 2D PCA projection: Reduces 9-dimensional audio space to 2D coordinates (x, y) independently for each user.
  - Understandable cluster labels: Semantically generates descriptive vibe groups (e.g., "High Energy & Dance Beats", "Acoustic & Mellow") based on positive/negative feature z-score deviations.
  - Private cosine similarity: Recommends similar tracks within user's library with normalized similarity percentages, strictly excluding the input track.
  - Lifecycle & staleness detection: Tracks model freshness in memory with persistence of training metadata in `PersonalModelMetadata`. Avoids retraining on every API call while automatically retraining when library size or feature values change.
- **Personalized Endpoints** (`backend/routers/personal.py`):
  - `GET /api/personal/status`: Overview of clustering status, eligible vs awaiting feature counts, cluster profiles, and model diagnostics.
  - `GET /api/personal/map`: 2D PCA coordinates and cluster assignments filtered by cluster and limit.
  - `GET /api/personal/clusters`: Cluster summaries, counts, and top feature descriptors.
  - `GET /api/personal/recommendations/{track_id}`: Cosine similarity recommendations strictly within user's library.
  - `POST /api/personal/train`: Explicit model training/refresh trigger.
  - Strict user isolation and session-cookie authentication on all endpoints.
- **Frontend "My Music Map" UI** (`frontend/src/App.jsx`):
  - Dedicated interactive 2D scatter plot using Plotly with cluster-based color coding.
  - Cluster filter dropdown, in-map track search, hover details (energy, valence, tempo, cluster name).
  - Selected track inspection card and "Find Similar in Library" trigger.
  - Informative banner showing tracks plotted, eligible count, and awaiting feature count with quick-action feature acquisition button.
  - Full loading, empty, and insufficient-tracks states.
- **Testing & Verification** (`backend/test_phase4.py`):
  - 10 comprehensive tests covering dataset isolation, eligibility filtering, insufficient tracks, adaptive K selection, degenerate cluster prevention, PCA coordinate generation, cosine similarity ranking, input track exclusion, model persistence/retraining, and Global Music Map regression.
  - 10/10 tests PASSED.
