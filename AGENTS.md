# SONICMAP — PROJECT DEVELOPMENT INSTRUCTIONS

## 1. Project identity

SonicMap is an AI-powered personalized music discovery platform.

The objective is to transform the existing application into a complete product where users connect their Spotify accounts, import accessible music data, discover music through machine learning, generate personalized playlists and listen through supported playback providers.

This is an existing project. Extend it; do not rebuild it unnecessarily.

## 2. Existing architecture

The existing application contains:

- FastAPI backend.
- React + Vite frontend.
- Spotify OAuth integration.
- Existing Global Music Map.
- Existing 20,000-song music dataset.
- KMeans clustering with eight existing global clusters.
- PCA visualization.
- Cosine similarity.
- Existing backend services, routers and frontend components.

Preserve existing functionality and authentication.

Inspect only the files relevant to the current task. Do not repeatedly scan the entire repository.

## 3. Two independent music experiences

### Global Music Map

- Continue using the existing 20,000-song dataset.
- Preserve the existing nine-feature schema.
- Preserve the existing eight-cluster KMeans implementation.
- Preserve PCA and cosine similarity.
- Do not replace the dataset with synthetic data.
- Do not modify the global model unnecessarily.

### My Music Map

Every authenticated user must have an independent personalized music experience.

Import accessible Spotify data such as:

- Liked songs.
- Accessible playlists.
- Recently played tracks.
- Top tracks, where permitted.
- Track and artist metadata.

Never mix different users' private libraries.

## 4. Nine audio features

Use the existing feature schema:

1. Danceability
2. Energy
3. Loudness
4. Speechiness
5. Acousticness
6. Instrumentalness
7. Liveness
8. Valence
9. Tempo

Spotify API access to these features must not be assumed.

Create a modular audio-feature acquisition service using:

- Reliable external music datasets.
- Legally accessible audio analysis using Librosa or equivalent.
- Validated ML-based feature estimation where appropriate.

Track the provenance and confidence of every feature.

Never invent missing values.

Never treat estimated features as actual measurements.

Songs without reliable feature vectors can remain in the library but must not enter clustering until suitable data is available.

## 5. Personalized ML pipeline

For each user:

1. Validate available audio features.
2. Preprocess and normalize data.
3. Apply personalized KMeans.
4. Choose a reasonable cluster count based on available data.
5. Generate PCA coordinates.
6. Calculate similar songs using cosine similarity.
7. Display meaningful cluster information.

Do not force eight clusters on every user.

Do not retrain the Global Music Map model when a user connects Spotify.

## 6. Music discovery

SonicMap must discover new music outside the user's existing library.

Create a separate discovery catalogue.

Recommendations should consider:

- Audio features.
- Cluster preferences.
- Similarity scores.
- User listening preferences.
- Songs already present in the user's library.

Avoid repetitive recommendations.

Use legitimate data and music providers.

Do not scrape protected services or download copyrighted Spotify streams.

## 7. Similar-song recommendations

Allow users to select a track and find similar songs.

Use normalized nine-feature vectors and cosine similarity, with other justified signals where useful.

Support recommendations from both the personal library and discovery catalogue.

## 8. Playlist generation

Generate playlists directly inside SonicMap.

Playlist types:

- Cluster-based playlists.
- Similar-song playlists.
- Mood-based playlists.
- Energy-based playlists.
- Discovery playlists.
- Custom playlists.

Allow users to save playlists in their SonicMap account.

Spotify playlist creation is not required.

Only include tracks with valid playback sources as playable tracks.

## 9. Music player

Build an independent playback-provider architecture.

Required player features:

- Play and pause.
- Next and previous.
- Seek and progress.
- Volume.
- Shuffle.
- Repeat.
- Queue.
- Track artwork.
- Playlist playback.
- Persistent player across navigation where possible.

Investigate Spotify Web Playback SDK, licensed music catalogues, permitted royalty-free sources and user-provided local audio.

Respect all licensing, Premium and provider restrictions.

Spotify login does not automatically provide permission to stream protected Spotify audio through a custom player.

Never display fake playback functionality.

## 10. Frontend pages

Create or improve:

1. Home / Dashboard
2. Global Music Map
3. My Music Map
4. My Library
5. Discover
6. Similar Songs
7. AI Playlists
8. Saved Playlists
9. Music Player
10. Profile / Settings

Design direction:

- Modern dark interface.
- Distinctive SonicMap branding.
- Professional typography.
- Smooth transitions.
- Interactive visualizations.
- Responsive layouts.
- Functional controls.
- Proper loading, empty and error states.

Avoid generic AI-generated dashboard designs.

## 11. Technical architecture

Preserve the existing FastAPI and React/Vite structure.

Keep backend functionality modular.

Use appropriate database relationships for users, tracks, playlists, feature vectors and personalized models.

Protect authentication tokens and isolate user data.

Keep environment variables out of frontend code.

Prepare the project for Render backend and Vercel frontend deployment.

Use persistent production database storage.

Do not introduce unnecessary complexity or microservices.

## 12. Development phases

Work in this order:

PHASE 0: Existing project audit and architecture.

PHASE 1: Preserve existing features and prepare backend/database architecture.

PHASE 2: Spotify library import.

PHASE 3: Nine-feature acquisition engine.

PHASE 4: Personalized KMeans, PCA and My Music Map.

PHASE 5: External music discovery and recommendations.

PHASE 6: Playlist generation and storage.

PHASE 7: Music player and playback-provider integration.

PHASE 8: UI integration, testing, security and deployment.

Complete one phase at a time.

Do not begin the next phase until the current milestone is tested and documented.

## 13. Token and context efficiency

These rules are mandatory:

- Do not dump entire source files into chat unless necessary.
- Work directly in the existing project files.
- Inspect relevant files instead of repeatedly scanning everything.
- Avoid unnecessary dependency installation.
- Avoid rewriting working code.
- Implement small, testable changes.
- Keep explanations concise.
- Maintain PROGRESS.md.
- Maintain HANDOVER.md.
- Update documentation after every milestone.
- Create a complete project ZIP after each major phase.
- Exclude .env, secrets, virtual environments, node_modules and caches from ZIPs.
- Never claim tests passed unless they were actually run.

If context becomes limited:

1. Save the current work.
2. Document modified files.
3. Record completed and incomplete tasks.
4. Record test results.
5. Update HANDOVER.md.
6. Update PROGRESS.md.
7. Stop safely at a working checkpoint.

## 14. Development quality

Do not create fake buttons, fake recommendations, fake track metadata or fake playback.

Use real data wherever possible.

Handle API failures and missing features gracefully.

Write maintainable code.

Do not silently break existing functionality.

Test the actual user workflow, not just whether the backend starts.

## 15. Immediate instruction

Wait for a specific development task.

Before modifying anything, inspect the relevant existing files and explain the proposed changes briefly.

Do not automatically start implementing the entire product just because these instructions are present.