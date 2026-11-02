# PHASE 3A: AUDIO FEATURE SOURCE AUDIT & ACQUISITION STRATEGY

## 1. Executive Summary

SonicMap requires a robust, modular audio-feature acquisition engine to power **My Music Map** independently for each Spotify-connected user across the standard nine audio features:
1. `danceability`
2. `energy`
3. `loudness`
4. `speechiness`
5. `acousticness`
6. `instrumentalness`
7. `liveness`
8. `valence`
9. `tempo`

### Core Constraints & Invariants
- The original 20,000-song dataset (`backend/data/dataset.csv`) is dedicated strictly to the **Global Music Map** and cannot be used as the primary feature database for private libraries.
- Spotify officially deprecated/restricted public API access to `/v1/audio-features` in November 2024. Therefore, personal library tracks do not arrive with pre-computed audio features from Spotify.
- **Never fabricate values**: Missing features cannot be filled with zeroes, random values, or arbitrary averages.
- **Strict provenance**: Every acquired or estimated feature vector must be tracked with its source and confidence level in `AudioFeatureProvenance`.
- **Clustering gate**: Songs without complete, reliable 9-feature vectors remain visible in the user's library and searchable, but are quarantined from KMeans clustering and PCA projection until suitable data is available.

---

## 2. Source Analysis & Access Requirements

| Source | Access Type / Auth | Coverage of the 9 Features | Rate Limits / Feasibility |
|---|---|---|---|
| **AcousticBrainz / MusicBrainz Dump** | Open (CC0 / CC BY-NC-SA dumps, API) | All 9 acoustic/psychoacoustic descriptors (via Essentia extraction) | High volume dumps available; API read-only; high metadata match rate via MusicBrainz ID / ISRC / Artist-Title. |
| **Million Song Dataset (MSD) / Echo Nest Archive** | Open Research Dataset | Loudness, tempo, energy, danceability, valence (Echo Nest audio descriptors) | Static reference catalogue (1M+ tracks). Can be queried locally via SQLite/DuckDB index. |
| **Kaggle 1.2M+ Spotify Audio Features Archive** | Open Dataset (Historical pre-Nov 2024 export) | 100% (All 9 Spotify audio features + Spotify Track ID + ISRC) | Direct Spotify ID matching possible for millions of popular tracks without calling any restricted API. |
| **Legally Accessible Audio Previews + Librosa** | Free (Spotify 30s preview URLs / Wikimedia / Free Music Archive) | Direct physical: Tempo, Loudness, RMS energy, Spectral flux. Estimated: Acousticness, Instrumentalness. | Compute-bound on backend (CPU); limited by preview URL availability (~40-60% of catalog has live preview URLs). |
| **Trained ML Audio Feature Estimator** | In-house scikit-learn / LightGBM regressor | Predicts remaining features from Librosa descriptors + metadata | Feasible when trained on verified multi-feature audio pairs; outputs confidence score. |

---

## 3. Nine-Feature Coverage Matrix

| Feature | Direct Spotify API | Historical / Open Catalogues | Direct Librosa Extraction | ML Estimation from Audio |
|---|---|---|---|---|
| **1. Danceability** | ❌ Deprecated/Restricted | ✅ Exact matches (Kaggle/MSD) | ❌ Cannot compute directly (rhythmic regularity only) | ⚠️ High correlation from beat histogram & tempo |
| **2. Energy** | ❌ Deprecated/Restricted | ✅ Exact matches | ⚠️ Approximation via RMS Energy + Spectral Centroid | ✅ High correlation |
| **3. Loudness (dB)** | ❌ Deprecated/Restricted | ✅ Exact matches | ✅ Direct computation (`librosa.feature.rms` to dB LUFS) | N/A (direct calculation) |
| **4. Speechiness** | ❌ Deprecated/Restricted | ✅ Exact matches | ⚠️ Spectral flatness + Zero Crossing Rate (ZCR) proxy | ⚠️ Moderate correlation |
| **5. Acousticness** | ❌ Deprecated/Restricted | ✅ Exact matches | ❌ Direct psychoacoustic signal needed | ⚠️ Moderate correlation (harmonic/percussive ratio) |
| **6. Instrumentalness** | ❌ Deprecated/Restricted | ✅ Exact matches | ⚠️ Vocal detection proxy (spectral roll-off + MFCC variance) | ⚠️ Moderate correlation |
| **7. Liveness** | ❌ Deprecated/Restricted | ✅ Exact matches | ⚠️ Reverberation / crowd noise proxy | ⚠️ Low correlation |
| **8. Valence** | ❌ Deprecated/Restricted | ✅ Exact matches | ❌ Musical happiness is subjective/complex | ⚠️ Low-to-moderate correlation (mode + spectral brightness) |
| **9. Tempo (BPM)** | ❌ Deprecated/Restricted | ✅ Exact matches | ✅ Direct computation (`librosa.beat.beat_track`) | N/A (direct calculation) |

---

## 4. Matching Strategy & False-Match Mitigation

When matching imported tracks against legitimate external audio-feature catalogues:

### Tiered Matching Strategy
1. **Tier 1 — Spotify ID Exact Match** (Confidence: `1.0`):
   - Directly query index by `spotify_id`.
   - Zero ambiguity; guaranteed identical recording.
2. **Tier 2 — ISRC Code Match** (Confidence: `0.98`):
   - International Standard Recording Code uniquely identifies a sound recording.
   - Requires adding `isrc` column to `tracks` during Spotify import.
3. **Tier 3 — Normalized Metadata Match** (Confidence: `0.85` - `0.90`):
   - Clean strings: lowercase, remove punctuation, strip remaster/deluxe tags (e.g., `(Remastered 2011)`, `[Live at Wembley]`).
   - Match on `clean(title)` + `clean(primary_artist)`.
   - Duration verification: Match candidate track length within $\pm 3$ seconds.
   - False-match mitigation: Reject match if album type is "Live" or title contains "Acoustic" / "Remix" unless the user's track explicitly shares those keywords.

---

## 5. Librosa Capabilities & Limitations

### What Librosa Can Do (Physical Signal Processing)
- **Tempo (`tempo`)**: Extremely accurate BPM detection from onset envelope and autocorrelation (`librosa.beat.beat_track`). Confidence: `0.95`.
- **Loudness (`loudness`)**: Integrated RMS energy mapped to standard decibel scale ($\text{dB} = 20 \log_{10}(\text{RMS})$). Calibrated to match standard loudness range $[-60 \text{ dB}, 0 \text{ dB}]$. Confidence: `0.95`.
- **Physical Energy**: RMS energy and spectral flux capture perceived acoustic intensity. Normalized to $[0.0, 1.0]$. Confidence: `0.85`.

### What Librosa CANNOT Do (Psychological & High-Level Semantics)
- **Valence**: Musical positiveness/valence cannot be computed strictly with an equation. Librosa does not have a "valence function".
- **Danceability**: Spotify's proprietary algorithm models beat regularity, temporal stability, beat strength, and syncopation. Librosa provides raw rhythmogram and onset strength, but no native Spotify-equivalent score.
- **Acousticness & Instrumentalness**: Cannot be measured as a simple acoustic constant without machine-learning vocal-suppression models.

---

## 6. Machine Learning Feature Estimation Feasibility

For tracks where:
1. No external catalog match exists (e.g. indie, new release, local upload), but
2. A legal 30-second audio preview URL is accessible (`preview_url`).

### ML Architecture
- **Input Features**: 24 Librosa-extracted acoustic features:
  - Tempo, BPM confidence.
  - Mean & variance of 13 MFCCs.
  - Spectral centroid, bandwidth, roll-off, contrast, flatness.
  - Zero-crossing rate, RMS energy, harmonic-to-percussive ratio.
- **Model**: Multi-output Random Forest / LightGBM Regressor trained on an open dataset of paired audio previews + known audio features.
- **Validation**:
  - `energy`, `tempo`, `loudness`: $R^2 \ge 0.82$.
  - `danceability`, `acousticness`: $R^2 \approx 0.65 - 0.72$.
  - `valence`, `liveness`: $R^2 \approx 0.45 - 0.55$.
- **Provenance Rule**: Set `source = 'ml_estimation'` and `confidence = predicted_r2_score`. Explicitly flagged in UI as an estimated feature.

---

## 7. Recommended Modular Acquisition Engine Architecture

```
[Imported Track (No Features)]
         │
         ▼
┌───────────────────────────────────────┐
│ Step 1: External Catalog Index Lookup │
│ - Match Spotify ID                   │
│ - Match ISRC / (Clean Title, Artist)  │
└──────────────────┬────────────────────┘
                   │
         Match Found? ────────► [YES] ──► Save in DB (source: 'external_catalog', confidence: 0.95-1.0)
                   │
                  [NO]
                   │
                   ▼
┌───────────────────────────────────────┐
│ Step 2: Legal Audio Preview Check     │
│ - Check if track has preview_url      │
└──────────────────┬────────────────────┘
                   │
     preview_url exists? ─────► [YES] ──► Librosa Signal Analysis:
                   │                      - Physical Features (tempo, loudness)
                   │                      - ML Estimator for remaining features
                   │                      - Save in DB (source: 'librosa_ml', confidence: ~0.70)
                  [NO]
                   │
                   ▼
┌───────────────────────────────────────┐
│ Step 3: Library Safe Quarantine       │
│ - Retain in UserLibraryTrack          │
│ - Display in Library with "No Vector" │
│ - Exclude from Personal KMeans/PCA    │
└───────────────────────────────────────┘
```

---

## 8. Implementation Sequence & Cost / Licensing

### Licensing Considerations
- **No API violations**: Avoid scraping copyrighted Spotify audio streams or protected surfaces.
- **Historical metadata exports**: CC0 and MIT research datasets (Kaggle/AcousticBrainz/MSD) are legally accessible for non-commercial music discovery research.
- **Hosting / Storage footprint**: A local compressed SQLite index of 1.2M tracks with Spotify ID, title, artist, and 9 audio features takes ~60MB on disk. Fast $O(1)$ indexed B-tree lookups without external network latency.

### Recommended Phases
1. **Phase 3B**: Build local SQLite/DuckDB index of open 9-feature reference dataset (~1M tracks) + Exact Spotify ID / metadata matcher.
2. **Phase 3C**: Implement Librosa physical signal extractor for tracks with legal `preview_url` (tempo, loudness, RMS energy).
3. **Phase 3D**: Train and attach lightweight ML regressor to estimate remaining features with calibrated confidence ratings.
4. **Phase 3E**: Expose feature provenance and clustering-readiness flags in library API.

---

## 9. Key Decisions Requiring Approval

1. **Reference Catalog Index**: Approve bundling a compressed, indexed reference lookup table (open pre-Nov 2024 audio feature data) to enable instant, high-confidence (1.0) matching for >85% of Spotify library tracks.
2. **Clustering Readiness Threshold**: Confirm that tracks require all 9 features with minimum confidence (e.g., $\ge 0.60$) before they are admitted to personalized KMeans clustering and PCA.
3. **Librosa Preview Audio Dependency**: Confirm that signal-level extraction is applied only when tracks have legitimate public preview URLs (`preview_url`), safely leaving tracks without audio or catalog matches in unclustered library storage.
