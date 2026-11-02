"""
Personalized ML Service for My Music Map (Phase 4)

Provides:
- Independent StandardScaler, adaptive KMeans, and 2D PCA projection for each user's library.
- Strict isolation from the Global Music Map model and dataset.
- Adaptive cluster count selection (evaluated using silhouette score for K in [2..min(8, N-1)]).
- Degenerate / singleton cluster prevention.
- Semantic cluster labeling based on feature dominance (e.g. "Acoustic & Mellow", "High Energy Beats").
- Cosine similarity recommendation within the user's private library (excluding input track).
- Model caching in memory with persistence of training metadata in PersonalModelMetadata.
"""

import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from models import Track, UserLibraryTrack, PersonalModelMetadata, utcnow
from state import AUDIO_FEATURES, RANDOM_STATE
from services.feature_engine_service import check_clustering_eligibility

logger = logging.getLogger("sonicmap.personal_ml")

# In-memory store for trained personal user models:
# { user_id: { "df": pd.DataFrame, "scaled": np.ndarray, "coords": np.ndarray, "kmeans": KMeans, "pca": PCA, "scaler": StandardScaler, "clusters": list, "trained_at": datetime, "version": str, "eligible_count": int } }
PERSONAL_MODELS = {}

MIN_TRACKS_FOR_CLUSTERING = 4


def generate_cluster_label(cluster_df: pd.DataFrame, all_df: pd.DataFrame, cluster_id: int) -> str:
    """
    Generates meaningful cluster title based on feature deviations from user's overall library average.
    """
    if len(cluster_df) == 0:
        return f"Sound Group {cluster_id + 1}"

    mean_cluster = cluster_df[AUDIO_FEATURES].mean()
    mean_all = all_df[AUDIO_FEATURES].mean()
    std_all = all_df[AUDIO_FEATURES].std().fillna(1.0).replace(0, 1.0)

    # Z-score of cluster features relative to library
    z_scores = (mean_cluster - mean_all) / std_all

    sorted_pos = z_scores.sort_values(ascending=False)
    top_pos = sorted_pos.index[0]
    top_pos_val = sorted_pos.iloc[0]

    sorted_neg = z_scores.sort_values(ascending=True)
    lowest = sorted_neg.index[0]

    if top_pos == "energy" and top_pos_val > 0.2:
        if z_scores.get("danceability", 0) > 0.2:
            return "High Energy & Dance Beats"
        return "High Energy & Intense"
    elif top_pos == "danceability" and top_pos_val > 0.2:
        if z_scores.get("valence", 0) > 0.2:
            return "Rhythmic & Upbeat Groove"
        return "Rhythmic & Danceable"
    elif top_pos == "acousticness" and top_pos_val > 0.2:
        if lowest == "energy":
            return "Acoustic & Mellow"
        return "Acoustic & Organic"
    elif top_pos == "valence" and top_pos_val > 0.2:
        return "Bright & Uplifting"
    elif top_pos == "instrumentalness" and top_pos_val > 0.2:
        return "Instrumental & Ambient Soundscapes"
    elif top_pos == "tempo" and top_pos_val > 0.2:
        return "Fast-Paced & Up-Tempo"
    elif top_pos == "speechiness" and top_pos_val > 0.2:
        return "Lyric-Heavy & Spoken Word"
    elif top_pos == "loudness" and top_pos_val > 0.2:
        return "Loud & Punchy Anthems"
    else:
        # Check negative extreme
        if lowest == "energy":
            return "Calm, Chill & Relaxed"
        elif lowest == "valence":
            return "Melancholic & Reflective"

    return f"Vibe Group {cluster_id + 1}"


def select_best_k(scaled_matrix: np.ndarray, n_samples: int) -> tuple[int, float | None]:
    """
    Adaptively selects K based on available sample count and silhouette score evaluation.
    Evaluates candidate K from 2 to min(8, n_samples - 1, n_samples // 2).
    Avoids degenerate singletons or overly fragmented clustering.
    """
    if n_samples < MIN_TRACKS_FOR_CLUSTERING:
        return 1, None

    # Determine maximum candidate K based on sample size
    if n_samples < 6:
        max_k = 2
    elif n_samples < 9:
        max_k = min(3, n_samples - 1)
    else:
        max_k = min(8, n_samples // 2, n_samples - 1)

    if max_k < 2:
        return 1, None

    candidate_ks = list(range(2, max_k + 1))
    best_k = 2
    best_score = -2.0
    best_non_singleton_k = None
    best_non_singleton_score = -2.0

    for k in candidate_ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled_matrix)

        # Check for degenerate solutions (e.g. fewer unique clusters than k or empty clusters)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < k or min(counts) < 1:
            continue

        try:
            score = float(silhouette_score(scaled_matrix, labels, random_state=RANDOM_STATE))
        except Exception:
            continue

        has_singleton = any(c < 2 for c in counts)
        if not has_singleton:
            if score > best_non_singleton_score:
                best_non_singleton_score = score
                best_non_singleton_k = k

        if score > best_score:
            best_score = score
            best_k = k

    # Strongly prefer non-singleton solutions to prevent degenerate clustering
    if best_non_singleton_k is not None:
        return best_non_singleton_k, best_non_singleton_score
    elif best_score > -1.0:
        return best_k, best_score

    return min(2, max_k), None


def get_user_library_counts(db: Session, user_id: int) -> dict:
    """Returns total unique tracks, eligible tracks, and tracks awaiting features for a user."""
    user_tracks = (
        db.query(Track)
        .join(UserLibraryTrack, UserLibraryTrack.track_id == Track.id)
        .filter(UserLibraryTrack.user_id == user_id)
        .all()
    )

    # Deduplicate by track ID
    seen = {}
    for t in user_tracks:
        if t.id not in seen:
            seen[t.id] = t

    unique_tracks = list(seen.values())
    total_tracks = len(unique_tracks)
    eligible = [t for t in unique_tracks if check_clustering_eligibility(t)[0]]
    eligible_count = len(eligible)
    awaiting = max(0, total_tracks - eligible_count)

    return {
        "total_tracks": total_tracks,
        "eligible_tracks": eligible_count,
        "awaiting_features": awaiting,
        "eligible": eligible,
    }


def get_eligible_user_tracks(db: Session, user_id: int) -> list[Track]:
    """Retrieves all tracks in user's private library passing Phase 3 clustering eligibility (deduplicated)."""
    counts = get_user_library_counts(db, user_id)
    return counts["eligible"]


def is_model_stale(db: Session, user_id: int) -> tuple[bool, str]:
    """
    Checks whether the user's personal model needs retraining due to:
    - Model not in memory or never trained
    - Change in eligible track count
    - Track updates after the last training timestamp
    """
    cached = PERSONAL_MODELS.get(user_id)
    meta = db.query(PersonalModelMetadata).filter(PersonalModelMetadata.user_id == user_id).first()

    if not meta or not cached:
        return True, "Model not trained or not cached in memory"

    counts = get_user_library_counts(db, user_id)
    if counts["eligible_tracks"] != meta.track_count:
        return True, f"Eligible track count changed: {meta.track_count} -> {counts['eligible_tracks']}"

    # Check if any eligible track was updated after last training
    last_trained = meta.last_trained_at
    if last_trained:
        if last_trained.tzinfo is None:
            last_trained = last_trained.replace(tzinfo=datetime.timezone.utc)
        for t in counts["eligible"]:
            t_updated = t.updated_at
            if t_updated:
                if t_updated.tzinfo is None:
                    t_updated = t_updated.replace(tzinfo=datetime.timezone.utc)
                if t_updated > last_trained:
                    return True, f"Track {t.id} updated after last training"

    return False, "Model is up to date"


def train_personal_model(db: Session, user_id: int, force: bool = False) -> dict:
    """
    Trains or refreshes the user's personalized KMeans and PCA models.
    Persists diagnostics to PersonalModelMetadata.
    """
    counts = get_user_library_counts(db, user_id)
    n_tracks = counts["eligible_tracks"]
    eligible_tracks = counts["eligible"]

    if n_tracks < MIN_TRACKS_FOR_CLUSTERING:
        return {
            "status": "insufficient_tracks",
            "message": f"Requires at least {MIN_TRACKS_FOR_CLUSTERING} eligible tracks to train My Music Map (currently {n_tracks} eligible).",
            "total_tracks": counts["total_tracks"],
            "eligible_tracks": n_tracks,
            "awaiting_features": counts["awaiting_features"],
            "min_required": MIN_TRACKS_FOR_CLUSTERING,
        }

    # Build DataFrame from eligible tracks only
    data = []
    for t in eligible_tracks:
        row = {
            "id": t.id,
            "spotify_id": t.spotify_id or "",
            "title": t.title,
            "artist": t.artist,
            "album": t.album or "",
            "image_url": t.image_url or "",
            "preview_url": t.preview_url or "",
            "duration_ms": t.duration_ms or 0,
        }
        for feat in AUDIO_FEATURES:
            row[feat] = float(getattr(t, feat))
        data.append(row)

    df = pd.DataFrame(data)

    # 1. StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[AUDIO_FEATURES])

    # 2. Adaptive KMeans selection
    k, sil_score = select_best_k(scaled, n_tracks)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(scaled)
    df["cluster"] = labels

    # 3. PCA Projection (2D)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)
    df["x"] = [round(float(c[0]), 4) if np.isfinite(c[0]) else 0.0 for c in coords]
    df["y"] = [round(float(c[1]), 4) if np.isfinite(c[1]) else 0.0 for c in coords]

    # 4. Generate cluster details
    clusters_info = []
    for cid in range(k):
        c_df = df[df["cluster"] == cid]
        name = generate_cluster_label(c_df, df, cid)
        mean_c = c_df[AUDIO_FEATURES].mean()
        mean_all = df[AUDIO_FEATURES].mean()
        std_all = df[AUDIO_FEATURES].std().fillna(1.0).replace(0, 1.0)
        z = (mean_c - mean_all) / std_all
        top_feats = [f for f in z.sort_values(ascending=False).index[:2] if z[f] > 0]
        clusters_info.append({
            "id": int(cid),
            "name": name,
            "count": int(len(c_df)),
            "top_features": top_feats,
        })

    model_version = f"v_{utcnow().strftime('%Y%m%d_%H%M%S')}_{utcnow().microsecond:06d}"

    # Cache in memory
    PERSONAL_MODELS[user_id] = {
        "df": df,
        "scaled": scaled,
        "coords": coords,
        "kmeans": kmeans,
        "pca": pca,
        "scaler": scaler,
        "clusters": clusters_info,
        "trained_at": utcnow(),
        "version": model_version,
        "eligible_count": n_tracks,
    }

    # 5. Persist diagnostics in PersonalModelMetadata
    meta = db.query(PersonalModelMetadata).filter(PersonalModelMetadata.user_id == user_id).first()
    if not meta:
        meta = PersonalModelMetadata(user_id=user_id)
        db.add(meta)

    meta.cluster_count = k
    meta.silhouette_score = sil_score
    meta.track_count = n_tracks
    meta.variance_ratio_pca1 = (
        float(pca.explained_variance_ratio_[0])
        if len(pca.explained_variance_ratio_) > 0 and np.isfinite(pca.explained_variance_ratio_[0])
        else 0.0
    )
    meta.variance_ratio_pca2 = (
        float(pca.explained_variance_ratio_[1])
        if len(pca.explained_variance_ratio_) > 1 and np.isfinite(pca.explained_variance_ratio_[1])
        else 0.0
    )
    meta.model_version = model_version
    meta.last_trained_at = utcnow()
    db.commit()

    return {
        "status": "trained",
        "user_id": user_id,
        "model_version": model_version,
        "cluster_count": k,
        "eligible_tracks": n_tracks,
        "total_tracks": counts["total_tracks"],
        "awaiting_features": counts["awaiting_features"],
        "silhouette_score": sil_score,
        "clusters": clusters_info,
        "explained_variance": [
            round(float(v), 4) if np.isfinite(v) else 0.0 for v in pca.explained_variance_ratio_
        ],
    }


def get_personal_model(db: Session, user_id: int, auto_train: bool = True) -> dict | None:
    """Gets the active model from memory or trains if stale/uninitialized."""
    if user_id in PERSONAL_MODELS:
        stale, _ = is_model_stale(db, user_id)
        if not stale:
            return PERSONAL_MODELS[user_id]
        if auto_train:
            train_personal_model(db, user_id)
            return PERSONAL_MODELS.get(user_id)
        return PERSONAL_MODELS.get(user_id)

    # Check if user has eligible tracks to train
    counts = get_user_library_counts(db, user_id)
    if counts["eligible_tracks"] >= MIN_TRACKS_FOR_CLUSTERING and auto_train:
        train_personal_model(db, user_id)
        return PERSONAL_MODELS.get(user_id)

    return None


def get_personal_map_points(db: Session, user_id: int, cluster_id: int = -1, limit: int = 2000) -> list[dict]:
    """Returns 2D PCA coordinates and cluster assignments for the user's personal library map."""
    model = get_personal_model(db, user_id)
    if not model:
        return []

    df = model["df"]
    if cluster_id >= 0:
        filtered = df[df["cluster"] == cluster_id]
    else:
        filtered = df

    points = []
    for _, row in filtered.head(limit).iterrows():
        points.append({
            "id": int(row["id"]),
            "spotify_id": str(row["spotify_id"]),
            "title": str(row["title"]),
            "artist": str(row["artist"]),
            "album": str(row["album"]),
            "image_url": str(row["image_url"]),
            "cluster": int(row["cluster"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "danceability": round(float(row["danceability"]), 3),
            "energy": round(float(row["energy"]), 3),
            "valence": round(float(row["valence"]), 3),
            "tempo": round(float(row["tempo"]), 1),
            "loudness": round(float(row["loudness"]), 1),
        })
    return points


def get_personal_clusters_overview(db: Session, user_id: int) -> dict:
    """Returns cluster summaries, library statistics, and model metadata for My Music Map."""
    counts = get_user_library_counts(db, user_id)
    model = get_personal_model(db, user_id)
    meta = db.query(PersonalModelMetadata).filter(PersonalModelMetadata.user_id == user_id).first()

    if counts["eligible_tracks"] < MIN_TRACKS_FOR_CLUSTERING:
        return {
            "status": "insufficient_tracks",
            "message": f"Requires at least {MIN_TRACKS_FOR_CLUSTERING} eligible tracks with complete audio features (currently {counts['eligible_tracks']} eligible).",
            "total_tracks": counts["total_tracks"],
            "eligible_tracks": counts["eligible_tracks"],
            "awaiting_features": counts["awaiting_features"],
            "min_required": MIN_TRACKS_FOR_CLUSTERING,
            "clusters": [],
            "cluster_count": 0,
            "silhouette_score": None,
            "model_version": None,
            "last_trained_at": None,
            "explained_variance": [],
        }

    if not model or not meta:
        return {
            "status": "not_trained",
            "message": "Personal model not yet trained. Click 'Train My Map' to generate your personalized map.",
            "total_tracks": counts["total_tracks"],
            "eligible_tracks": counts["eligible_tracks"],
            "awaiting_features": counts["awaiting_features"],
            "min_required": MIN_TRACKS_FOR_CLUSTERING,
            "clusters": [],
            "cluster_count": 0,
            "silhouette_score": None,
            "model_version": None,
            "last_trained_at": None,
            "explained_variance": [],
        }

    return {
        "status": "active",
        "cluster_count": meta.cluster_count,
        "total_tracks": counts["total_tracks"],
        "eligible_tracks": meta.track_count,
        "awaiting_features": max(0, counts["total_tracks"] - meta.track_count),
        "min_required": MIN_TRACKS_FOR_CLUSTERING,
        "silhouette_score": meta.silhouette_score,
        "model_version": meta.model_version,
        "last_trained_at": meta.last_trained_at.isoformat() if meta.last_trained_at else None,
        "clusters": model["clusters"],
        "explained_variance": [
            round(meta.variance_ratio_pca1 or 0.0, 4),
            round(meta.variance_ratio_pca2 or 0.0, 4),
        ],
    }


def get_personal_recommendations(db: Session, user_id: int, track_id: int, limit: int = 10) -> list[dict]:
    """
    Computes cosine similarity across the user's private eligible tracks.
    Strictly excludes the input track from recommendations.
    Never searches or returns tracks belonging to other users.
    """
    model = get_personal_model(db, user_id)
    if not model:
        return []

    df = model["df"]
    scaled = model["scaled"]

    # Locate input track in user's model DataFrame
    indices = df.index[df["id"] == track_id].tolist()
    if not indices:
        return []

    target_idx = indices[0]
    target_vec = scaled[target_idx : target_idx + 1]

    # Compute cosine similarity across user's scaled vectors
    sims = cosine_similarity(target_vec, scaled)[0]

    # Exclude the input track itself
    sims[target_idx] = -2.0

    candidate_count = min(limit, len(df) - 1)
    if candidate_count <= 0:
        return []

    top_indices = np.argsort(sims)[::-1][:candidate_count]
    results = []

    for idx in top_indices:
        sim_score = float(sims[idx])
        if sim_score < -1.0:
            continue
        row = df.iloc[idx]
        pct = max(0.0, min(100.0, (sim_score + 1.0) / 2.0 * 100.0))
        results.append({
            "id": int(row["id"]),
            "spotify_id": str(row["spotify_id"]),
            "title": str(row["title"]),
            "artist": str(row["artist"]),
            "album": str(row["album"]),
            "image_url": str(row["image_url"]),
            "cluster": int(row["cluster"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "similarity": round(pct, 1),
            "danceability": round(float(row["danceability"]), 3),
            "energy": round(float(row["energy"]), 3),
            "valence": round(float(row["valence"]), 3),
            "tempo": round(float(row["tempo"]), 1),
            "loudness": round(float(row["loudness"]), 1),
        })

    return results
