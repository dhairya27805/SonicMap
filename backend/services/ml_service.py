import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from state import DEFAULT_CLUSTERS, RANDOM_STATE, STATE
from services.data_service import prepare_dataframe


def fit_model(raw_df, source="Spotify dataset"):
    """Train the model and update the shared application state."""
    df, features = prepare_dataframe(raw_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])

    # Reduce the scaled audio features to two dimensions.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled)

    cluster_count = min(DEFAULT_CLUSTERS, len(df) - 1)

    clusterer = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = clusterer.fit_predict(scaled)
    df["cluster"] = labels

    score = None

    if len(df) >= 20 and len(np.unique(labels)) > 1:
        try:
            score = float(
                silhouette_score(
                    scaled,
                    labels,
                    sample_size=min(2000, len(df)),
                    random_state=RANDOM_STATE,
                )
            )
        except ValueError:
            score = None

    # Update state only after preprocessing and training succeed.
    STATE.update(
        {
            "df": df,
            "features": features,
            "scaled": scaled,
            "coordinates": coordinates,
            "clusterer": clusterer,
            "scaler": scaler,
            "pca": pca,
            "silhouette": score,
            "source": source,
        }
    )

    return {
        "total_tracks": len(df),
        "cluster_count": int(df["cluster"].nunique()),
        "features": features,
        "silhouette_score": score,
    }


def get_overview():
    df = STATE["df"]

    if df is None:
        raise RuntimeError("The ML model has not been initialized.")

    clusters = []

    for cluster_id, group in df.groupby("cluster"):
        clusters.append(
            {
                "id": int(cluster_id),
                "count": int(len(group)),
                "name": f"Sound Cluster {int(cluster_id) + 1}",
            }
        )

    return {
        "total_tracks": int(len(df)),
        "cluster_count": int(df["cluster"].nunique()),
        "features": STATE["features"],
        "source": STATE["source"],
        "silhouette_score": STATE["silhouette"],
        "clusters": clusters,
        "explained_variance": [
            round(float(value), 4) for value in STATE["pca"].explained_variance_ratio_
        ],
    }


def get_song_card(index):
    df = STATE["df"]

    if df is None or index < 0 or index >= len(df):
        raise IndexError("Song not found.")

    row = df.iloc[index]
    x, y = STATE["coordinates"][index]

    def get_number(field, digits=2):
        if field not in df.columns:
            return None

        value = row[field]

        if not np.isfinite(value):
            return None

        return round(float(value), digits)

    return {
        "id": int(index),
        "track_id": str(row["track_id"]),
        "title": str(row["track_name"]),
        "artist": str(row["artists"]),
        "cluster": int(row["cluster"]),
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "danceability": get_number("danceability"),
        "energy": get_number("energy"),
        "valence": get_number("valence"),
        "acousticness": get_number("acousticness"),
        "tempo": get_number("tempo", 1),
        "popularity": get_number("popularity", 0),
    }


def get_map_points(limit=4000, cluster=-1):
    df = STATE["df"]

    if df is None:
        raise RuntimeError("The ML model has not been initialized.")

    limit = max(100, min(int(limit), 5000))
    indices = np.arange(len(df))

    if cluster >= 0:
        indices = indices[df["cluster"].to_numpy()[indices] == cluster]

    if len(indices) > limit:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = np.sort(rng.choice(indices, size=limit, replace=False))

    return [get_song_card(int(i)) for i in indices]
