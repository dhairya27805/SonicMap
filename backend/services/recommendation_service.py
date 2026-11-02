import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from state import STATE
from services.ml_service import get_song_card


def search_songs(query="", limit=30):
    df = STATE["df"]

    if df is None:
        raise RuntimeError("The music dataset has not been loaded.")

    limit = max(1, min(int(limit), 100))

    if query:
        mask = df["track_name"].astype(str).str.contains(
            query, case=False, regex=False
        ) | df["artists"].astype(str).str.contains(query, case=False, regex=False)
        indices = np.flatnonzero(mask.to_numpy())[:limit]
    else:
        indices = np.arange(min(limit, len(df)))

    return [get_song_card(int(i)) for i in indices]


def get_recommendations(song_id, limit=10):
    df = STATE["df"]
    scaled = STATE["scaled"]
    features = STATE["features"]

    if df is None or scaled is None:
        raise RuntimeError("The recommendation model is not ready.")

    song_id = int(song_id)

    if song_id < 0 or song_id >= len(df):
        raise IndexError("Song ID was not found.")

    limit = max(1, min(int(limit), 30))

    similarities = cosine_similarity(scaled[song_id : song_id + 1], scaled)[0]

    # Exclude the selected song itself.
    similarities[song_id] = -1

    candidate_count = min(limit, len(df) - 1)
    if candidate_count <= 0:
        return []

    indices = np.argsort(similarities)[::-1][:candidate_count]
    selected_features = df.iloc[song_id]

    recommendations = []

    for index in indices:
        candidate = df.iloc[int(index)]
        card = get_song_card(int(index))

        # Convert cosine similarity into a bounded display score.
        similarity = float(np.clip(similarities[index], 0, 1))
        card["similarity"] = round(similarity * 100, 1)

        comparisons = {}

        for feature in features:
            try:
                a = float(selected_features[feature])
                b = float(candidate[feature])

                if np.isfinite(a) and np.isfinite(b):
                    comparisons[feature] = {
                        "selected": round(a, 3),
                        "recommended": round(b, 3),
                        "difference": round(abs(a - b), 3),
                    }
            except (TypeError, ValueError):
                continue

        # Most comparable features first.
        ordered = sorted(comparisons.items(), key=lambda item: item[1]["difference"])

        card["feature_comparison"] = [
            {
                "feature": name,
                **values,
            }
            for name, values in ordered[:3]
        ]

        recommendations.append(card)

    return recommendations


def get_playlist(mood="discovery", limit=15):
    df = STATE["df"]

    if df is None:
        raise RuntimeError("The music dataset has not been loaded.")

    limit = max(1, min(int(limit), 30))
    rng = np.random.default_rng()

    def feature(name):
        if name in df.columns:
            return df[name].fillna(0).to_numpy(dtype=float)
        return np.zeros(len(df))

    energy = feature("energy")
    danceability = feature("danceability")
    acousticness = feature("acousticness")
    valence = feature("valence")

    if mood == "energetic":
        scores = energy
    elif mood == "chill":
        scores = 0.6 * acousticness + 0.4 * (1 - energy)
    elif mood == "dance":
        scores = 0.6 * danceability + 0.4 * energy
    elif mood == "happy":
        scores = valence
    else:
        cluster_col = df["cluster"].to_numpy()
        indices = []

        for cluster_id in np.unique(cluster_col):
            members = np.flatnonzero(cluster_col == cluster_id)
            if len(members):
                indices.extend(
                    rng.choice(
                        members, size=min(3, len(members)), replace=False
                    ).tolist()
                )

        rng.shuffle(indices)
        return [get_song_card(int(i)) for i in indices[:limit]]

    ranked = np.argsort(scores)[::-1]
    chosen = []
    cluster_counts = {}

    for index in ranked:
        cluster_id = int(df.iloc[int(index)]["cluster"])

        if cluster_counts.get(cluster_id, 0) >= 4:
            continue

        chosen.append(int(index))
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        if len(chosen) >= limit:
            break

    return [get_song_card(i) for i in chosen]
