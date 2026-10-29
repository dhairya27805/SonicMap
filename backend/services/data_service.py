import pandas as pd

from state import AUDIO_FEATURES, MAX_TRAINING_SONGS, RANDOM_STATE


def find_column(df, candidates):
    """Find a column without depending on capitalization."""
    lookup = {str(column).strip().lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def prepare_dataframe(raw_df):
    """Validate and standardize a music dataset."""
    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    title_col = find_column(df, ["track_name", "track", "title", "name"])
    artist_col = find_column(df, ["artists", "artist", "artist_name"])

    if title_col is None:
        raise ValueError(
            "Missing track title column. Expected track_name, track, title, or name."
        )

    df = df.rename(columns={title_col: "track_name"})

    if artist_col is not None and artist_col != title_col:
        df = df.rename(columns={artist_col: "artists"})
    elif "artists" not in df.columns:
        df["artists"] = "Unknown Artist"

    features = [feature for feature in AUDIO_FEATURES if feature in df.columns]

    if len(features) < 3:
        raise ValueError(
            "At least 3 supported numerical audio features "
            "are required. Examples: energy, tempo, "
            "danceability, valence, acousticness."
        )

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=features).copy()

    if len(df) < 10:
        raise ValueError("The dataset needs at least 10 valid songs.")

    df["track_name"] = df["track_name"].fillna("Untitled").astype(str)
    df["artists"] = df["artists"].fillna("Unknown Artist").astype(str)

    if "track_id" not in df.columns:
        df["track_id"] = [f"track_{i}" for i in range(len(df))]
    else:
        df["track_id"] = df["track_id"].fillna("").astype(str)

    # A fixed sample keeps the first version responsive.
    if len(df) > MAX_TRAINING_SONGS:
        df = df.sample(
            n=MAX_TRAINING_SONGS,
            random_state=RANDOM_STATE,
        )

    return df.reset_index(drop=True), features


def load_csv(path):
    """Load a CSV from a local path."""
    return pd.read_csv(path)
