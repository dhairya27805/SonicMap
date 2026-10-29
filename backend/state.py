from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

MAX_TRAINING_SONGS = 12000
DEFAULT_CLUSTERS = 8
RANDOM_STATE = 42

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Shared state for the active dataset and trained ML model.
STATE = {
    "df": None,
    "features": [],
    "scaled": None,
    "coordinates": None,
    "clusterer": None,
    "scaler": None,
    "pca": None,
    "silhouette": None,
    "source": "Spotify dataset",
}
