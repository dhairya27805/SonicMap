import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

# --- Spotify OAuth / Web API configuration -------------------------------
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.environ.get(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/api/auth/callback"
)
# Where the frontend lives, so we can redirect back to it after login.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://127.0.0.1:5173")

# Secret used to sign our own session cookie (NOT the Spotify secret).
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-insecure-secret-change-me")
SESSION_COOKIE_NAME = "sonicmap_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

# Scopes needed: read profile, control playback, manage playlists, read library, read history & top tracks.
SPOTIFY_SCOPES = " ".join(
    [
        "user-read-email",
        "user-read-private",
        "streaming",
        "user-read-playback-state",
        "user-modify-playback-state",
        "playlist-modify-public",
        "playlist-modify-private",
        "user-library-read",
        "playlist-read-private",
        "playlist-read-collaborative",
        "user-top-read",
        "user-read-recently-played",
    ]
)

# In-memory session store: {session_id: {access_token, refresh_token, expires_at, user}}
# NOTE: in-memory storage means sessions are lost on server restart and do not
# scale across multiple processes/workers. For production, replace with Redis
# or a database-backed session store (see README "Production notes").
SESSIONS = {}

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
