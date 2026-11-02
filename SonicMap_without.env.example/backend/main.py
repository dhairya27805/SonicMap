from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # loads backend/.env if present, before state.py reads os.environ

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.auth import router as auth_router
from routers.music import router as music_router
from routers.playback import router as playback_router
from routers.library import router as library_router
from routers.personal import router as personal_router
from routers.discovery import router as discovery_router
from routers.sonic_playlists import router as sonic_playlists_router
from services.ml_service import fit_model
from state import DATA_PATH, FRONTEND_URL, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DATA_PATH.exists():
        raise RuntimeError(
            f"Dataset not found at {DATA_PATH}. Place dataset.csv inside backend/data/."
        )

    from database import init_db
    init_db()
    print("Database initialized successfully.")

    raw_df = pd.read_csv(DATA_PATH)
    fit_model(raw_df, source="Spotify dataset")

    print("SonicMap ML engine loaded successfully.")

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print(
            "WARNING: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set. "
            "Spotify login, playlist creation, and playback will not work until "
            "backend/.env is configured. See backend/.env.example."
        )

    yield

    print("SonicMap backend stopped.")


app = FastAPI(
    title="SonicMap API",
    description="Music discovery powered by unsupervised learning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        FRONTEND_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(music_router)
app.include_router(auth_router)
app.include_router(playback_router)
app.include_router(library_router)
app.include_router(personal_router)
app.include_router(discovery_router)
app.include_router(sonic_playlists_router)


@app.get("/")
def root():
    return {
        "name": "SonicMap",
        "message": "Explore music beyond genres.",
        "docs": "/docs",
    }
