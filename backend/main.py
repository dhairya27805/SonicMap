from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.music import router as music_router
from routers.upload import router as upload_router
from services.ml_service import fit_model
from state import DATA_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DATA_PATH.exists():
        raise RuntimeError(
            f"Dataset not found at {DATA_PATH}. Place dataset.csv inside backend/data/."
        )

    raw_df = pd.read_csv(DATA_PATH)
    fit_model(raw_df, source="Spotify dataset")

    print("SonicMap ML engine loaded successfully.")

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
        "https://sonicmap-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(music_router)
app.include_router(upload_router)


@app.get("/")
def root():
    return {
        "name": "SonicMap",
        "message": "Explore music beyond genres.",
        "docs": "/docs",
    }
