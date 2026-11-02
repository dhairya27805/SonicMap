from fastapi import APIRouter, Cookie, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from models import User
from routers.auth import get_session
from services import personal_ml_service

router = APIRouter(prefix="/api/personal", tags=["Personal ML"])


def _get_authenticated_user_id(session_cookie: str | None, db: Session) -> int:
    """Helper to authenticate and extract the database user ID from the session cookie."""
    session_data = get_session(session_cookie)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_spotify_id = session_data.get("user", {}).get("id")
    if not user_spotify_id:
        raise HTTPException(status_code=401, detail="Spotify user identity missing.")

    user = db.query(User).filter(User.spotify_id == user_spotify_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    return user.id


@router.get("/status")
def personal_model_status(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns training status, eligible track counts, and model diagnostics for My Music Map."""
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    return personal_ml_service.get_personal_clusters_overview(db, user_id)


@router.get("/map")
def personal_map(
    cluster: int = Query(default=-1, ge=-1),
    limit: int = Query(default=2000, ge=10, le=5000),
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns 2D PCA coordinates and cluster assignments for the user's private library."""
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    points = personal_ml_service.get_personal_map_points(db, user_id, cluster_id=cluster, limit=limit)
    return {"points": points, "total": len(points)}


@router.get("/clusters")
def personal_clusters(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns cluster summaries and feature profiles for My Music Map."""
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    return personal_ml_service.get_personal_clusters_overview(db, user_id)


@router.get("/recommendations/{track_id}")
def personal_recommendations(
    track_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Computes cosine similarity recommendations strictly within the user's private eligible library,
    excluding the queried track.
    """
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    model = personal_ml_service.get_personal_model(db, user_id)
    if not model:
        raise HTTPException(status_code=400, detail="Personal model is not trained yet or insufficient tracks.")

    if track_id not in model["df"]["id"].values:
        raise HTTPException(status_code=404, detail="Track ID not found in user's eligible library.")

    recs = personal_ml_service.get_personal_recommendations(db, user_id, track_id=track_id, limit=limit)
    return {"track_id": track_id, "recommendations": recs}


@router.post("/train")
def train_personal(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Explicitly trains or refreshes the user's personalized KMeans and PCA models."""
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    result = personal_ml_service.train_personal_model(db, user_id, force=True)
    return result
