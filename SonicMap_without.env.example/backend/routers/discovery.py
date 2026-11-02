from fastapi import APIRouter, Cookie, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from models import User
from routers.auth import get_session
from services import discovery_service

router = APIRouter(prefix="/api/discovery", tags=["Discovery"])


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
def discovery_status(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns external catalogue status, provider availability, and user exclusion metrics."""
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    return discovery_service.get_discovery_status(db, user_id)


@router.get("/clusters")
def discovery_clusters(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns available personal clusters for targeted external discovery."""
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    return discovery_service.get_discovery_clusters(db, user_id)


@router.get("/recommendations")
async def discovery_recommendations(
    mode: str = Query(default="similar_to_library", regex="^(similar_to_library|cluster|mood|different)$"),
    cluster_id: int = Query(default=-1, ge=-1),
    mood: str | None = Query(default=None),
    q: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=15, ge=1, le=50),
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Discovers new songs outside the authenticated user's Spotify library.
    Excludes tracks already in user's library and deduplicates by Spotify ID and normalized metadata.
    """
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    return await discovery_service.get_discovery_recommendations(
        db=db,
        user_id=user_id,
        mode=mode,
        cluster_id=cluster_id,
        mood=mood,
        query=q,
        page=page,
        limit=limit,
    )


@router.get("/similar/{track_id}")
async def discovery_similar(
    track_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Finds external new tracks similar to an input track ID,
    strictly excluding the input track and all tracks in user's library.
    """
    user_id = _get_authenticated_user_id(sonicmap_session, db)
    result = await discovery_service.get_similar_track_discovery(
        db=db,
        user_id=user_id,
        track_id=track_id,
        limit=limit,
    )
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Track not found.")
    return result

