from fastapi import APIRouter, HTTPException, Query

from services import ml_service, recommendation_service

router = APIRouter(prefix="/api", tags=["Music"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/overview")
def overview():
    try:
        return ml_service.get_overview()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/map")
def music_map(
    limit: int = Query(default=4000, ge=100, le=5000),
    cluster: int = Query(default=-1, ge=-1),
):
    try:
        return {"points": ml_service.get_map_points(limit, cluster)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/songs")
def songs(
    q: str = "",
    limit: int = Query(default=30, ge=1, le=100),
):
    try:
        return {"songs": recommendation_service.search_songs(q, limit)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/recommend/{song_id}")
def recommend(
    song_id: int,
    limit: int = Query(default=10, ge=1, le=30),
):
    try:
        return recommendation_service.get_recommendations(song_id, limit)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/playlist")
def playlist(
    mood: str = "discovery",
    limit: int = Query(default=15, ge=1, le=30),
):
    allowed_moods = {"discovery", "energetic", "chill", "dance", "happy"}

    if mood not in allowed_moods:
        raise HTTPException(
            status_code=400,
            detail=f"Mood must be one of: {', '.join(sorted(allowed_moods))}",
        )

    try:
        return recommendation_service.get_playlist(mood, limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
