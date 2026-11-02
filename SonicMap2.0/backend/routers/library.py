import asyncio
from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db, SessionLocal
from models import User, Track, UserLibraryTrack, Playlist, PlaylistItem
from routers.auth import get_session, get_valid_access_token
from services import spotify_import_service

router = APIRouter(prefix="/api/library", tags=["Library"])


def _get_current_user_db_id(session_data: dict, db: Session) -> int:
    """Helper to safely extract the DB user id for the authenticated session."""
    user_spotify_id = session_data.get("user", {}).get("id")
    if not user_spotify_id:
        raise HTTPException(status_code=401, detail="User Spotify ID not found in session.")

    user = db.query(User).filter(User.spotify_id == user_spotify_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account record not found.")
    return user.id


async def _run_import_in_background(access_token: str, user_id: int):
    """Background task to run import using its own dedicated DB session."""
    with SessionLocal() as db:
        await spotify_import_service.run_full_library_import(access_token, db, user_id)


@router.post("/sync")
async def start_sync(
    background_tasks: BackgroundTasks,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Trigger library synchronization for the authenticated user in background."""
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    access_token = await get_valid_access_token(sonicmap_session)
    user_id = _get_current_user_db_id(session_data, db)

    # Check if an import is already running
    current_status = spotify_import_service.get_user_import_status(user_id)
    if current_status.get("status") == "running":
        return {"status": "running", "message": "Import is already in progress.", "progress": current_status.get("progress")}

    background_tasks.add_task(_run_import_in_background, access_token, user_id)

    return {
        "status": "started",
        "message": "Library synchronization started in background.",
        "user_id": user_id,
    }


@router.get("/status")
def sync_status(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns the current synchronization status and stats for the authenticated user."""
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_id = _get_current_user_db_id(session_data, db)
    status_info = spotify_import_service.get_user_import_status(user_id)

    # If status is idle, compute current DB counts
    if status_info.get("status") == "idle":
        total_unique = (
            db.query(UserLibraryTrack.track_id)
            .filter(UserLibraryTrack.user_id == user_id)
            .distinct()
            .count()
        )
        status_info["stats"]["total_tracks"] = total_unique

    return status_info


@router.get("/tracks")
def list_library_tracks(
    q: str = "",
    source: str = "",  # 'liked', 'playlist', 'recently_played', 'top_tracks'
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Fetch paginated, searchable, and source-filtered tracks from the user's private library."""
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_id = _get_current_user_db_id(session_data, db)

    query = (
        db.query(Track, UserLibraryTrack.source, UserLibraryTrack.added_at)
        .join(UserLibraryTrack, UserLibraryTrack.track_id == Track.id)
        .filter(UserLibraryTrack.user_id == user_id)
    )

    if source:
        query = query.filter(UserLibraryTrack.source == source)

    if q.strip():
        term = f"%{q.strip()}%"
        query = query.filter((Track.title.ilike(term)) | (Track.artist.ilike(term)) | (Track.album.ilike(term)))

    total_count = query.count()
    results = query.order_by(UserLibraryTrack.added_at.desc()).offset(offset).limit(limit).all()

    items = []
    for track, src, added_at in results:
        items.append({
            "id": track.id,
            "spotify_id": track.spotify_id,
            "title": track.title,
            "artist": track.artist,
            "album": track.album,
            "preview_url": track.preview_url,
            "image_url": track.image_url,
            "duration_ms": track.duration_ms,
            "source": src,
            "added_at": added_at.isoformat() if added_at else None,
        })

    return {
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "tracks": items,
    }


@router.get("/stats")
def library_stats(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Returns detailed statistics for the user's private library."""
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_id = _get_current_user_db_id(session_data, db)

    # Unique tracks count
    total_tracks = (
        db.query(UserLibraryTrack.track_id)
        .filter(UserLibraryTrack.user_id == user_id)
        .distinct()
        .count()
    )

    # Breakdown by source
    breakdown = (
        db.query(UserLibraryTrack.source, func.count(UserLibraryTrack.id))
        .filter(UserLibraryTrack.user_id == user_id)
        .group_by(UserLibraryTrack.source)
        .all()
    )
    by_source = {src: count for src, count in breakdown}

    # Imported playlists count
    playlist_count = (
        db.query(Playlist)
        .filter(Playlist.user_id == user_id, Playlist.playlist_type == "spotify_imported")
        .count()
    )

    return {
        "total_tracks": total_tracks,
        "by_source": by_source,
        "imported_playlists": playlist_count,
    }


@router.get("/playlists")
def list_imported_playlists(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """List playlists imported from Spotify for this user."""
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_id = _get_current_user_db_id(session_data, db)

    playlists = (
        db.query(Playlist)
        .filter(Playlist.user_id == user_id)
        .order_by(Playlist.created_at.desc())
        .all()
    )

    return {
        "playlists": [
            {
                "id": pl.id,
                "name": pl.name,
                "description": pl.description,
                "playlist_type": pl.playlist_type,
                "is_public": pl.is_public,
                "spotify_playlist_id": pl.spotify_playlist_id,
                "track_count": len(pl.items),
                "created_at": pl.created_at.isoformat() if pl.created_at else None,
            }
            for pl in playlists
        ]
    }


from services import feature_engine_service

# In-memory background acquisition job status
FEATURE_JOB_STATUS = {}


async def _run_feature_acquisition_job(user_id: int):
    """Background task to run feature acquisition across user's library tracks."""
    FEATURE_JOB_STATUS[user_id] = {
        "status": "running",
        "progress": 0,
        "processed": 0,
        "total": 0,
        "eligible": 0,
    }
    with SessionLocal() as db:
        user_tracks = (
            db.query(Track)
            .join(UserLibraryTrack, UserLibraryTrack.track_id == Track.id)
            .filter(UserLibraryTrack.user_id == user_id)
            .all()
        )
        total = len(user_tracks)
        FEATURE_JOB_STATUS[user_id]["total"] = total

        for idx, track in enumerate(user_tracks):
            await feature_engine_service.acquire_features_for_track(db, track)
            eligible, _ = feature_engine_service.check_clustering_eligibility(track)
            if eligible:
                FEATURE_JOB_STATUS[user_id]["eligible"] += 1
            FEATURE_JOB_STATUS[user_id]["processed"] = idx + 1
            FEATURE_JOB_STATUS[user_id]["progress"] = int(((idx + 1) / max(total, 1)) * 100)

        FEATURE_JOB_STATUS[user_id]["status"] = "completed"


@router.get("/feature-status")
def feature_status(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Returns feature completeness and clustering eligibility stats for user's library,
    plus ML estimator status.
    """
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_id = _get_current_user_db_id(session_data, db)

    user_tracks = (
        db.query(Track)
        .join(UserLibraryTrack, UserLibraryTrack.track_id == Track.id)
        .filter(UserLibraryTrack.user_id == user_id)
        .all()
    )

    total_tracks = len(user_tracks)
    eligible_count = 0
    with_partial_features = 0
    without_features = 0

    for track in user_tracks:
        eligible, _ = feature_engine_service.check_clustering_eligibility(track)
        if eligible:
            eligible_count += 1
        else:
            # Check if any feature is present
            has_any = any(getattr(track, f) is not None for f in feature_engine_service.AUDIO_FEATURES)
            if has_any:
                with_partial_features += 1
            else:
                without_features += 1

    job_info = FEATURE_JOB_STATUS.get(user_id, {"status": "idle", "progress": 0})
    ml_status = feature_engine_service.ml_estimator.get_status()

    return {
        "user_id": user_id,
        "total_library_tracks": total_tracks,
        "eligible_for_clustering": eligible_count,
        "partial_features": with_partial_features,
        "unacquired_features": without_features,
        "min_clustering_confidence": feature_engine_service.MIN_CLUSTERING_CONFIDENCE,
        "job_status": job_info,
        "ml_estimator_status": ml_status,
    }


@router.post("/features/acquire")
async def acquire_features(
    background_tasks: BackgroundTasks,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Trigger background audio-feature acquisition for all tracks in user's library."""
    session_data = get_session(sonicmap_session)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    user_id = _get_current_user_db_id(session_data, db)

    current_job = FEATURE_JOB_STATUS.get(user_id)
    if current_job and current_job.get("status") == "running":
        return {
            "status": "running",
            "message": "Feature acquisition job is already running.",
            "progress": current_job.get("progress"),
        }

    background_tasks.add_task(_run_feature_acquisition_job, user_id)

    return {
        "status": "started",
        "message": "Audio feature acquisition started in background.",
        "user_id": user_id,
    }
