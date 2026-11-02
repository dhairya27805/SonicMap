"""
SonicMap Playlist Router (Phase 6)

Endpoints:
  GET    /api/sonic-playlists                        — list user's playlists
  POST   /api/sonic-playlists/generate               — generate & optionally persist
  GET    /api/sonic-playlists/{playlist_id}          — playlist detail
  PATCH  /api/sonic-playlists/{playlist_id}          — rename/redescribe
  DELETE /api/sonic-playlists/{playlist_id}          — delete
  POST   /api/sonic-playlists/{playlist_id}/tracks   — add a track
  DELETE /api/sonic-playlists/{playlist_id}/tracks/{track_id} — remove track
  PUT    /api/sonic-playlists/{playlist_id}/reorder  — reorder tracks

All endpoints require authenticated user session. Never expose another user's playlists.
"""

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models import User
from routers.auth import get_session
from services import playlist_service

router = APIRouter(prefix="/api/sonic-playlists", tags=["Playlists"])


# ── Auth helper ───────────────────────────────────────────────────────────────

def _get_user_id(session_cookie: str | None, db: Session) -> int:
    session_data = get_session(session_cookie)
    if not session_data:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    spotify_id = session_data.get("user", {}).get("id")
    if not spotify_id:
        raise HTTPException(status_code=401, detail="Spotify identity missing.")
    user = db.query(User).filter(User.spotify_id == spotify_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user.id


# ── Request / response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    playlist_type: str = Field(default="chill_relax")
    name: str | None = Field(default=None)
    cluster_id: int = Field(default=-1, ge=-1)
    seed_track_id: int | None = Field(default=None)
    mood: str | None = Field(default=None)
    energy_min: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_max: float = Field(default=1.0, ge=0.0, le=1.0)
    danceability_min: float = Field(default=0.0, ge=0.0, le=1.0)
    danceability_max: float = Field(default=1.0, ge=0.0, le=1.0)
    tempo_min: float = Field(default=30.0, ge=30.0, le=300.0)
    tempo_max: float = Field(default=300.0, ge=30.0, le=300.0)
    target_size: int = Field(default=20, ge=1, le=100)
    include_discoveries: bool = Field(default=True)
    save: bool = Field(default=True, description="Persist the playlist to the database.")


class UpdatePlaylistRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None)


class AddTrackRequest(BaseModel):
    track_id: int


class ReorderRequest(BaseModel):
    ordered_track_ids: list[int] = Field(
        description="Full ordered list of track IDs currently in the playlist."
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def list_playlists(
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Return all playlists owned by the authenticated user."""
    user_id = _get_user_id(sonicmap_session, db)
    return {"playlists": playlist_service.get_user_playlists(db, user_id)}


@router.post("/generate")
async def generate_playlist(
    body: GenerateRequest,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """
    Generate a playlist using the user's personalized model and preferences.
    If save=True (default), persists the playlist and returns it with a DB id.
    """
    user_id = _get_user_id(sonicmap_session, db)

    result = await playlist_service.generate_playlist_tracks(
        db=db,
        user_id=user_id,
        playlist_type=body.playlist_type,
        cluster_id=body.cluster_id,
        seed_track_id=body.seed_track_id,
        mood=body.mood,
        energy_min=body.energy_min,
        energy_max=body.energy_max,
        danceability_min=body.danceability_min,
        danceability_max=body.danceability_max,
        tempo_min=body.tempo_min,
        tempo_max=body.tempo_max,
        target_size=body.target_size,
        include_discoveries=body.include_discoveries,
    )

    if body.save:
        label = body.name or result["playlist_label"]
        persisted = playlist_service.persist_generated_playlist(
            db=db,
            user_id=user_id,
            name=label,
            playlist_type=body.playlist_type,
            description=result["description"],
            tracks=result["tracks"],
        )
        result["saved"] = True
        result["playlist_id"] = persisted["id"]
        result["saved_tracks"] = persisted["track_count"]
        result["skipped_tracks"] = persisted.get("skipped_tracks", 0)
    else:
        result["saved"] = False

    return result


@router.get("/{playlist_id}")
def get_playlist(
    playlist_id: int,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Return full details of a playlist owned by the authenticated user."""
    user_id = _get_user_id(sonicmap_session, db)
    detail = playlist_service.get_playlist_detail(db, user_id, playlist_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    return detail


@router.patch("/{playlist_id}")
def update_playlist(
    playlist_id: int,
    body: UpdatePlaylistRequest,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Rename or update a playlist's description."""
    user_id = _get_user_id(sonicmap_session, db)
    result = playlist_service.update_playlist(db, user_id, playlist_id, name=body.name, description=body.description)
    if result is None:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    return result


@router.delete("/{playlist_id}")
def delete_playlist(
    playlist_id: int,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Delete a playlist and all its items."""
    user_id = _get_user_id(sonicmap_session, db)
    ok = playlist_service.delete_playlist(db, user_id, playlist_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    return {"status": "deleted", "playlist_id": playlist_id}


@router.post("/{playlist_id}/tracks")
def add_track(
    playlist_id: int,
    body: AddTrackRequest,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Add a track to a playlist."""
    user_id = _get_user_id(sonicmap_session, db)
    result = playlist_service.add_track_to_playlist(db, user_id, playlist_id, body.track_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Playlist or track not found.")
    return result


@router.delete("/{playlist_id}/tracks/{track_id}")
def remove_track(
    playlist_id: int,
    track_id: int,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Remove a track from a playlist."""
    user_id = _get_user_id(sonicmap_session, db)
    ok = playlist_service.remove_track_from_playlist(db, user_id, playlist_id, track_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Playlist or track not found.")
    return {"status": "removed", "playlist_id": playlist_id, "track_id": track_id}


@router.put("/{playlist_id}/reorder")
def reorder_tracks(
    playlist_id: int,
    body: ReorderRequest,
    sonicmap_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Reorder tracks in a playlist. Supply the full ordered list of track IDs."""
    user_id = _get_user_id(sonicmap_session, db)
    result = playlist_service.reorder_playlist_tracks(db, user_id, playlist_id, body.ordered_track_ids)
    if result is None:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

