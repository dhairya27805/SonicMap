from typing import Literal

from fastapi import APIRouter, Cookie, HTTPException, Query, Response
from pydantic import BaseModel, Field

from routers.auth import get_session, get_valid_access_token
from services import playback_service, recommendation_service, spotify_service
from services.playback_service import PlaybackError

router = APIRouter(prefix="/api/playback", tags=["Playback"])


class PlayRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)
    uris: list[str] = Field(default_factory=list, max_length=500)
    context_uri: str | None = None
    offset_index: int = Field(default=0, ge=0)
    position_ms: int = Field(default=0, ge=0)


def _raise(exc: PlaybackError):
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    raise HTTPException(status_code=exc.status_code, detail=exc.to_detail(), headers=headers)


def _require_session(sonicmap_session: str | None) -> dict:
    session = get_session(sonicmap_session)
    if session is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "not_connected", "message": "Connect Spotify to use the SonicMap player."},
        )
    return session


def _playback_capability(session: dict) -> dict:
    """
    Honest capability report. Since Spotify's Feb 2026 Web API changes, GET /me
    no longer returns `product` for Development Mode apps, so Premium status is
    often *unknown* until the Web Playback SDK (account_error) or a play request
    (403 PREMIUM_REQUIRED) tells us.
    """
    product = (session.get("user") or {}).get("product")
    scopes = playback_service.scope_report(session.get("scope"))

    if scopes["missing"]:
        reason = "missing_scopes"
    elif product and product != "premium":
        reason = "premium_required"
    else:
        reason = "ready"

    return {
        "authenticated": True,
        "product": product,
        "premium": True if product == "premium" else (False if product else None),
        "missing_scopes": scopes["missing"],
        "sdk_supported": reason == "ready",
        "reason": reason,
    }


@router.get("/status")
async def playback_status(sonicmap_session: str | None = Cookie(default=None)):
    """Whether this session can use the in-browser Spotify player. Never returns tokens."""
    session = get_session(sonicmap_session)
    if session is None:
        return {
            "authenticated": False,
            "product": None,
            "premium": None,
            "missing_scopes": [],
            "sdk_supported": False,
            "reason": "not_connected",
        }
    return _playback_capability(session)


@router.get("/token")
async def playback_token(response: Response, sonicmap_session: str | None = Cookie(default=None)):
    """
    Returns a short-lived Spotify access token for the Web Playback SDK
    to use directly in the browser (Spotify's SDK requires it client-side).
    The refresh_token and client secret never leave the server, and the
    response is marked non-cacheable.
    """
    session = _require_session(sonicmap_session)
    capability = _playback_capability(session)
    if capability["reason"] == "premium_required":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "premium_required",
                "message": "Spotify Premium is required for in-browser playback.",
            },
        )

    access_token = await get_valid_access_token(sonicmap_session)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return {"access_token": access_token}


@router.post("/create-mood-playlist")
async def create_mood_playlist(
    mood: str = "discovery",
    limit: int = 15,
    sonicmap_session: str | None = Cookie(default=None),
):
    """
    Zero-manual-input flow:
    1. Ask the ML engine (KMeans/PCA/cosine-similarity driven mood
       scoring) for `limit` tracks matching `mood`.
    2. Resolve each to a Spotify track URI using the dataset's track_id
       (already real Spotify IDs) -- falling back to a title/artist
       search only if a direct ID lookup fails.
    3. Create a private playlist on the user's account and add them.
    No song titles or search terms are ever typed by the user.
    """
    session = get_session(sonicmap_session)
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    access_token = await get_valid_access_token(sonicmap_session)
    user_id = session["user"]["id"]

    try:
        tracks = recommendation_service.get_playlist(mood=mood, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not tracks:
        raise HTTPException(status_code=404, detail="No tracks available for this mood.")

    track_ids = [t["track_id"] for t in tracks if t.get("track_id")]
    playability = await spotify_service.check_tracks_playable(access_token, track_ids)

    uris = []
    unresolved = []

    for track in tracks:
        tid = track.get("track_id")
        if tid and playability.get(tid):
            uris.append(f"spotify:track:{tid}")
            continue

        # Fallback: search by title/artist for tracks whose stored ID
        # is no longer resolvable (removed, region-locked, etc.).
        found = await spotify_service.search_track_uri(
            access_token, track.get("title", ""), track.get("artist", "")
        )
        if found:
            uris.append(found)
        else:
            unresolved.append(track.get("title", "Unknown track"))

    if not uris:
        raise HTTPException(
            status_code=502,
            detail=(
                "None of the selected tracks could be resolved to playable "
                "Spotify URIs. This can happen if the dataset's tracks are "
                "unavailable in your Spotify market."
            ),
        )

    try:
        playlist = await spotify_service.create_playlist(
            access_token,
            user_id,
            name=f"SonicMap · {mood.capitalize()} Mix",
            description="Generated automatically by SonicMap's mood-based ML engine (KMeans + PCA + cosine similarity).",
        )
        await spotify_service.add_tracks_to_playlist(access_token, playlist["id"], uris)
    except Exception as exc:  # httpx.HTTPStatusError from raise_for_status()
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Spotify rejected the playlist request{f' ({status})' if status else ''}. "
                "Reconnect Spotify to grant playlist permissions and try again."
            ),
        ) from exc

    return {
        "playlist_id": playlist["id"],
        "playlist_url": playlist.get("external_urls", {}).get("spotify"),
        "uris": uris,
        "track_count": len(uris),
        "unresolved_tracks": unresolved,
    }


@router.put("/play")
async def play(payload: PlayRequest, sonicmap_session: str | None = Cookie(default=None)):
    """
    Starts playback on the given Web Playback SDK device.
    Body: {device_id, uris: [spotify IDs | spotify:track: URIs], offset_index}
       or {device_id, context_uri: "spotify:playlist:..."}.
    Unplayable tracks (not found / region-restricted) are removed and reported.
    """
    session = _require_session(sonicmap_session)
    access_token = await get_valid_access_token(sonicmap_session)
    cache_key = str((session.get("user") or {}).get("id") or "anon")

    try:
        return await playback_service.start_playback(
            access_token,
            cache_key,
            device_id=payload.device_id,
            uris=payload.uris,
            context_uri=payload.context_uri,
            offset_index=payload.offset_index,
            position_ms=payload.position_ms,
        )
    except PlaybackError as exc:
        _raise(exc)


@router.put("/pause")
async def pause(
    device_id: str = Query(min_length=1, max_length=128),
    sonicmap_session: str | None = Cookie(default=None),
):
    _require_session(sonicmap_session)
    access_token = await get_valid_access_token(sonicmap_session)
    try:
        await playback_service.player_command(access_token, "PUT", "/pause", params={"device_id": device_id})
    except PlaybackError as exc:
        _raise(exc)
    return {"message": "Playback paused."}


@router.put("/shuffle")
async def shuffle(
    state: bool,
    device_id: str = Query(min_length=1, max_length=128),
    sonicmap_session: str | None = Cookie(default=None),
):
    _require_session(sonicmap_session)
    access_token = await get_valid_access_token(sonicmap_session)
    try:
        await playback_service.player_command(
            access_token,
            "PUT",
            "/shuffle",
            params={"state": "true" if state else "false", "device_id": device_id},
        )
    except PlaybackError as exc:
        _raise(exc)
    return {"shuffle": state}


@router.put("/repeat")
async def repeat(
    state: Literal["off", "track", "context"],
    device_id: str = Query(min_length=1, max_length=128),
    sonicmap_session: str | None = Cookie(default=None),
):
    _require_session(sonicmap_session)
    access_token = await get_valid_access_token(sonicmap_session)
    try:
        await playback_service.player_command(
            access_token, "PUT", "/repeat", params={"state": state, "device_id": device_id}
        )
    except PlaybackError as exc:
        _raise(exc)
    return {"repeat": state}
