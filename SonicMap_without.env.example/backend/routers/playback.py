from fastapi import APIRouter, Cookie, HTTPException
import httpx

from routers.auth import get_session, get_valid_access_token
from services import recommendation_service, spotify_service
from state import SPOTIFY_API_BASE

router = APIRouter(prefix="/api/playback", tags=["Playback"])


@router.get("/token")
async def playback_token(sonicmap_session: str | None = Cookie(default=None)):
    """
    Returns a short-lived Spotify access token for the Web Playback SDK
    to use directly in the browser. The SDK needs the raw token
    client-side; we still keep the refresh_token server-side only.
    """
    access_token = await get_valid_access_token(sonicmap_session)
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

    playlist = await spotify_service.create_playlist(
        access_token,
        user_id,
        name=f"SonicMap · {mood.capitalize()} Mix",
        description="Generated automatically by SonicMap's mood-based ML engine (KMeans + PCA + cosine similarity).",
    )

    await spotify_service.add_tracks_to_playlist(access_token, playlist["id"], uris)

    return {
        "playlist_id": playlist["id"],
        "playlist_url": playlist.get("external_urls", {}).get("spotify"),
        "uris": uris,
        "track_count": len(uris),
        "unresolved_tracks": unresolved,
    }


@router.put("/play")
async def play(
    device_id: str,
    uris: list[str] | None = None,
    context_uri: str | None = None,
    sonicmap_session: str | None = Cookie(default=None),
):
    """
    Starts/resumes playback on the given Web Playback SDK device.
    Pass either `uris` (a list of spotify:track:... URIs) or
    `context_uri` (e.g. a playlist URI), matching Spotify's own API.
    """
    access_token = await get_valid_access_token(sonicmap_session)

    body = {}
    if uris:
        body["uris"] = uris
    if context_uri:
        body["context_uri"] = context_uri

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.put(
            f"{SPOTIFY_API_BASE}/me/player/play",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"device_id": device_id},
            json=body,
        )

    if response.status_code not in (200, 202, 204):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Spotify playback request failed ({response.status_code}): "
                f"{response.text}. Note: playback requires a Spotify "
                f"Premium account."
            ),
        )

    return {"message": "Playback started."}


@router.put("/pause")
async def pause(device_id: str, sonicmap_session: str | None = Cookie(default=None)):
    access_token = await get_valid_access_token(sonicmap_session)

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.put(
            f"{SPOTIFY_API_BASE}/me/player/pause",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"device_id": device_id},
        )

    if response.status_code not in (200, 202, 204):
        raise HTTPException(
            status_code=502, detail=f"Could not pause playback: {response.text}"
        )

    return {"message": "Playback paused."}
