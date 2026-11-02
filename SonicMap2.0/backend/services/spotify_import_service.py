"""
Spotify Library Import Service

Handles importing:
A. Saved/Liked tracks (/v1/me/tracks)
B. User Playlists and their accessible tracks (/v1/me/playlists and /v1/playlists/{id}/tracks)
C. Recently played tracks (/v1/me/player/recently-played)
D. Top tracks (/v1/me/top/tracks)

Features:
- Pagination handling
- Rate limiting (429 Retry-After)
- Error resilience (preserves partial successes if an endpoint has insufficient permissions)
- Deduplication and upserting into shared Track catalog and UserLibraryTrack
"""

import asyncio
import datetime
import logging
import httpx
from sqlalchemy.orm import Session

from models import Track, UserLibraryTrack, Playlist, PlaylistItem, utcnow
from state import SPOTIFY_API_BASE

logger = logging.getLogger("sonicmap.spotify_import")

# In-memory progress tracking for asynchronous import jobs
# { user_id: { "status": "idle"|"running"|"completed"|"failed", "progress": int, "message": str, "stats": dict, "errors": list, "last_synced_at": str } }
IMPORT_STATUS = {}


def get_user_import_status(user_id: int) -> dict:
    return IMPORT_STATUS.get(
        user_id,
        {
            "status": "idle",
            "progress": 0,
            "message": "No import has been started yet.",
            "stats": {"total_tracks": 0, "liked": 0, "playlists": 0, "recently_played": 0, "top_tracks": 0},
            "errors": [],
            "last_synced_at": None,
        },
    )


async def _spotify_get(
    client: httpx.AsyncClient,
    url: str,
    access_token: str,
    params: dict = None,
    max_retries: int = 3,
    return_status: bool = False,
) -> dict | tuple[dict | None, int]:
    headers = {"Authorization": f"Bearer {access_token}"}
    last_status = 0
    for attempt in range(max_retries):
        try:
            res = await client.get(url, headers=headers, params=params)
            last_status = res.status_code
            if res.status_code == 200:
                data = res.json()
                return (data, 200) if return_status else data
            elif res.status_code == 429:
                retry_after = int(res.headers.get("Retry-After", 2))
                logger.warning(f"Rate limited by Spotify on {url}. Sleeping {retry_after}s.")
                await asyncio.sleep(retry_after)
                continue
            elif res.status_code in (401, 403, 404):
                logger.info(f"Spotify API returned {res.status_code} for {url}: {res.text}")
                return (None, res.status_code) if return_status else None
            else:
                logger.warning(f"Spotify request {url} returned {res.status_code}: {res.text}")
                await asyncio.sleep(1)
        except httpx.RequestError as exc:
            logger.error(f"Network error calling Spotify {url}: {exc}")
            await asyncio.sleep(1)
    return (None, last_status) if return_status else None


def _extract_track_data(item: dict) -> dict | None:
    if not item:
        return None
    # Support Feb 2026 migration (.get("item")) as well as legacy/liked wrapper (.get("track")) or direct track
    raw = item.get("item") or item.get("track") or item
    if not raw or not isinstance(raw, dict):
        return None

    # Exclude non-track items or local empty items
    track_id = raw.get("id")
    title = raw.get("name")
    if not title:
        return None

    artists = ", ".join([a.get("name", "") for a in raw.get("artists", []) if a.get("name")])
    album_obj = raw.get("album", {})
    album_name = album_obj.get("name")
    images = album_obj.get("images") or []
    image_url = images[0].get("url") if images else None

    return {
        "spotify_id": track_id,
        "title": title,
        "artist": artists or "Unknown Artist",
        "album": album_name,
        "duration_ms": raw.get("duration_ms"),
        "preview_url": raw.get("preview_url"),
        "image_url": image_url,
    }


def _upsert_track(db: Session, track_info: dict) -> Track:
    """Find existing track by spotify_id or (title, artist) to avoid duplicate metadata."""
    track = None
    if track_info.get("spotify_id"):
        track = db.query(Track).filter(Track.spotify_id == track_info["spotify_id"]).first()

    if not track:
        track = (
            db.query(Track)
            .filter(
                Track.title == track_info["title"],
                Track.artist == track_info["artist"],
            )
            .first()
        )

    if not track:
        track = Track(
            spotify_id=track_info.get("spotify_id"),
            title=track_info["title"],
            artist=track_info["artist"],
            album=track_info.get("album"),
            duration_ms=track_info.get("duration_ms"),
            preview_url=track_info.get("preview_url"),
            image_url=track_info.get("image_url"),
        )
        db.add(track)
        db.flush()
    else:
        # Update missing fields if new info arrived
        if not track.spotify_id and track_info.get("spotify_id"):
            track.spotify_id = track_info["spotify_id"]
        if not track.image_url and track_info.get("image_url"):
            track.image_url = track_info["image_url"]
        if not track.preview_url and track_info.get("preview_url"):
            track.preview_url = track_info["preview_url"]

    return track


def _link_user_track(db: Session, user_id: int, track_id: int, source: str):
    """Link track to user's library safely with source deduplication."""
    existing = (
        db.query(UserLibraryTrack)
        .filter(
            UserLibraryTrack.user_id == user_id,
            UserLibraryTrack.track_id == track_id,
            UserLibraryTrack.source == source,
        )
        .first()
    )
    if not existing:
        link = UserLibraryTrack(user_id=user_id, track_id=track_id, source=source)
        db.add(link)


async def import_saved_tracks(client: httpx.AsyncClient, access_token: str, db: Session, user_id: int, max_tracks: int = 150) -> int:
    """Import saved / liked tracks (/v1/me/tracks) with pagination."""
    imported = 0
    url = f"{SPOTIFY_API_BASE}/me/tracks"
    params = {"limit": 50, "offset": 0}

    while url and imported < max_tracks:
        data = await _spotify_get(client, url, access_token, params=params)
        if not data or "items" not in data:
            break

        items = data["items"]
        if not items:
            break

        for item in items:
            tinfo = _extract_track_data(item)
            if tinfo:
                track = _upsert_track(db, tinfo)
                _link_user_track(db, user_id, track.id, source="liked")
                imported += 1
                if imported >= max_tracks:
                    break

        db.commit()
        url = data.get("next")
        params = None  # Next URL already includes offset and limit query params

    return imported


async def import_playlists_and_tracks(client: httpx.AsyncClient, access_token: str, db: Session, user_id: int, max_playlists: int = 10, max_tracks_per_playlist: int = 50) -> tuple[int, int]:
    """Import user playlists and their accessible tracks."""
    playlist_count = 0
    track_count = 0
    url = f"{SPOTIFY_API_BASE}/me/playlists"
    params = {"limit": 20, "offset": 0}

    while url and playlist_count < max_playlists:
        data = await _spotify_get(client, url, access_token, params=params)
        if not data or "items" not in data:
            break

        items = data["items"]
        if not items:
            break

        for pl in items:
            if not pl or not isinstance(pl, dict):
                continue
            sp_pl_id = pl.get("id")
            pl_name = pl.get("name") or "Untitled Playlist"
            pl_desc = pl.get("description")
            is_pub = bool(pl.get("public", False))

            # Find or create playlist record in SonicMap
            existing_pl = (
                db.query(Playlist)
                .filter(Playlist.user_id == user_id, Playlist.spotify_playlist_id == sp_pl_id)
                .first()
            )
            if not existing_pl:
                existing_pl = Playlist(
                    user_id=user_id,
                    name=pl_name,
                    description=pl_desc,
                    playlist_type="spotify_imported",
                    is_public=is_pub,
                    spotify_playlist_id=sp_pl_id,
                )
                db.add(existing_pl)
                db.flush()
            playlist_count += 1

            # Fetch tracks/items for this playlist using exclusively GET /playlists/{id}/items (Feb 2026 official API)
            tracks_url = f"{SPOTIFY_API_BASE}/playlists/{sp_pl_id}/items"
            tr_params = {"limit": 50, "offset": 0}
            tr_data, status_code = await _spotify_get(
                client, tracks_url, access_token, params=tr_params, return_status=True
            )

            if status_code == 403:
                # Under Spotify's Feb 2026 quota and access policies, 403 indicates
                # the user does not own or collaborate on this playlist.
                # Playlist metadata is safely preserved; tracks are marked inaccessible.
                logger.info(
                    f"Playlist '{pl_name}' ({sp_pl_id}) items are inaccessible (403 Forbidden - non-owned/collaborative)."
                )
                db.commit()
                continue

            # Support Feb 2026 schema ({ "items": { "items": [...] } }) or standard ({ "items": [...] })
            items_list = []
            if tr_data and isinstance(tr_data, dict):
                if isinstance(tr_data.get("items"), dict) and "items" in tr_data["items"]:
                    items_list = tr_data["items"]["items"]
                elif isinstance(tr_data.get("items"), list):
                    items_list = tr_data["items"]

            # If items_list is empty, the playlist is either empty or has no accessible tracks;
            # this is NOT an API failure. Playlist metadata is preserved.
            if items_list:
                pos = 0
                for item in items_list[:max_tracks_per_playlist]:
                    tinfo = _extract_track_data(item)
                    if tinfo:
                        track = _upsert_track(db, tinfo)
                        _link_user_track(db, user_id, track.id, source="playlist")

                        # Link to playlist item if not already linked
                        pl_item = (
                            db.query(PlaylistItem)
                            .filter(PlaylistItem.playlist_id == existing_pl.id, PlaylistItem.track_id == track.id)
                            .first()
                        )
                        if not pl_item:
                            pl_item = PlaylistItem(
                                playlist_id=existing_pl.id,
                                track_id=track.id,
                                position=pos,
                            )
                            db.add(pl_item)
                        pos += 1
                        track_count += 1

            db.commit()
            if playlist_count >= max_playlists:
                break

        url = data.get("next")
        params = None

    return playlist_count, track_count


async def import_recently_played(client: httpx.AsyncClient, access_token: str, db: Session, user_id: int, limit: int = 50) -> int:
    """Import recently played tracks (/v1/me/player/recently-played)."""
    url = f"{SPOTIFY_API_BASE}/me/player/recently-played"
    params = {"limit": min(limit, 50)}
    data = await _spotify_get(client, url, access_token, params=params)
    if not data or "items" not in data:
        return 0

    count = 0
    for item in data["items"]:
        tinfo = _extract_track_data(item)
        if tinfo:
            track = _upsert_track(db, tinfo)
            _link_user_track(db, user_id, track.id, source="recently_played")
            count += 1

    db.commit()
    return count


async def import_top_tracks(client: httpx.AsyncClient, access_token: str, db: Session, user_id: int, limit: int = 50) -> int:
    """Import top tracks (/v1/me/top/tracks)."""
    url = f"{SPOTIFY_API_BASE}/me/top/tracks"
    params = {"limit": min(limit, 50), "time_range": "medium_term"}
    data = await _spotify_get(client, url, access_token, params=params)
    if not data or "items" not in data:
        return 0

    count = 0
    for item in data["items"]:
        tinfo = _extract_track_data(item)
        if tinfo:
            track = _upsert_track(db, tinfo)
            _link_user_track(db, user_id, track.id, source="top_tracks")
            count += 1

    db.commit()
    return count


async def run_full_library_import(access_token: str, db: Session, user_id: int) -> dict:
    """Executes the complete multi-source Spotify library import."""
    status_entry = {
        "status": "running",
        "progress": 5,
        "message": "Connecting to Spotify API...",
        "stats": {"total_tracks": 0, "liked": 0, "playlists": 0, "playlist_tracks": 0, "recently_played": 0, "top_tracks": 0},
        "errors": [],
        "last_synced_at": None,
    }
    IMPORT_STATUS[user_id] = status_entry

    async with httpx.AsyncClient(timeout=20) as client:
        # 1. Liked songs
        try:
            status_entry["message"] = "Importing saved/liked songs..."
            status_entry["progress"] = 20
            liked_count = await import_saved_tracks(client, access_token, db, user_id)
            status_entry["stats"]["liked"] = liked_count
        except Exception as exc:
            logger.error(f"Error importing liked tracks: {exc}")
            status_entry["errors"].append(f"Liked tracks: {str(exc)}")

        # 2. Playlists
        try:
            status_entry["message"] = "Importing user playlists and tracks..."
            status_entry["progress"] = 45
            pls, pl_tracks = await import_playlists_and_tracks(client, access_token, db, user_id)
            status_entry["stats"]["playlists"] = pls
            status_entry["stats"]["playlist_tracks"] = pl_tracks
        except Exception as exc:
            logger.error(f"Error importing playlists: {exc}")
            status_entry["errors"].append(f"Playlists: {str(exc)}")

        # 3. Recently played
        try:
            status_entry["message"] = "Importing recently played songs..."
            status_entry["progress"] = 70
            rec_count = await import_recently_played(client, access_token, db, user_id)
            status_entry["stats"]["recently_played"] = rec_count
        except Exception as exc:
            logger.error(f"Error importing recently played: {exc}")
            status_entry["errors"].append(f"Recently played: {str(exc)}")

        # 4. Top tracks
        try:
            status_entry["message"] = "Importing top tracks..."
            status_entry["progress"] = 90
            top_count = await import_top_tracks(client, access_token, db, user_id)
            status_entry["stats"]["top_tracks"] = top_count
        except Exception as exc:
            logger.error(f"Error importing top tracks: {exc}")
            status_entry["errors"].append(f"Top tracks: {str(exc)}")

    # Total unique tracks count for this user in DB
    total_unique = (
        db.query(UserLibraryTrack.track_id)
        .filter(UserLibraryTrack.user_id == user_id)
        .distinct()
        .count()
    )
    status_entry["stats"]["total_tracks"] = total_unique
    status_entry["progress"] = 100
    status_entry["status"] = "completed"
    now_str = utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    status_entry["last_synced_at"] = now_str
    status_entry["message"] = f"Library synced successfully ({total_unique} unique tracks imported)."

    return status_entry
