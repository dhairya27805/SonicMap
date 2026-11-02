"""
Thin wrapper around Spotify's Accounts and Web API endpoints.

Handles:
- Authorization Code + PKCE exchange and refresh
- Fetching the logged-in user's profile
- Creating a playlist and adding tracks to it
- Looking up track URIs by (title, artist) when we only have our
  internal dataset's track_id (which may not always be a live,
  playable Spotify ID in every market/catalog state)

IMPORTANT LIMITATION (see README "Spotify API Limitations" section):
Spotify deprecated public access to the `/v1/audio-features` and
`/v1/recommendations` endpoints for most apps in Nov 2024. This service
intentionally does NOT depend on those endpoints. SonicMap's mood/ML
recommendations are computed entirely from the local dataset
(backend/data/dataset.csv), which already carries real Spotify track_ids
collected at dataset-build time. Spotify is used only for identity
(OAuth), playlist creation, and playback (Web Playback SDK) -- not for
fetching fresh per-user audio-feature data.
"""

import base64
import time

import httpx

from state import (
    SPOTIFY_API_BASE,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
    SPOTIFY_TOKEN_URL,
)


class SpotifyAuthError(Exception):
    """Raised when Spotify OAuth/token operations fail."""


def _basic_auth_header():
    raw = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode("utf-8")
    encoded = base64.b64encode(raw).decode("utf-8")
    return {"Authorization": f"Basic {encoded}"}


async def exchange_code_for_token(code: str, code_verifier: str):
    """Exchange an authorization code (PKCE) for access/refresh tokens."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "code_verifier": code_verifier,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            SPOTIFY_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        raise SpotifyAuthError(
            f"Token exchange failed ({response.status_code}): {response.text}"
        )

    payload = response.json()
    payload["expires_at"] = time.time() + payload.get("expires_in", 3600) - 30
    return payload


async def refresh_access_token(refresh_token: str):
    """Use a refresh token to obtain a new access token."""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": SPOTIFY_CLIENT_ID,
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    # Confidential-client flow also allows Basic auth on refresh.
    if SPOTIFY_CLIENT_SECRET:
        headers.update(_basic_auth_header())

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(SPOTIFY_TOKEN_URL, data=data, headers=headers)

    if response.status_code != 200:
        raise SpotifyAuthError(
            f"Token refresh failed ({response.status_code}): {response.text}"
        )

    payload = response.json()
    payload["expires_at"] = time.time() + payload.get("expires_in", 3600) - 30

    # Spotify does not always return a new refresh_token; keep the old one.
    if "refresh_token" not in payload:
        payload["refresh_token"] = refresh_token

    return payload


async def get_current_user(access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(f"{SPOTIFY_API_BASE}/me", headers=headers)

    if response.status_code != 200:
        raise SpotifyAuthError(
            f"Could not fetch Spotify profile ({response.status_code}): {response.text}"
        )

    return response.json()


async def create_playlist(access_token: str, user_id: str, name: str, description: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    body = {
        "name": name,
        "description": description,
        "public": False,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{SPOTIFY_API_BASE}/users/{user_id}/playlists",
            headers=headers,
            json=body,
        )

    if response.status_code not in (200, 201):
        raise SpotifyAuthError(
            f"Could not create playlist ({response.status_code}): {response.text}"
        )

    return response.json()


async def add_tracks_to_playlist(access_token: str, playlist_id: str, track_uris: list):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Spotify allows up to 100 URIs per call; chunk defensively.
    results = []

    async with httpx.AsyncClient(timeout=15) as client:
        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i : i + 100]
            response = await client.post(
                f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks",
                headers=headers,
                json={"uris": chunk},
            )

            if response.status_code not in (200, 201):
                raise SpotifyAuthError(
                    f"Could not add tracks ({response.status_code}): {response.text}"
                )

            results.append(response.json())

    return results


async def search_track_uri(access_token: str, title: str, artist: str):
    """
    Fallback lookup: search Spotify's catalog for a track by title/artist
    when we need to confirm or re-resolve a playable URI (e.g. the
    dataset's track_id is region-restricted or has since been removed).
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    query = f"track:{title} artist:{artist}"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"{SPOTIFY_API_BASE}/search",
            headers=headers,
            params={"q": query, "type": "track", "limit": 1},
        )

    if response.status_code != 200:
        return None

    items = response.json().get("tracks", {}).get("items", [])
    return items[0]["uri"] if items else None


async def check_tracks_playable(access_token: str, track_ids: list):
    """
    Verify which of our dataset's track_ids still resolve to real,
    fetchable Spotify tracks for this user's market. Spotify catalogs
    change over time (tracks get pulled, go region-locked, etc.), so
    this acts as a safety check before handing URIs to the player.
    Returns a dict {track_id: bool}.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    result = {}

    async with httpx.AsyncClient(timeout=15) as client:
        for i in range(0, len(track_ids), 50):
            chunk = [t for t in track_ids[i : i + 50] if t]
            if not chunk:
                continue

            response = await client.get(
                f"{SPOTIFY_API_BASE}/tracks",
                headers=headers,
                params={"ids": ",".join(chunk)},
            )

            if response.status_code != 200:
                for t in chunk:
                    result[t] = False
                continue

            for track in response.json().get("tracks", []):
                if track is None:
                    continue
                result[track["id"]] = True

            for t in chunk:
                result.setdefault(t, False)

    return result
