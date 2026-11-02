"""
Phase 7 — Spotify Web Playback integration helpers.

Playback itself happens inside the user's browser via the Spotify Web Playback
SDK (a real Spotify Connect device). The backend only:
  * validates and normalises Spotify track URIs,
  * verifies (honestly) whether tracks are playable in the user's market,
  * forwards Web API player commands (play / pause / shuffle / repeat) and
  * translates Spotify's error responses into clear, typed errors
    (Premium required, device missing, rate limited, ...).

No audio is ever proxied, mocked or fabricated here.
"""

import asyncio
import re
import time

import httpx

from services import spotify_service
from state import SPOTIFY_API_BASE

SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
TRACK_URI_RE = re.compile(r"^spotify:track:([A-Za-z0-9]{22})$")
CONTEXT_URI_RE = re.compile(r"^spotify:(album|playlist|artist):[A-Za-z0-9]{22}$")
OPEN_URL_RE = re.compile(r"open\.spotify\.com/(?:intl-[a-z-]+/)?track/([A-Za-z0-9]{22})")

# Spotify's Web Playback SDK requires these scopes on the access token.
SDK_REQUIRED_SCOPES = {"streaming", "user-read-email", "user-read-private"}
# Web API player control requires these.
CONTROL_REQUIRED_SCOPES = {"user-read-playback-state", "user-modify-playback-state"}

MAX_QUEUE_URIS = 100  # keep play bodies small; Spotify rejects very large uri lists
AVAILABILITY_WINDOW = 20  # how many tracks (from the start index) we verify per play
AVAILABILITY_TTL_SECONDS = 6 * 60 * 60

# (cache_key, spotify_id) -> (expires_at, result_dict)
_availability_cache: dict = {}


class PlaybackError(Exception):
    """Typed playback failure surfaced to the client as {code, message}."""

    def __init__(self, status_code: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retry_after = retry_after

    def to_detail(self) -> dict:
        detail = {"code": self.code, "message": self.message}
        if self.retry_after is not None:
            detail["retry_after"] = self.retry_after
        return detail


def normalize_track_uri(value) -> str | None:
    """Accept a bare Spotify ID, a spotify:track: URI or an open.spotify.com URL."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if SPOTIFY_ID_RE.match(value):
        return f"spotify:track:{value}"
    if TRACK_URI_RE.match(value):
        return value
    match = OPEN_URL_RE.search(value)
    if match:
        return f"spotify:track:{match.group(1)}"
    return None


def scope_report(scope_string: str | None) -> dict:
    """Which playback-related scopes are missing from the stored token (None if unknown)."""
    if not scope_string:
        return {"known": False, "missing": []}
    granted = set(scope_string.split())
    missing = sorted((SDK_REQUIRED_SCOPES | CONTROL_REQUIRED_SCOPES) - granted)
    return {"known": True, "missing": missing}


def map_spotify_error(response: httpx.Response) -> PlaybackError:
    """Translate a Spotify Web API error response into a PlaybackError."""
    status = response.status_code
    reason = ""
    message = ""
    try:
        err = (response.json() or {}).get("error") or {}
        if isinstance(err, dict):
            reason = str(err.get("reason") or "")
            message = str(err.get("message") or "")
        else:
            message = str(err)
    except ValueError:
        message = response.text[:200]

    lowered = f"{reason} {message}".lower()

    if status == 401:
        return PlaybackError(
            401, "spotify_auth", "Your Spotify session expired or was revoked. Please reconnect Spotify."
        )
    if status == 403 and ("premium" in lowered or reason == "PREMIUM_REQUIRED"):
        return PlaybackError(
            403,
            "premium_required",
            "Spotify Premium is required to play music inside SonicMap. "
            "You can still open tracks in the Spotify app.",
        )
    if status == 403:
        return PlaybackError(
            403,
            "restricted",
            f"Spotify refused this playback command{': ' + message if message else ''}.",
        )
    if status == 404:
        return PlaybackError(
            404,
            "device_not_found",
            "The SonicMap web player is not active on Spotify yet. "
            "Wait for it to finish connecting (or reload the page) and try again.",
        )
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        return PlaybackError(
            429,
            "rate_limited",
            "Spotify is rate-limiting requests right now. Please wait a moment and try again.",
            retry_after=int(retry_after) if retry_after and retry_after.isdigit() else None,
        )
    return PlaybackError(
        502,
        "spotify_error",
        f"Spotify playback request failed ({status}){': ' + message if message else ''}.",
    )


async def player_command(
    access_token: str,
    method: str,
    path: str,
    params: dict | None = None,
    json: dict | None = None,
) -> None:
    """Send a /me/player/* command. Raises PlaybackError on any non-success."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.request(
                method,
                f"{SPOTIFY_API_BASE}/me/player{path}",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                json=json,
            )
    except httpx.HTTPError as exc:
        raise PlaybackError(502, "spotify_unreachable", "Could not reach Spotify. Check your connection.") from exc

    if response.status_code not in (200, 202, 204):
        raise map_spotify_error(response)


def _availability_from_track(status: int, payload: dict | None) -> dict:
    if status == 200 and payload:
        if payload.get("is_playable", True) is False:
            reason = (payload.get("restrictions") or {}).get("reason") or "market"
            return {"status": "unavailable", "reason": reason}
        return {"status": "playable", "reason": None}
    if status in (400, 404):
        return {"status": "unavailable", "reason": "not_found"}
    # 401/403/429/5xx: we could not verify — do not claim it is unavailable.
    return {"status": "unverified", "reason": f"http_{status}"}


async def check_availability(access_token: str, spotify_ids: list[str], cache_key: str) -> dict:
    """
    Returns {spotify_id: {"status": playable|unavailable|unverified, "reason": str|None}}.
    Uses GET /tracks/{id}?market=from_token (batch endpoint removed Feb 2026).
    Results are cached per user for a few hours to respect Spotify rate limits.
    """
    now = time.time()
    results: dict = {}
    to_fetch: list[str] = []

    for sid in dict.fromkeys(spotify_ids):
        cached = _availability_cache.get((cache_key, sid))
        if cached and cached[0] > now:
            results[sid] = cached[1]
        else:
            to_fetch.append(sid)

    semaphore = asyncio.Semaphore(5)

    async def _fetch(sid: str):
        async with semaphore:
            try:
                status, payload = await spotify_service.get_track(access_token, sid)
            except httpx.HTTPError:
                results[sid] = {"status": "unverified", "reason": "network"}
                return
        result = _availability_from_track(status, payload)
        results[sid] = result
        if result["status"] != "unverified":
            _availability_cache[(cache_key, sid)] = (now + AVAILABILITY_TTL_SECONDS, result)

    await asyncio.gather(*[_fetch(sid) for sid in to_fetch])
    return results


async def start_playback(
    access_token: str,
    cache_key: str,
    device_id: str,
    uris: list[str] | None = None,
    context_uri: str | None = None,
    offset_index: int = 0,
    position_ms: int = 0,
) -> dict:
    """
    Start playback of a track list (or a Spotify context) on the given device.
    Unplayable tracks near the start of the queue are removed and reported.
    """
    if not device_id or not device_id.strip():
        raise PlaybackError(400, "device_required", "No playback device. The SonicMap player has not connected yet.")

    if context_uri:
        if not CONTEXT_URI_RE.match(context_uri):
            raise PlaybackError(400, "invalid_uri", "context_uri must be a spotify:album|playlist|artist URI.")
        body = {"context_uri": context_uri, "position_ms": position_ms}
        if offset_index:
            body["offset"] = {"position": offset_index}
        await player_command(access_token, "PUT", "/play", params={"device_id": device_id}, json=body)
        return {
            "status": "playing",
            "device_id": device_id,
            "context_uri": context_uri,
            "queued": None,
            "unavailable": [],
            "invalid": [],
            "unverified": 0,
        }

    raw = list(uris or [])
    if not raw:
        raise PlaybackError(400, "no_tracks", "Provide at least one Spotify track to play.")

    normalized: list[str] = []
    invalid: list[str] = []
    start = 0  # offset_index refers to the caller's list; map it past dropped invalid IDs
    for raw_index, item in enumerate(raw):
        if raw_index == offset_index:
            start = len(normalized)
        uri = normalize_track_uri(item)
        if uri:
            normalized.append(uri)
        else:
            invalid.append(str(item)[:80])

    if not normalized:
        raise PlaybackError(422, "no_playable_tracks", "None of these tracks have a valid Spotify track ID.")

    offset_index = min(start, len(normalized) - 1)
    # Rotate the queue so the chosen track comes first, then cap its size.
    queue = (normalized[offset_index:] + normalized[:offset_index])[:MAX_QUEUE_URIS]

    window_ids = [u.split(":")[-1] for u in queue[:AVAILABILITY_WINDOW]]
    availability = await check_availability(access_token, window_ids, cache_key)

    unavailable = []
    playable_queue = []
    for index, uri in enumerate(queue):
        sid = uri.split(":")[-1]
        info = availability.get(sid) if index < AVAILABILITY_WINDOW else None
        if info and info["status"] == "unavailable":
            unavailable.append({"spotify_id": sid, "reason": info["reason"]})
        else:
            playable_queue.append(uri)

    if not playable_queue:
        raise PlaybackError(
            422,
            "no_playable_tracks",
            "None of these tracks are currently playable on Spotify in your market.",
        )

    unverified = sum(
        1
        for uri in playable_queue
        if (availability.get(uri.split(":")[-1]) or {}).get("status") != "playable"
    )

    body = {"uris": playable_queue}
    if position_ms:
        body["position_ms"] = position_ms
    await player_command(access_token, "PUT", "/play", params={"device_id": device_id}, json=body)

    return {
        "status": "playing",
        "device_id": device_id,
        "context_uri": None,
        "queued": len(playable_queue),
        "start_uri": playable_queue[0],
        "unavailable": unavailable,
        "invalid": invalid,
        "unverified": unverified,
    }
