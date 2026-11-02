"""
Phase 7 Test Suite — Spotify Web Playback integration

Spotify's HTTP API is stubbed with httpx.MockTransport *inside these tests only*
so we can verify exactly what SonicMap sends to Spotify and how it reacts to
Spotify's responses. Production code always talks to the real Web API.

Tests:
1.  Track URI normalisation (IDs, URIs, open.spotify.com links, junk)
2.  Spotify error mapping (401 / 403 Premium / 403 other / 404 device / 429 / 5xx)
3.  /api/playback/status capability reporting (no tokens leaked)
4.  /api/playback/token auth, no-store headers, Premium gate, no refresh token
5.  /api/playback/play — unplayable tracks removed + reported, queue rotation
6.  /api/playback/play — Premium-required / device-missing errors surfaced honestly
7.  /api/playback/play — validation (auth, device, invalid IDs)
8.  pause / shuffle / repeat forwarding + validation
9.  Feb 2026 Web API endpoints (/me/playlists, /items, single-track lookups)
10. SonicMap Playlists persist playable discovery tracks (metadata only)
11. Phase 1-6 route regression smoke
"""

import os
import sys
import time
import uuid
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import SessionLocal, init_db
from models import User, UserToken, UserSession, Track, PlaylistItem
from services import playback_service, spotify_service
from services.playback_service import normalize_track_uri, map_spotify_error, PlaybackError
from services.playlist_service import persist_generated_playlist
from main import app

REAL_ASYNC_CLIENT = httpx.AsyncClient
ID_A = "4uLU6hMCjMI75M1A2tKUQC"
ID_B = "7ouMYWpwJ422jRcDASZB7P"
ID_C = "0VjIjW4GlUZAMYd2vXMi3b"
ID_GONE = "1111111111111111111111"
ID_LOCKED = "2222222222222222222222"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_user(db, product=None, scope=None):
    suffix = uuid.uuid4().hex[:12]
    user = User(spotify_id=f"user_p7_{suffix}", display_name=f"P7 {suffix}", product=product)
    db.add(user)
    db.flush()
    sid = f"sess_{uuid.uuid4().hex}"
    db.add(
        UserToken(
            user_id=user.id,
            access_token="tok_p7_access",
            refresh_token="tok_p7_refresh_SECRET",
            expires_at=time.time() + 3600,
            scope=scope,
        )
    )
    db.add(UserSession(session_id=sid, user_id=user.id, expires_at=time.time() + 3600))
    db.commit()
    return user, sid


class FakeSpotify:
    """Records requests and answers like the Spotify Web API."""

    def __init__(self, play_status=204, play_body=None):
        self.requests = []
        self.play_status = play_status
        self.play_body = play_body

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if request.method == "GET" and path.startswith("/v1/tracks/"):
            tid = path.rsplit("/", 1)[-1]
            if tid == ID_GONE:
                return httpx.Response(404, json={"error": {"status": 404, "message": "Non existing id"}})
            if tid == ID_LOCKED:
                return httpx.Response(
                    200, json={"id": tid, "is_playable": False, "restrictions": {"reason": "market"}}
                )
            return httpx.Response(200, json={"id": tid, "is_playable": True})

        if path.startswith("/v1/me/player"):
            if path == "/v1/me/player/play" and self.play_status not in (200, 202, 204):
                return httpx.Response(self.play_status, json=self.play_body or {})
            return httpx.Response(204)

        if request.method == "POST" and path == "/v1/me/playlists":
            return httpx.Response(
                201, json={"id": "pl123", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl123"}}
            )
        if request.method == "POST" and path == "/v1/playlists/pl123/items":
            return httpx.Response(201, json={"snapshot_id": "snap"})

        return httpx.Response(410, json={"error": {"status": 410, "message": f"Removed endpoint {path}"}})

    def calls(self, method, path):
        return [r for r in self.requests if r.method == method and r.url.path == path]


@contextmanager
def fake_spotify(fake: FakeSpotify):
    transport = httpx.MockTransport(fake.handler)

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return REAL_ASYNC_CLIENT(*args, **kwargs)

    with patch.object(httpx, "AsyncClient", factory):
        yield fake


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPhase7Playback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()
        self.client = TestClient(app)
        playback_service._availability_cache.clear()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    # 1 ────────────────────────────────────────────────────────────────────────
    def test_normalize_track_uri(self):
        self.assertEqual(normalize_track_uri(ID_A), f"spotify:track:{ID_A}")
        self.assertEqual(normalize_track_uri(f"spotify:track:{ID_A}"), f"spotify:track:{ID_A}")
        self.assertEqual(
            normalize_track_uri(f"https://open.spotify.com/intl-de/track/{ID_A}?si=x"), f"spotify:track:{ID_A}"
        )
        for junk in ["", "abc", "spotify:album:" + ID_A, None, 42, "sp_" + "x" * 19]:
            self.assertIsNone(normalize_track_uri(junk))

    # 2 ────────────────────────────────────────────────────────────────────────
    def test_spotify_error_mapping(self):
        cases = [
            (httpx.Response(401, json={"error": {"status": 401, "message": "expired"}}), 401, "spotify_auth"),
            (
                httpx.Response(403, json={"error": {"status": 403, "message": "Player command failed: Premium required", "reason": "PREMIUM_REQUIRED"}}),
                403,
                "premium_required",
            ),
            (httpx.Response(403, json={"error": {"status": 403, "message": "Restriction violated"}}), 403, "restricted"),
            (httpx.Response(404, json={"error": {"status": 404, "message": "Device not found", "reason": "NO_ACTIVE_DEVICE"}}), 404, "device_not_found"),
            (httpx.Response(500, text="boom"), 502, "spotify_error"),
        ]
        for response, status, code in cases:
            err = map_spotify_error(response)
            self.assertIsInstance(err, PlaybackError)
            self.assertEqual((err.status_code, err.code), (status, code), response)

        limited = map_spotify_error(httpx.Response(429, headers={"Retry-After": "7"}, json={}))
        self.assertEqual((limited.status_code, limited.code, limited.retry_after), (429, "rate_limited", 7))

    # 3 ────────────────────────────────────────────────────────────────────────
    def test_status_endpoint(self):
        anon = TestClient(app).get("/api/playback/status")
        self.assertEqual(anon.status_code, 200)
        self.assertEqual(anon.json()["reason"], "not_connected")
        self.assertFalse(anon.json()["sdk_supported"])

        # Feb 2026: product usually missing from GET /me -> Premium unknown, SDK attempt allowed
        _, sid = make_user(self.db, product=None)
        self.client.cookies.set("sonicmap_session", sid)
        body = self.client.get("/api/playback/status").json()
        self.assertEqual((body["reason"], body["premium"], body["sdk_supported"]), ("ready", None, True))
        self.assertNotIn("tok_p7", json.dumps(body))

        _, sid_free = make_user(self.db, product="free")
        self.client.cookies.set("sonicmap_session", sid_free)
        body = self.client.get("/api/playback/status").json()
        self.assertEqual((body["reason"], body["premium"], body["sdk_supported"]), ("premium_required", False, False))

        _, sid_scope = make_user(self.db, product="premium", scope="user-read-email user-read-private")
        self.client.cookies.set("sonicmap_session", sid_scope)
        body = self.client.get("/api/playback/status").json()
        self.assertEqual(body["reason"], "missing_scopes")
        self.assertIn("streaming", body["missing_scopes"])

    # 4 ────────────────────────────────────────────────────────────────────────
    def test_token_endpoint(self):
        self.assertEqual(TestClient(app).get("/api/playback/token").status_code, 401)

        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)
        resp = self.client.get("/api/playback/token")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"access_token": "tok_p7_access"})
        self.assertEqual(resp.headers.get("cache-control"), "no-store")
        self.assertNotIn("refresh", resp.text.lower())

        _, sid_free = make_user(self.db, product="free")
        self.client.cookies.set("sonicmap_session", sid_free)
        resp = self.client.get("/api/playback/token")
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["detail"]["code"], "premium_required")

    # 5 ────────────────────────────────────────────────────────────────────────
    def test_play_filters_unavailable_and_rotates_queue(self):
        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)

        with fake_spotify(FakeSpotify()) as fake:
            resp = self.client.put(
                "/api/playback/play",
                json={
                    "device_id": "dev-1",
                    "uris": [ID_A, f"spotify:track:{ID_GONE}", ID_B, ID_LOCKED, "not-an-id", ID_C],
                    "offset_index": 2,
                },
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "playing")
        self.assertEqual(body["start_uri"], f"spotify:track:{ID_B}")
        self.assertEqual(
            sorted(u["spotify_id"] for u in body["unavailable"]), sorted([ID_GONE, ID_LOCKED])
        )
        self.assertEqual(body["invalid"], ["not-an-id"])
        self.assertEqual(body["queued"], 3)
        self.assertEqual(body["unverified"], 0)

        plays = fake.calls("PUT", "/v1/me/player/play")
        self.assertEqual(len(plays), 1)
        self.assertEqual(plays[0].url.params.get("device_id"), "dev-1")
        self.assertEqual(
            json.loads(plays[0].content)["uris"],
            [f"spotify:track:{ID_B}", f"spotify:track:{ID_C}", f"spotify:track:{ID_A}"],
        )
        self.assertEqual(plays[0].headers["authorization"], "Bearer tok_p7_access")
        # Availability uses single-track lookups with the user's market (batch endpoint removed)
        self.assertFalse(fake.calls("GET", "/v1/tracks"))
        self.assertTrue(all(r.url.params.get("market") == "from_token" for r in fake.requests if r.method == "GET"))

    def test_play_all_unavailable_returns_422(self):
        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)
        with fake_spotify(FakeSpotify()) as fake:
            resp = self.client.put("/api/playback/play", json={"device_id": "d", "uris": [ID_GONE, ID_LOCKED]})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["code"], "no_playable_tracks")
        self.assertFalse(fake.calls("PUT", "/v1/me/player/play"))

    def test_play_offset_skips_invalid_entries(self):
        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)
        with fake_spotify(FakeSpotify()):
            resp = self.client.put(
                "/api/playback/play",
                json={"device_id": "d", "uris": ["junk", ID_A, ID_B], "offset_index": 2},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["start_uri"], f"spotify:track:{ID_B}")

    # 6 ────────────────────────────────────────────────────────────────────────
    def test_play_premium_required_and_device_missing(self):
        _, sid = make_user(self.db, product=None)
        self.client.cookies.set("sonicmap_session", sid)

        premium = FakeSpotify(
            play_status=403,
            play_body={"error": {"status": 403, "message": "Player command failed: Premium required", "reason": "PREMIUM_REQUIRED"}},
        )
        with fake_spotify(premium):
            resp = self.client.put("/api/playback/play", json={"device_id": "d", "uris": [ID_A]})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["detail"]["code"], "premium_required")
        self.assertIn("Premium", resp.json()["detail"]["message"])

        missing = FakeSpotify(play_status=404, play_body={"error": {"status": 404, "message": "Device not found"}})
        with fake_spotify(missing):
            resp = self.client.put("/api/playback/play", json={"device_id": "d", "uris": [ID_A]})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"]["code"], "device_not_found")

        limited = FakeSpotify(play_status=429, play_body={"error": {"status": 429, "message": "rate"}})
        with fake_spotify(limited):
            resp = self.client.put("/api/playback/play", json={"device_id": "d", "uris": [ID_A]})
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.json()["detail"]["code"], "rate_limited")

    def test_play_context_uri(self):
        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)
        with fake_spotify(FakeSpotify()) as fake:
            ok = self.client.put(
                "/api/playback/play", json={"device_id": "d", "context_uri": f"spotify:playlist:{ID_A}"}
            )
            bad = self.client.put("/api/playback/play", json={"device_id": "d", "context_uri": "spotify:evil:x"})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(json.loads(fake.calls("PUT", "/v1/me/player/play")[0].content)["context_uri"], f"spotify:playlist:{ID_A}")
        self.assertEqual(bad.status_code, 400)

    # 7 ────────────────────────────────────────────────────────────────────────
    def test_play_validation(self):
        anon = TestClient(app).put("/api/playback/play", json={"device_id": "d", "uris": [ID_A]})
        self.assertEqual(anon.status_code, 401)
        self.assertEqual(anon.json()["detail"]["code"], "not_connected")

        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)
        self.assertEqual(self.client.put("/api/playback/play", json={"uris": [ID_A]}).status_code, 422)
        self.assertEqual(self.client.put("/api/playback/play", json={"device_id": "", "uris": [ID_A]}).status_code, 422)
        resp = self.client.put("/api/playback/play", json={"device_id": "d", "uris": []})
        self.assertEqual(resp.status_code, 400)
        resp = self.client.put("/api/playback/play", json={"device_id": "d", "uris": ["sp_fake_local_id"]})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["code"], "no_playable_tracks")

    # 8 ────────────────────────────────────────────────────────────────────────
    def test_pause_shuffle_repeat(self):
        self.assertEqual(TestClient(app).put("/api/playback/pause?device_id=d").status_code, 401)

        _, sid = make_user(self.db, product="premium")
        self.client.cookies.set("sonicmap_session", sid)
        with fake_spotify(FakeSpotify()) as fake:
            self.assertEqual(self.client.put("/api/playback/pause?device_id=d1").status_code, 200)
            self.assertEqual(self.client.put("/api/playback/shuffle?state=true&device_id=d1").json(), {"shuffle": True})
            self.assertEqual(self.client.put("/api/playback/repeat?state=track&device_id=d1").json(), {"repeat": "track"})
            self.assertEqual(self.client.put("/api/playback/repeat?state=forever&device_id=d1").status_code, 422)

        self.assertEqual(fake.calls("PUT", "/v1/me/player/pause")[0].url.params["device_id"], "d1")
        self.assertEqual(fake.calls("PUT", "/v1/me/player/shuffle")[0].url.params["state"], "true")
        self.assertEqual(fake.calls("PUT", "/v1/me/player/repeat")[0].url.params["state"], "track")

    # 9 ────────────────────────────────────────────────────────────────────────
    def test_feb_2026_endpoints(self):
        import asyncio

        async def run():
            playable = await spotify_service.check_tracks_playable("tok", [ID_A, ID_GONE, ID_LOCKED, ID_A])
            playlist = await spotify_service.create_playlist("tok", "someone", "SonicMap · Test", "d")
            await spotify_service.add_tracks_to_playlist("tok", playlist["id"], [f"spotify:track:{ID_A}"])
            return playable, playlist

        with fake_spotify(FakeSpotify()) as fake:
            playable, playlist = asyncio.run(run())

        self.assertEqual(playable, {ID_A: True, ID_GONE: False, ID_LOCKED: False})
        self.assertEqual(playlist["id"], "pl123")
        self.assertEqual(len(fake.calls("POST", "/v1/me/playlists")), 1)
        self.assertEqual(len(fake.calls("POST", "/v1/playlists/pl123/items")), 1)
        self.assertFalse([r for r in fake.requests if "/users/" in r.url.path or r.url.path.endswith("/tracks")])

    # 10 ───────────────────────────────────────────────────────────────────────
    def test_playlist_persists_discovery_tracks_with_spotify_ids(self):
        user, _ = make_user(self.db)
        new_id = uuid.uuid4().hex[:22]
        existing = Track(spotify_id=uuid.uuid4().hex[:22], title="Existing", artist="Known")
        self.db.add(existing)
        self.db.commit()

        result = persist_generated_playlist(
            self.db,
            user.id,
            name="P7 discovery",
            playlist_type="explore_new",
            description="test",
            tracks=[
                {"source": "discovery", "spotify_id": new_id, "title": "Ref Song", "artist": "Ref Artist", "duration_ms": 200000},
                {"source": "discovery", "spotify_id": existing.spotify_id, "title": "Existing", "artist": "Known"},
                {"source": "discovery", "spotify_id": "", "title": "iTunes only", "artist": "X"},
            ],
        )

        self.assertEqual(result["skipped_tracks"], 1)
        saved_ids = {t["spotify_id"] for t in result["tracks"]}
        self.assertEqual(saved_ids, {new_id, existing.spotify_id})
        created = self.db.query(Track).filter(Track.spotify_id == new_id).one()
        self.assertEqual((created.title, created.artist), ("Ref Song", "Ref Artist"))
        # No audio features are invented for metadata-only discovery rows
        self.assertIsNone(created.energy)
        self.assertEqual(self.db.query(PlaylistItem).filter(PlaylistItem.track_id == existing.id).count(), 1)

    # 11 ───────────────────────────────────────────────────────────────────────
    def test_phase_regression_routes(self):
        anon = TestClient(app)
        self.assertEqual(anon.get("/api/health").json(), {"status": "ok"})
        self.assertEqual(anon.get("/api/auth/me").json(), {"authenticated": False})
        for path in ["/api/library/tracks", "/api/personal/status", "/api/sonic-playlists"]:
            self.assertEqual(anon.get(path).status_code, 401, path)


if __name__ == "__main__":
    unittest.main()
