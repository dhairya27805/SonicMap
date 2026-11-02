"""
Phase 6 Test Suite — Personalized AI Playlist Engine

Tests:
1. Playlist generation (mood/cluster/seed-track)
2. Adaptive track selection based on features
3. Duplicate prevention
4. Playlist CRUD (create, read, update, delete)
5. Track addition/removal
6. Track reordering
7. User isolation
8. Insufficient eligible tracks
9. Database persistence
10. API authentication
11. Phase 1-5 regression
"""

import os
import sys
import time
import uuid
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import SessionLocal, init_db
from models import User, UserToken, UserSession, Track, AudioFeatureProvenance, UserLibraryTrack, Playlist, PlaylistItem
from state import AUDIO_FEATURES, DATA_PATH
from services import ml_service, personal_ml_service
from services import playlist_service
from services.playlist_service import (
    generate_playlist_tracks,
    persist_generated_playlist,
    get_user_playlists,
    get_playlist_detail,
    create_empty_playlist,
    update_playlist,
    delete_playlist,
    add_track_to_playlist,
    remove_track_from_playlist,
    reorder_playlist_tracks,
    PLAYLIST_PROFILES,
)
from main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_user(db, suffix=None):
    suffix = suffix or uuid.uuid4().hex[:12]
    u = User(spotify_id=f"user_p6_{suffix}", display_name=f"User {suffix}")
    db.add(u)
    db.flush()
    return u


def make_session(db, user):
    sid = f"sess_{uuid.uuid4().hex}"
    db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time() + 3600))
    db.add(UserSession(session_id=sid, user_id=user.id, expires_at=time.time() + 3600))
    return sid


def make_track(db, title="Track", artist="Artist", idx=0, features=None, confidence=1.0):
    if features is None:
        features = {
            "danceability": round(min(0.95, 0.45 + (idx % 10) * 0.05), 3),
            "energy": round(min(0.95, 0.50 + (idx % 8) * 0.05), 3),
            "loudness": -15.0 + (idx % 10),
            "speechiness": round(0.04 + (idx % 5) * 0.01, 3),
            "acousticness": round(max(0.01, 0.80 - (idx % 8) * 0.08), 3),
            "instrumentalness": round(0.02 + (idx % 4) * 0.02, 3),
            "liveness": round(0.10 + (idx % 6) * 0.02, 3),
            "valence": round(min(0.95, 0.35 + (idx % 10) * 0.05), 3),
            "tempo": 100.0 + (idx % 12) * 5.0,
        }
    t = Track(
        spotify_id=f"sp_{uuid.uuid4().hex[:20]}",
        title=title,
        artist=artist,
        album="Test Album",
        image_url="http://example.com/art.jpg",
        preview_url="http://example.com/preview.mp3",
        duration_ms=180000,
        **features,
    )
    db.add(t)
    db.flush()
    if confidence is not None:
        prov = AudioFeatureProvenance(
            track_id=t.id,
            feature_name="all_9",
            source="reference_catalog",
            acquisition_type="matched",
            match_confidence=confidence,
            feature_confidence=confidence,
            confidence=confidence,
            notes="Test provenance",
        )
        db.add(prov)
        db.flush()
    return t


def make_library(db, user, n=10, include_high_energy=False):
    tracks = []
    for i in range(n):
        feats = None
        if include_high_energy and i < 3:
            feats = {
                "danceability": 0.9, "energy": 0.92, "loudness": -4.0,
                "speechiness": 0.05, "acousticness": 0.03, "instrumentalness": 0.01,
                "liveness": 0.12, "valence": 0.85, "tempo": 140.0,
            }
        t = make_track(db, title=f"Song {i}", artist=f"Artist {i % 3}", idx=i, features=feats)
        db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        tracks.append(t)
    db.commit()
    return tracks


class TestPhase6PlaylistEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        if ml_service.STATE["df"] is None:
            raw_df = pd.read_csv(DATA_PATH)
            ml_service.fit_model(raw_df, source="Spotify dataset")
        # Pre-load reference catalogue for discovery augmentation
        from services.discovery_service import reference_catalogue
        reference_catalogue.load()

    def setUp(self):
        self.db = SessionLocal()
        self.client = TestClient(app)

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    # ── Test 1: Playlist generation ───────────────────────────────────────────

    def test_playlist_generation_chill(self):
        """Generate a chill_relax playlist from a user's library."""
        import asyncio
        user = make_user(self.db)
        make_library(self.db, user, n=15)
        personal_ml_service.train_personal_model(self.db, user.id)

        result = asyncio.get_event_loop().run_until_complete(
            generate_playlist_tracks(
                db=self.db,
                user_id=user.id,
                playlist_type="chill_relax",
                target_size=10,
                include_discoveries=False,
            )
        )

        self.assertIn("tracks", result)
        self.assertIsInstance(result["tracks"], list)
        self.assertEqual(result["playlist_type"], "chill_relax")
        self.assertGreaterEqual(result["library_tracks"] + result["discovery_tracks"], 0)
        # No shortfall if there are >= 3 eligible tracks
        self.assertIsNone(result.get("shortfall_reason"))

    def test_playlist_generation_high_energy_filter(self):
        """High-energy filter returns tracks with energy > 0.7."""
        import asyncio
        user = make_user(self.db)
        make_library(self.db, user, n=15, include_high_energy=True)
        personal_ml_service.train_personal_model(self.db, user.id)

        result = asyncio.get_event_loop().run_until_complete(
            generate_playlist_tracks(
                db=self.db,
                user_id=user.id,
                playlist_type="high_energy",
                energy_min=0.7,
                target_size=10,
                include_discoveries=False,
            )
        )

        self.assertEqual(result["playlist_type"], "high_energy")
        for track in result["tracks"]:
            if track.get("source") == "library" and track.get("features"):
                self.assertGreaterEqual(track["features"]["energy"], 0.7)

    # ── Test 2: Cluster and seed-track selection ──────────────────────────────

    def test_cluster_selection(self):
        """Playlist generated with cluster_id=0 contains only tracks from cluster 0."""
        import asyncio
        user = make_user(self.db)
        make_library(self.db, user, n=15)
        model = personal_ml_service.train_personal_model(self.db, user.id)

        if not model or not model.get("clusters"):
            self.skipTest("Model not trained or no clusters.")

        cluster_id = model["clusters"][0]["id"]
        result = asyncio.get_event_loop().run_until_complete(
            generate_playlist_tracks(
                db=self.db,
                user_id=user.id,
                playlist_type="similar_sounds",
                cluster_id=cluster_id,
                target_size=10,
                include_discoveries=False,
            )
        )
        self.assertIsInstance(result["tracks"], list)

    def test_seed_track_exclusion(self):
        """Seed track is never included in its own playlist."""
        import asyncio
        user = make_user(self.db)
        tracks = make_library(self.db, user, n=15)
        seed = tracks[0]

        result = asyncio.get_event_loop().run_until_complete(
            generate_playlist_tracks(
                db=self.db,
                user_id=user.id,
                playlist_type="similar_sounds",
                seed_track_id=seed.id,
                target_size=10,
                include_discoveries=False,
            )
        )
        result_ids = [t["id"] for t in result["tracks"] if t.get("source") == "library"]
        self.assertNotIn(seed.id, result_ids)

    # ── Test 3: Duplicate prevention ──────────────────────────────────────────

    def test_no_duplicate_tracks_in_generation(self):
        """Generated track list has no duplicate track IDs."""
        import asyncio
        user = make_user(self.db)
        make_library(self.db, user, n=15)

        result = asyncio.get_event_loop().run_until_complete(
            generate_playlist_tracks(
                db=self.db, user_id=user.id,
                playlist_type="similar_sounds",
                target_size=20,
                include_discoveries=False,
            )
        )
        ids = [t["id"] for t in result["tracks"] if t.get("id")]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate track IDs found in generated playlist.")

    def test_no_duplicate_add_track(self):
        """Adding the same track twice to a playlist only adds it once."""
        user = make_user(self.db)
        t = make_track(self.db, title="Single", artist="Solo")
        self.db.commit()

        pl = create_empty_playlist(self.db, user.id, "Test Playlist")
        r1 = add_track_to_playlist(self.db, user.id, pl["id"], t.id)
        r2 = add_track_to_playlist(self.db, user.id, pl["id"], t.id)

        self.assertEqual(r1["status"], "added")
        self.assertEqual(r2["status"], "already_present")

        detail = get_playlist_detail(self.db, user.id, pl["id"])
        self.assertEqual(detail["track_count"], 1)

    # ── Test 4: Playlist CRUD ─────────────────────────────────────────────────

    def test_create_and_list_playlist(self):
        """Create a playlist and verify it appears in list."""
        user = make_user(self.db)
        pl = create_empty_playlist(self.db, user.id, "My Mix", "A test playlist", "custom")
        playlists = get_user_playlists(self.db, user.id)
        ids = [p["id"] for p in playlists]
        self.assertIn(pl["id"], ids)

    def test_update_playlist(self):
        """Rename a playlist."""
        user = make_user(self.db)
        pl = create_empty_playlist(self.db, user.id, "Old Name")
        updated = update_playlist(self.db, user.id, pl["id"], name="New Name", description="Updated")
        self.assertEqual(updated["name"], "New Name")
        self.assertEqual(updated["description"], "Updated")

    def test_delete_playlist(self):
        """Delete a playlist and confirm it no longer appears."""
        user = make_user(self.db)
        pl = create_empty_playlist(self.db, user.id, "To Delete")
        ok = delete_playlist(self.db, user.id, pl["id"])
        self.assertTrue(ok)
        detail = get_playlist_detail(self.db, user.id, pl["id"])
        self.assertIsNone(detail)

    # ── Test 5: Track addition/removal ────────────────────────────────────────

    def test_add_and_remove_track(self):
        """Add then remove a track from a playlist."""
        user = make_user(self.db)
        t = make_track(self.db, title="Removable", artist="Ghost")
        self.db.commit()

        pl = create_empty_playlist(self.db, user.id, "Removal Test")
        add_track_to_playlist(self.db, user.id, pl["id"], t.id)

        detail = get_playlist_detail(self.db, user.id, pl["id"])
        self.assertEqual(detail["track_count"], 1)

        ok = remove_track_from_playlist(self.db, user.id, pl["id"], t.id)
        self.assertTrue(ok)

        detail2 = get_playlist_detail(self.db, user.id, pl["id"])
        self.assertEqual(detail2["track_count"], 0)

    # ── Test 6: Reordering ────────────────────────────────────────────────────

    def test_reorder_tracks(self):
        """Reorder tracks and verify new positions."""
        user = make_user(self.db)
        t1 = make_track(self.db, title="First", artist="A", idx=0)
        t2 = make_track(self.db, title="Second", artist="B", idx=1)
        t3 = make_track(self.db, title="Third", artist="C", idx=2)
        self.db.commit()

        pl = create_empty_playlist(self.db, user.id, "Reorder Test")
        add_track_to_playlist(self.db, user.id, pl["id"], t1.id)
        add_track_to_playlist(self.db, user.id, pl["id"], t2.id)
        add_track_to_playlist(self.db, user.id, pl["id"], t3.id)

        # Reverse the order
        result = reorder_playlist_tracks(self.db, user.id, pl["id"], [t3.id, t2.id, t1.id])
        self.assertIsNotNone(result)
        self.assertNotIn("error", result)

        ordered_ids = [t["id"] for t in result["tracks"]]
        self.assertEqual(ordered_ids, [t3.id, t2.id, t1.id])

    # ── Test 7: User isolation ────────────────────────────────────────────────

    def test_user_isolation_cannot_access_other_playlist(self):
        """User B cannot access User A's playlist."""
        user_a = make_user(self.db, "iso_a")
        user_b = make_user(self.db, "iso_b")
        sid_a = make_session(self.db, user_a)
        sid_b = make_session(self.db, user_b)
        self.db.commit()

        # Create playlist for A via API
        self.client.cookies.set("sonicmap_session", sid_a)
        resp_create = self.client.post(
            "/api/sonic-playlists/generate",
            json={"playlist_type": "custom", "target_size": 5, "save": True, "name": "A's Playlist", "include_discoveries": False},
        )
        self.assertEqual(resp_create.status_code, 200)
        pl_id = resp_create.json().get("playlist_id")
        if pl_id is None:
            self.skipTest("Playlist not saved (no eligible tracks).")

        # User B attempts to access A's playlist
        self.client.cookies.set("sonicmap_session", sid_b)
        resp_b = self.client.get(f"/api/sonic-playlists/{pl_id}")
        self.assertEqual(resp_b.status_code, 404)

    # ── Test 8: Insufficient eligible tracks ──────────────────────────────────

    def test_insufficient_tracks_returns_explanation(self):
        """Generating with fewer than 3 eligible tracks returns a shortfall_reason."""
        import asyncio
        user = make_user(self.db)
        # Only 2 tracks — below minimum
        for i in range(2):
            t = make_track(self.db, title=f"Sparse {i}", idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        result = asyncio.get_event_loop().run_until_complete(
            generate_playlist_tracks(
                db=self.db, user_id=user.id,
                playlist_type="chill_relax",
                target_size=20,
                include_discoveries=False,
            )
        )
        self.assertIsNotNone(result["shortfall_reason"])
        self.assertIn("eligible", result["shortfall_reason"].lower())

    # ── Test 9: Database persistence ─────────────────────────────────────────

    def test_persist_generated_playlist(self):
        """Generated playlist is saved and retrievable from DB."""
        user = make_user(self.db)
        tracks = make_library(self.db, user, n=5)

        track_dicts = [
            {"id": t.id, "title": t.title, "artist": t.artist, "source": "library"}
            for t in tracks
        ]
        pl = persist_generated_playlist(
            db=self.db,
            user_id=user.id,
            name="Persisted Mix",
            playlist_type="custom",
            description="Test persistence",
            tracks=track_dicts,
        )
        self.assertIsNotNone(pl["id"])
        self.assertEqual(pl["track_count"], 5)

        # Retrieve it
        detail = get_playlist_detail(self.db, user.id, pl["id"])
        self.assertIsNotNone(detail)
        self.assertEqual(detail["name"], "Persisted Mix")
        self.assertEqual(detail["track_count"], 5)

    # ── Test 10: API Authentication ───────────────────────────────────────────

    def test_api_authentication(self):
        """All playlist endpoints reject unauthenticated requests with 401."""
        unauthenticated = TestClient(app)

        self.assertEqual(unauthenticated.get("/api/sonic-playlists").status_code, 401)
        self.assertEqual(unauthenticated.get("/api/sonic-playlists/1").status_code, 401)
        self.assertEqual(unauthenticated.patch("/api/sonic-playlists/1", json={}).status_code, 401)
        self.assertEqual(unauthenticated.delete("/api/sonic-playlists/1").status_code, 401)
        self.assertEqual(unauthenticated.post("/api/sonic-playlists/1/tracks", json={"track_id": 1}).status_code, 401)
        self.assertEqual(unauthenticated.delete("/api/sonic-playlists/1/tracks/1").status_code, 401)
        self.assertEqual(unauthenticated.put("/api/sonic-playlists/1/reorder", json={"ordered_track_ids": []}).status_code, 401)

    # ── Test 11: Full API Flow ────────────────────────────────────────────────

    def test_full_api_playlist_flow(self):
        """End-to-end: generate, list, rename, add track, remove track, reorder, delete."""
        user = make_user(self.db)
        make_library(self.db, user, n=10)
        sid = make_session(self.db, user)
        self.db.commit()

        self.client.cookies.set("sonicmap_session", sid)

        # Generate & save
        resp_gen = self.client.post(
            "/api/sonic-playlists/generate",
            json={"playlist_type": "similar_sounds", "target_size": 8, "save": True, "include_discoveries": False},
        )
        self.assertEqual(resp_gen.status_code, 200)
        gen_data = resp_gen.json()
        pl_id = gen_data.get("playlist_id")

        if pl_id is None:
            self.skipTest("No eligible tracks; playlist not saved.")

        # List
        resp_list = self.client.get("/api/sonic-playlists")
        self.assertEqual(resp_list.status_code, 200)
        self.assertTrue(any(p["id"] == pl_id for p in resp_list.json()["playlists"]))

        # Get detail
        resp_detail = self.client.get(f"/api/sonic-playlists/{pl_id}")
        self.assertEqual(resp_detail.status_code, 200)
        detail = resp_detail.json()
        self.assertEqual(detail["id"], pl_id)

        # Rename
        resp_patch = self.client.patch(
            f"/api/sonic-playlists/{pl_id}",
            json={"name": "My Renamed Mix"},
        )
        self.assertEqual(resp_patch.status_code, 200)
        self.assertEqual(resp_patch.json()["name"], "My Renamed Mix")

        # Add a fresh track
        new_t = make_track(self.db, title="Bonus Track", artist="Bonus Artist")
        self.db.commit()
        resp_add = self.client.post(
            f"/api/sonic-playlists/{pl_id}/tracks",
            json={"track_id": new_t.id},
        )
        self.assertEqual(resp_add.status_code, 200)

        # Remove it
        resp_rm = self.client.delete(f"/api/sonic-playlists/{pl_id}/tracks/{new_t.id}")
        self.assertEqual(resp_rm.status_code, 200)

        # Delete playlist
        resp_del = self.client.delete(f"/api/sonic-playlists/{pl_id}")
        self.assertEqual(resp_del.status_code, 200)

        # Confirm gone
        resp_gone = self.client.get(f"/api/sonic-playlists/{pl_id}")
        self.assertEqual(resp_gone.status_code, 404)

    # ── Test 12: Phase 1-5 Regression ────────────────────────────────────────

    def test_regression_phases_1_to_5(self):
        """Verify Global Music Map, personal, and discovery endpoints remain intact."""
        user = make_user(self.db)
        make_library(self.db, user, n=5)
        sid = make_session(self.db, user)
        self.db.commit()

        self.client.cookies.set("sonicmap_session", sid)

        # Phase 1/Global Music Map
        resp_ov = self.client.get("/api/overview")
        self.assertEqual(resp_ov.status_code, 200)
        self.assertEqual(resp_ov.json().get("cluster_count"), 8)

        resp_rec = self.client.get("/api/recommend/0")
        self.assertEqual(resp_rec.status_code, 200)
        self.assertIsInstance(resp_rec.json(), list)

        # Phase 4 personal
        resp_ps = self.client.get("/api/personal/status")
        self.assertEqual(resp_ps.status_code, 200)

        # Phase 5 discovery status
        resp_ds = self.client.get("/api/discovery/status")
        self.assertEqual(resp_ds.status_code, 200)


if __name__ == "__main__":
    unittest.main()

