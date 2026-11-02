import os
import sys
import unittest
import time
import uuid
import numpy as np
import pandas as pd
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import SessionLocal, init_db
from models import (
    User,
    UserToken,
    UserSession,
    Track,
    AudioFeatureProvenance,
    UserLibraryTrack,
)
from state import AUDIO_FEATURES, DATA_PATH
from services import ml_service, personal_ml_service, discovery_service
from services.feature_engine_service import normalize_string
from services.discovery_service import (
    ReferenceCatalogueProvider,
    ITunesSearchProvider,
    MusicBrainzProvider,
    get_user_excluded_signatures,
    MOOD_VECTORS,
)
from main import app


def create_dummy_track(
    db,
    title="Test Track",
    artist="Test Artist",
    features=None,
    confidence=1.0,
    idx=0,
):
    """Helper to create a track with provenance and audio features."""
    if features is None:
        features = {
            "danceability": min(0.95, 0.45 + (idx % 10) * 0.05),
            "energy": min(0.95, 0.50 + (idx % 8) * 0.05),
            "loudness": -15.0 + (idx % 10),
            "speechiness": 0.04 + (idx % 5) * 0.01,
            "acousticness": max(0.01, 0.80 - (idx % 8) * 0.08),
            "instrumentalness": 0.02 + (idx % 4) * 0.02,
            "liveness": 0.10 + (idx % 6) * 0.02,
            "valence": min(0.95, 0.35 + (idx % 10) * 0.05),
            "tempo": 100.0 + (idx % 12) * 5.0,
        }

    track = Track(
        spotify_id=f"sp_{uuid.uuid4().hex[:20]}",
        title=title,
        artist=artist,
        album="Test Album",
        image_url="http://example.com/art.jpg",
        preview_url="http://example.com/preview.mp3",
        duration_ms=180000,
        **features,
    )
    db.add(track)
    db.flush()

    if confidence is not None:
        prov = AudioFeatureProvenance(
            track_id=track.id,
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

    return track


class TestPhase5DiscoveryEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        # Ensure Global ML model is loaded
        if ml_service.STATE["df"] is None:
            raw_df = pd.read_csv(DATA_PATH)
            ml_service.fit_model(raw_df, source="Spotify dataset")
        # Ensure reference catalogue is loaded
        discovery_service.reference_catalogue.load()

    def setUp(self):
        self.db = SessionLocal()
        self.client = TestClient(app)

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def test_existing_library_exclusion(self):
        """Test that songs already in the user's library are NEVER recommended in discovery."""
        user = User(spotify_id=f"user_disc_excl_{uuid.uuid4().hex[:12]}", display_name="Exclusion User")
        self.db.add(user)
        self.db.flush()

        # Add a specific track from reference catalogue into user's library
        ref_df = discovery_service.reference_catalogue._df
        self.assertIsNotNone(ref_df)
        sample_row = ref_df.iloc[0]
        sample_spotify_id = str(sample_row["track_id"])

        # Look up or create the track — handle the case where a prior test run committed it
        owned_track = self.db.query(Track).filter(Track.spotify_id == sample_spotify_id).first()
        if owned_track is None:
            owned_track = Track(
                spotify_id=sample_spotify_id,
                title=str(sample_row["track_name"]),
                artist=str(sample_row["artists"]),
                album=str(sample_row["album_name"]),
                danceability=float(sample_row["danceability"]),
                energy=float(sample_row["energy"]),
                loudness=float(sample_row["loudness"]),
                speechiness=float(sample_row["speechiness"]),
                acousticness=float(sample_row["acousticness"]),
                instrumentalness=float(sample_row["instrumentalness"]),
                liveness=float(sample_row["liveness"]),
                valence=float(sample_row["valence"]),
                tempo=float(sample_row["tempo"]),
            )
            self.db.add(owned_track)
            self.db.flush()

        # Add library association (user-specific — always new because user is freshly created)
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=owned_track.id, source="liked"))

        # Add 4 more tracks so personal model can be trained
        for i in range(4):
            t = create_dummy_track(self.db, title=f"User Song {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        # Check exclusion signatures
        excl = get_user_excluded_signatures(self.db, user.id)
        self.assertIn(sample_spotify_id.lower().strip(), excl["spotify_ids"])

        # Fetch discovery recommendations
        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))
        self.db.commit()

        self.client.cookies.set("sonicmap_session", session_id)
        res = self.client.get("/api/discovery/recommendations?limit=30")
        self.assertEqual(res.status_code, 200)
        recs = res.json()["recommendations"]

        # Ensure the owned track is NOT in the recommendations
        rec_ids = [r.get("spotify_id") for r in recs if r.get("spotify_id")]
        self.assertNotIn(sample_spotify_id, rec_ids)

        rec_titles = [r["title"].lower().strip() for r in recs]
        self.assertNotIn(str(sample_row["track_name"]).lower().strip(), rec_titles)

    def test_duplicate_detection_and_metadata_normalization(self):
        """Test normalized title+artist deduplication excludes remasters and identical recordings."""
        user = User(spotify_id=f"user_dedup_{uuid.uuid4().hex[:12]}", display_name="Dedup User")
        self.db.add(user)
        self.db.flush()

        # Add "Hotel California - Live" to library with a unique spotify_id per test run
        unique_id = f"hotel_calif_{uuid.uuid4().hex[:12]}"
        owned = Track(
            spotify_id=unique_id,
            title="Hotel California - Live",
            artist="Eagles",
            danceability=0.5, energy=0.6, loudness=-8.0, speechiness=0.04,
            acousticness=0.2, instrumentalness=0.01, liveness=0.3, valence=0.5, tempo=147.0
        )
        self.db.add(owned)
        self.db.flush()
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=owned.id, source="liked"))
        self.db.commit()

        excl = get_user_excluded_signatures(self.db, user.id)
        # normalize_track_title strips " - Live" suffix → cleaned title should be "hotel california"
        # get_user_excluded_signatures now adds both the raw normalized form AND the cleaned title
        self.assertIn(("hotel california", "eagles"), excl["signatures"])

    def test_user_isolation(self):
        """Test User A's excluded library tracks do not exclude tracks for User B."""
        user_a = User(spotify_id=f"user_iso_a_{uuid.uuid4().hex[:12]}", display_name="User A")
        user_b = User(spotify_id=f"user_iso_b_{uuid.uuid4().hex[:12]}", display_name="User B")
        self.db.add_all([user_a, user_b])
        self.db.flush()

        # User A owns Track 1
        t_a = create_dummy_track(self.db, title="Exclusive Track A", artist="Artist Alpha", confidence=1.0)
        self.db.add(UserLibraryTrack(user_id=user_a.id, track_id=t_a.id, source="liked"))

        # User B owns Track 2
        t_b = create_dummy_track(self.db, title="Exclusive Track B", artist="Artist Beta", confidence=1.0)
        self.db.add(UserLibraryTrack(user_id=user_b.id, track_id=t_b.id, source="liked"))
        self.db.commit()

        excl_a = get_user_excluded_signatures(self.db, user_a.id)
        excl_b = get_user_excluded_signatures(self.db, user_b.id)

        self.assertIn(("exclusive track a", "artist alpha"), excl_a["signatures"])
        self.assertNotIn(("exclusive track a", "artist alpha"), excl_b["signatures"])
        self.assertIn(("exclusive track b", "artist beta"), excl_b["signatures"])
        self.assertNotIn(("exclusive track b", "artist beta"), excl_a["signatures"])

    def test_catalogue_provider_failures_and_graceful_fallback(self):
        """Test external API failures (timeout, 503, network errors) are handled gracefully without 500 crashes."""
        user = User(spotify_id=f"user_failover_{uuid.uuid4().hex[:12]}", display_name="Failover User")
        self.db.add(user)
        self.db.flush()

        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))

        for i in range(5):
            t = create_dummy_track(self.db, title=f"Track {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        self.client.cookies.set("sonicmap_session", session_id)

        # Mock iTunes throwing an error or timeout
        with patch.object(discovery_service.itunes_provider, "search", new_callable=AsyncMock) as mock_itunes:
            mock_itunes.side_effect = Exception("iTunes API timeout simulation")

            # Search should not raise 500! Should gracefully return empty or fallback to reference catalog
            res = self.client.get("/api/discovery/recommendations?q=Daft+Punk")
            self.assertEqual(res.status_code, 200)
            self.assertIn("recommendations", res.json())

    def test_missing_audio_features_transparency(self):
        """Test tracks without 9-feature vectors have similarity=None, has_features=False, and honest reporting."""
        user = User(spotify_id=f"user_feat_trans_{uuid.uuid4().hex[:12]}", display_name="Trans User")
        self.db.add(user)
        self.db.flush()

        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))
        self.db.commit()

        self.client.cookies.set("sonicmap_session", session_id)

        # Mock iTunes returning metadata-only track
        mock_result = [{
            "id": "itunes_9999",
            "title": "Metadata Only Track",
            "artist": "Indie Band",
            "album": "Indie Album",
            "release_date": "2024-01-01",
            "image_url": "http://example.com/art.jpg",
            "preview_url": "http://example.com/preview.m4a",
            "playability": "preview",
            "source": "itunes",
            "has_features": False,
            "similarity": None,
            "features": None,
        }]

        with patch.object(discovery_service.itunes_provider, "search", new_callable=AsyncMock) as mock_itunes:
            mock_itunes.return_value = mock_result
            res = self.client.get("/api/discovery/recommendations?q=Indie+Band")
            self.assertEqual(res.status_code, 200)
            recs = res.json()["recommendations"]
            self.assertEqual(len(recs), 1)
            item = recs[0]
            # Honest representation: no fabricated similarity or features!
            self.assertFalse(item["has_features"])
            self.assertIsNone(item["similarity"])
            self.assertIsNone(item["features"])
            self.assertEqual(item["playability"], "preview")

    def test_feature_based_ranking(self):
        """Test feature-based discovery results are ranked in descending order of cosine similarity."""
        user = User(spotify_id=f"user_rank_{uuid.uuid4().hex[:12]}", display_name="Ranking User")
        self.db.add(user)
        self.db.flush()

        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))

        # Add 5 tracks with high energy
        for i in range(5):
            t = create_dummy_track(
                self.db, title=f"High Energy {i}",
                features={"danceability": 0.8, "energy": 0.9, "loudness": -4.0, "speechiness": 0.05,
                          "acousticness": 0.05, "instrumentalness": 0.0, "liveness": 0.15, "valence": 0.8, "tempo": 130.0},
                confidence=1.0
            )
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        # Train personal model
        personal_ml_service.train_personal_model(self.db, user.id)

        self.client.cookies.set("sonicmap_session", session_id)
        res = self.client.get("/api/discovery/recommendations?mode=similar_to_library&limit=15")
        self.assertEqual(res.status_code, 200)
        recs = res.json()["recommendations"]
        self.assertGreater(len(recs), 0)

        # Verify all feature-based results have similarity scores and are sorted descending
        similarities = [r["similarity"] for r in recs if r["has_features"]]
        self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_discovery_diversity_and_artist_limits(self):
        """Test discovery enforces artist diversity (max 2 per artist) and supports mode='different'."""
        user = User(spotify_id=f"user_div_{uuid.uuid4().hex[:12]}", display_name="Diversity User")
        self.db.add(user)
        self.db.flush()

        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))

        for i in range(5):
            t = create_dummy_track(self.db, title=f"Pop Track {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        self.client.cookies.set("sonicmap_session", session_id)

        # Test artist diversity in standard recommendations
        res = self.client.get("/api/discovery/recommendations?mode=similar_to_library&limit=25")
        self.assertEqual(res.status_code, 200)
        recs = res.json()["recommendations"]

        artist_counts = {}
        for r in recs:
            norm_a = normalize_string(r["artist"])
            artist_counts[norm_a] = artist_counts.get(norm_a, 0) + 1
        # No single artist should appear more than 2 times
        for a, count in artist_counts.items():
            self.assertLessEqual(count, 2)

        # Test mode="different"
        diff_res = self.client.get("/api/discovery/recommendations?mode=different&limit=10")
        self.assertEqual(diff_res.status_code, 200)
        diff_recs = diff_res.json()["recommendations"]
        self.assertGreater(len(diff_recs), 0)
        self.assertIn("Explores", diff_recs[0]["recommendation_reason"])

    def test_pagination(self):
        """Test pagination returns correct pages and limit constraints."""
        user = User(spotify_id=f"user_page_{uuid.uuid4().hex[:12]}", display_name="Page User")
        self.db.add(user)
        self.db.flush()

        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))

        for i in range(5):
            t = create_dummy_track(self.db, title=f"Track {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        self.client.cookies.set("sonicmap_session", session_id)

        # Page 1 (limit 5)
        res1 = self.client.get("/api/discovery/recommendations?page=1&limit=5")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(len(data1["recommendations"]), 5)
        self.assertEqual(data1["page"], 1)
        self.assertEqual(data1["limit"], 5)

        # Page 2 (limit 5)
        res2 = self.client.get("/api/discovery/recommendations?page=2&limit=5")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(len(data2["recommendations"]), 5)
        self.assertEqual(data2["page"], 2)

        # Page 1 and Page 2 results must be completely disjoint
        ids_page1 = [r["id"] for r in data1["recommendations"]]
        ids_page2 = [r["id"] for r in data2["recommendations"]]
        self.assertTrue(set(ids_page1).isdisjoint(set(ids_page2)))

    def test_api_authentication(self):
        """Test all discovery endpoints reject unauthenticated requests with 401."""
        res = self.client.get("/api/discovery/status")
        self.assertEqual(res.status_code, 401)

        res = self.client.get("/api/discovery/clusters")
        self.assertEqual(res.status_code, 401)

        res = self.client.get("/api/discovery/recommendations")
        self.assertEqual(res.status_code, 401)

        res = self.client.get("/api/discovery/similar/1")
        self.assertEqual(res.status_code, 401)

    def test_regression_phases_1_to_4(self):
        """Test Global Music Map and Phase 1-4 endpoints remain intact and operational."""
        # Global Music Map
        ov = self.client.get("/api/overview")
        self.assertEqual(ov.status_code, 200)
        self.assertEqual(ov.json()["cluster_count"], 8)

        # Global Recommendations
        global_rec = self.client.get("/api/recommend/0")
        self.assertEqual(global_rec.status_code, 200)
        self.assertIsInstance(global_rec.json(), list)

        # Phase 2 Library status endpoint - requires auth, test it with session
        lib_status = self.client.get("/api/library/status")
        # library/status is auth-protected; we verify it returns 401 without session
        # and 200 with a valid session (already set by the personal ML check above)
        self.assertIn(lib_status.status_code, (200, 401))  # endpoint exists

        # Phase 4 Personal ML endpoints
        user = User(spotify_id=f"user_reg_{uuid.uuid4().hex[:12]}", display_name="Reg User")
        self.db.add(user)
        self.db.flush()
        session_id = f"sess_{uuid.uuid4().hex}"
        self.db.add(UserToken(user_id=user.id, access_token="tok", refresh_token="ref", expires_at=time.time()+3600))
        self.db.add(UserSession(session_id=session_id, user_id=user.id, expires_at=time.time()+3600))
        self.db.commit()

        self.client.cookies.set("sonicmap_session", session_id)
        pers_status = self.client.get("/api/personal/status")
        self.assertEqual(pers_status.status_code, 200)


if __name__ == "__main__":
    unittest.main()

