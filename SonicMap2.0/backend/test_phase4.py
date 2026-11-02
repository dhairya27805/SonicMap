import os
import sys
import unittest
import time
import uuid
import datetime
import numpy as np
import pandas as pd
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
    PersonalModelMetadata,
    utcnow,
)
from state import AUDIO_FEATURES, DATA_PATH, RANDOM_STATE
from services import ml_service, personal_ml_service
from services.feature_engine_service import check_clustering_eligibility
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


class TestPhase4PersonalML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        # Ensure Global ML model is loaded
        if ml_service.STATE["df"] is None:
            raw_df = pd.read_csv(DATA_PATH)
            ml_service.fit_model(raw_df, source="Spotify dataset")

    def setUp(self):
        self.db = SessionLocal()
        self.client = TestClient(app)

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def test_eligibility_filtering(self):
        """Test that only tracks with complete features, confidence >= 0.75, and valid bounds enter clustering."""
        user = User(spotify_id=f"user_elig_{uuid.uuid4().hex[:12]}", display_name="Elig User")
        self.db.add(user)
        self.db.flush()

        # 1. Valid complete track
        t_valid = create_dummy_track(self.db, title="Valid Song", confidence=0.9, idx=1)
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=t_valid.id, source="liked"))

        # 2. Missing feature (danceability is None)
        feats_missing = {
            "danceability": None,
            "energy": 0.8,
            "loudness": -5.0,
            "speechiness": 0.05,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.5,
            "tempo": 120.0,
        }
        t_missing = create_dummy_track(self.db, title="Missing Feat Song", features=feats_missing, confidence=1.0)
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=t_missing.id, source="liked"))

        # 3. Low confidence (< 0.75)
        t_low_conf = create_dummy_track(self.db, title="Low Conf Song", confidence=0.5, idx=2)
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=t_low_conf.id, source="liked"))

        # 4. Out-of-bounds feature (energy = 1.8 > 1.0)
        feats_oob = {
            "danceability": 0.6,
            "energy": 1.8,
            "loudness": -5.0,
            "speechiness": 0.05,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.5,
            "tempo": 120.0,
        }
        t_oob = create_dummy_track(self.db, title="OOB Song", features=feats_oob, confidence=1.0)
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=t_oob.id, source="liked"))

        self.db.commit()

        # Check eligibility function
        is_ok, _ = check_clustering_eligibility(t_valid)
        self.assertTrue(is_ok)

        is_ok, reason = check_clustering_eligibility(t_missing)
        self.assertFalse(is_ok)
        self.assertIn("danceability", reason)

        is_ok, reason = check_clustering_eligibility(t_low_conf)
        self.assertFalse(is_ok)
        self.assertIn("confidence", reason)

        is_ok, reason = check_clustering_eligibility(t_oob)
        self.assertFalse(is_ok)
        self.assertIn("out of valid bounds", reason)

        # Check get_eligible_user_tracks only returns the valid track
        eligible = personal_ml_service.get_eligible_user_tracks(self.db, user.id)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].id, t_valid.id)

        # Verify ineligible tracks were NOT deleted from user library
        counts = personal_ml_service.get_user_library_counts(self.db, user.id)
        self.assertEqual(counts["total_tracks"], 4)
        self.assertEqual(counts["eligible_tracks"], 1)
        self.assertEqual(counts["awaiting_features"], 3)

    def test_insufficient_eligible_tracks(self):
        """Test clustering behavior when user has fewer than minimum tracks (< 4)."""
        user = User(spotify_id=f"user_insuff_{uuid.uuid4().hex[:12]}", display_name="Small Library")
        self.db.add(user)
        self.db.flush()

        # Add only 2 eligible tracks
        for i in range(2):
            t = create_dummy_track(self.db, title=f"Track {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        train_res = personal_ml_service.train_personal_model(self.db, user.id)
        self.assertEqual(train_res["status"], "insufficient_tracks")
        self.assertEqual(train_res["eligible_tracks"], 2)
        self.assertEqual(train_res["min_required"], 4)

        points = personal_ml_service.get_personal_map_points(self.db, user.id)
        self.assertEqual(points, [])

        overview = personal_ml_service.get_personal_clusters_overview(self.db, user.id)
        self.assertEqual(overview["status"], "insufficient_tracks")
        self.assertEqual(overview["eligible_tracks"], 2)

    def test_adaptive_cluster_selection_and_no_blind_eight(self):
        """Test that cluster count K is chosen adaptively and does not blindly force 8 clusters."""
        user = User(spotify_id=f"user_adapt_{uuid.uuid4().hex[:12]}", display_name="Adaptive User")
        self.db.add(user)
        self.db.flush()

        # Add 6 tracks with two distinct feature profiles (energy vs acoustic)
        for i in range(3):
            # High energy group
            t = create_dummy_track(
                self.db,
                title=f"Rock Track {i}",
                features={
                    "danceability": 0.5 + i * 0.02, "energy": 0.9 + i * 0.01, "loudness": -4.0 + i * 0.2, "speechiness": 0.05,
                    "acousticness": 0.05 + i * 0.01, "instrumentalness": 0.1, "liveness": 0.2, "valence": 0.6, "tempo": 140.0 + i * 2.0
                },
                confidence=1.0,
                idx=i
            )
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))

        for i in range(3):
            # Acoustic mellow group
            t = create_dummy_track(
                self.db,
                title=f"Acoustic Track {i}",
                features={
                    "danceability": 0.4 + i * 0.02, "energy": 0.2 + i * 0.01, "loudness": -15.0 + i * 0.2, "speechiness": 0.04,
                    "acousticness": 0.9 - i * 0.01, "instrumentalness": 0.5, "liveness": 0.1, "valence": 0.3, "tempo": 80.0 + i * 2.0
                },
                confidence=1.0,
                idx=10 + i
            )
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        res = personal_ml_service.train_personal_model(self.db, user.id)
        self.assertEqual(res["status"], "trained")
        # For 6 tracks, K must be adaptively selected (2 or 3), definitely NOT 8!
        self.assertIn(res["cluster_count"], [2, 3])
        self.assertNotEqual(res["cluster_count"], 8)
        self.assertIsNotNone(res["silhouette_score"])
        self.assertGreater(res["silhouette_score"], 0.2)

    def test_degenerate_clustering_prevention(self):
        """Test that degenerate/singleton clusters (size < 2 when avoidable) are avoided."""
        # Create scaled matrix where 8 points cluster naturally into 2 groups of 4
        group1 = np.ones((4, 9)) * 2.0 + np.random.RandomState(RANDOM_STATE).normal(0, 0.05, (4, 9))
        group2 = np.ones((4, 9)) * -2.0 + np.random.RandomState(RANDOM_STATE).normal(0, 0.05, (4, 9))
        matrix = np.vstack([group1, group2])

        k, score = personal_ml_service.select_best_k(matrix, n_samples=8)
        # Should select K=2 because it perfectly splits 4 and 4 without singletons
        self.assertEqual(k, 2)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.5)

    def test_pca_coordinate_generation(self):
        """Test PCA reduces 9 dimensions to 2D coordinates with explained variance ratios."""
        user = User(spotify_id=f"user_pca_{uuid.uuid4().hex[:12]}", display_name="PCA User")
        self.db.add(user)
        self.db.flush()

        for i in range(6):
            t = create_dummy_track(
                self.db,
                title=f"PCA Track {i}",
                features={
                    "danceability": 0.3 + i * 0.1, "energy": 0.4 + i * 0.08, "loudness": -12.0 + i,
                    "speechiness": 0.04 + i * 0.005, "acousticness": 0.8 - i * 0.1, "instrumentalness": 0.1,
                    "liveness": 0.1, "valence": 0.4 + i * 0.05, "tempo": 100.0 + i * 10
                },
                confidence=1.0,
                idx=i
            )
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        personal_ml_service.train_personal_model(self.db, user.id)
        points = personal_ml_service.get_personal_map_points(self.db, user.id)
        self.assertEqual(len(points), 6)

        for p in points:
            self.assertIn("x", p)
            self.assertIn("y", p)
            self.assertIn("cluster", p)
            self.assertTrue(np.isfinite(p["x"]))
            self.assertTrue(np.isfinite(p["y"]))

        # Verify variance ratios stored in metadata
        meta = self.db.query(PersonalModelMetadata).filter(PersonalModelMetadata.user_id == user.id).first()
        self.assertIsNotNone(meta.variance_ratio_pca1)
        self.assertIsNotNone(meta.variance_ratio_pca2)
        self.assertGreaterEqual(meta.variance_ratio_pca1, 0.0)
        self.assertLessEqual(meta.variance_ratio_pca1, 1.0)

    def test_cosine_similarity_ranking_and_exclusion(self):
        """Test cosine similarity ranks results descending and strictly excludes the queried track."""
        user = User(spotify_id=f"user_recs_{uuid.uuid4().hex[:12]}", display_name="Recs User")
        self.db.add(user)
        self.db.flush()

        tracks = []
        # Target track: very high energy dance track
        t_target = create_dummy_track(
            self.db, title="Target Dance Track",
            features={"danceability": 0.9, "energy": 0.9, "loudness": -3.0, "speechiness": 0.05,
                      "acousticness": 0.05, "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.9, "tempo": 130.0},
            confidence=1.0
        )
        tracks.append(t_target)

        # Track very similar to target
        t_similar = create_dummy_track(
            self.db, title="Very Similar Track",
            features={"danceability": 0.88, "energy": 0.89, "loudness": -3.2, "speechiness": 0.06,
                      "acousticness": 0.06, "instrumentalness": 0.0, "liveness": 0.12, "valence": 0.88, "tempo": 128.0},
            confidence=1.0
        )
        tracks.append(t_similar)

        # Track moderately similar
        t_moderate = create_dummy_track(
            self.db, title="Moderate Track",
            features={"danceability": 0.6, "energy": 0.6, "loudness": -7.0, "speechiness": 0.04,
                      "acousticness": 0.3, "instrumentalness": 0.0, "liveness": 0.15, "valence": 0.5, "tempo": 115.0},
            confidence=1.0
        )
        tracks.append(t_moderate)

        # Track totally opposite (classical acoustic)
        t_opposite = create_dummy_track(
            self.db, title="Opposite Acoustic Track",
            features={"danceability": 0.2, "energy": 0.1, "loudness": -25.0, "speechiness": 0.03,
                      "acousticness": 0.95, "instrumentalness": 0.9, "liveness": 0.08, "valence": 0.1, "tempo": 65.0},
            confidence=1.0
        )
        tracks.append(t_opposite)

        # Extra filler track to satisfy min 4 tracks
        t_filler = create_dummy_track(
            self.db, title="Filler Track",
            features={"danceability": 0.5, "energy": 0.5, "loudness": -10.0, "speechiness": 0.05,
                      "acousticness": 0.5, "instrumentalness": 0.1, "liveness": 0.2, "valence": 0.5, "tempo": 100.0},
            confidence=1.0
        )
        tracks.append(t_filler)

        for t in tracks:
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
        self.db.commit()

        recs = personal_ml_service.get_personal_recommendations(self.db, user.id, track_id=t_target.id, limit=5)
        self.assertGreater(len(recs), 0)

        # 1. Target track must be excluded!
        returned_ids = [r["id"] for r in recs]
        self.assertNotIn(t_target.id, returned_ids)

        # 2. Results must be ranked in descending similarity order
        similarities = [r["similarity"] for r in recs]
        self.assertEqual(similarities, sorted(similarities, reverse=True))

        # 3. t_similar must be rank 1
        self.assertEqual(recs[0]["id"], t_similar.id)

    def test_user_specific_dataset_isolation(self):
        """Test strict isolation between different users' private models and recommendations."""
        user_a = User(spotify_id=f"user_a_{uuid.uuid4().hex[:12]}", display_name="User A")
        user_b = User(spotify_id=f"user_b_{uuid.uuid4().hex[:12]}", display_name="User B")
        self.db.add_all([user_a, user_b])
        self.db.flush()

        # User A's tracks
        tracks_a = []
        for i in range(5):
            t = create_dummy_track(self.db, title=f"User A Track {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user_a.id, track_id=t.id, source="liked"))
            tracks_a.append(t)

        # User B's tracks
        tracks_b = []
        for i in range(5):
            t = create_dummy_track(self.db, title=f"User B Track {i}", confidence=1.0, idx=20 + i)
            self.db.add(UserLibraryTrack(user_id=user_b.id, track_id=t.id, source="liked"))
            tracks_b.append(t)
        self.db.commit()

        personal_ml_service.train_personal_model(self.db, user_a.id)
        personal_ml_service.train_personal_model(self.db, user_b.id)

        # Points isolation
        points_a = personal_ml_service.get_personal_map_points(self.db, user_a.id)
        ids_a = {p["id"] for p in points_a}
        points_b = personal_ml_service.get_personal_map_points(self.db, user_b.id)
        ids_b = {p["id"] for p in points_b}

        self.assertEqual(ids_a, {t.id for t in tracks_a})
        self.assertEqual(ids_b, {t.id for t in tracks_b})
        self.assertTrue(ids_a.isdisjoint(ids_b))

        # Recommendations isolation
        recs_a = personal_ml_service.get_personal_recommendations(self.db, user_a.id, track_id=tracks_a[0].id)
        for r in recs_a:
            self.assertIn(r["id"], ids_a)
            self.assertNotIn(r["id"], ids_b)

    def test_model_persistence_and_retraining_lifecycle(self):
        """Test model persistence in PersonalModelMetadata, caching, and automatic staleness detection."""
        user = User(spotify_id=f"user_meta_{uuid.uuid4().hex[:12]}", display_name="Lifecycle User")
        self.db.add(user)
        self.db.flush()

        tracks = []
        for i in range(5):
            t = create_dummy_track(self.db, title=f"Track {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user.id, track_id=t.id, source="liked"))
            tracks.append(t)
        self.db.commit()

        res1 = personal_ml_service.train_personal_model(self.db, user.id)
        v1 = res1["model_version"]

        meta = self.db.query(PersonalModelMetadata).filter(PersonalModelMetadata.user_id == user.id).first()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.track_count, 5)
        self.assertEqual(meta.model_version, v1)

        # Verify caching: get_personal_model returns cached instance without retraining
        model_cached = personal_ml_service.get_personal_model(self.db, user.id)
        self.assertEqual(model_cached["version"], v1)

        # Add a new eligible track to user library
        t_new = create_dummy_track(self.db, title="New Track", confidence=1.0, idx=99)
        self.db.add(UserLibraryTrack(user_id=user.id, track_id=t_new.id, source="liked"))
        self.db.commit()

        # is_model_stale should now detect the library change
        stale, reason = personal_ml_service.is_model_stale(self.db, user.id)
        self.assertTrue(stale)
        self.assertIn("Eligible track count changed", reason)

        # get_personal_model will auto-retrain due to staleness
        model_new = personal_ml_service.get_personal_model(self.db, user.id, auto_train=True)
        self.assertNotEqual(model_new["version"], v1)
        self.assertEqual(model_new["eligible_count"], 6)

    def test_api_authentication_and_user_isolation(self):
        """Test API endpoints enforce authentication and block cross-user data access."""
        # 1. Unauthenticated requests must return 401
        res = self.client.get("/api/personal/status")
        self.assertEqual(res.status_code, 401)

        res = self.client.get("/api/personal/map")
        self.assertEqual(res.status_code, 401)

        res = self.client.get("/api/personal/clusters")
        self.assertEqual(res.status_code, 401)

        res = self.client.get("/api/personal/recommendations/1")
        self.assertEqual(res.status_code, 401)

        res = self.client.post("/api/personal/train")
        self.assertEqual(res.status_code, 401)

        # 2. Authenticated requests with active session
        user_a = User(spotify_id=f"api_user_a_{uuid.uuid4().hex[:12]}", display_name="API User A")
        user_b = User(spotify_id=f"api_user_b_{uuid.uuid4().hex[:12]}", display_name="API User B")
        self.db.add_all([user_a, user_b])
        self.db.flush()

        token_a = UserToken(user_id=user_a.id, access_token="token_a", refresh_token="ref_a", expires_at=time.time()+3600)
        token_b = UserToken(user_id=user_b.id, access_token="token_b", refresh_token="ref_b", expires_at=time.time()+3600)
        self.db.add_all([token_a, token_b])

        session_id_a = f"sess_a_{uuid.uuid4().hex}"
        sess_a = UserSession(
            session_id=session_id_a,
            user_id=user_a.id,
            expires_at=time.time() + 3600,
        )
        self.db.add(sess_a)

        # User A has 5 tracks
        t_a_list = []
        for i in range(5):
            t = create_dummy_track(self.db, title=f"User A Song {i}", confidence=1.0, idx=i)
            self.db.add(UserLibraryTrack(user_id=user_a.id, track_id=t.id, source="liked"))
            t_a_list.append(t)

        # User B has 5 tracks
        t_b_list = []
        for i in range(5):
            t = create_dummy_track(self.db, title=f"User B Song {i}", confidence=1.0, idx=20 + i)
            self.db.add(UserLibraryTrack(user_id=user_b.id, track_id=t.id, source="liked"))
            t_b_list.append(t)
        self.db.commit()

        # Set cookies on client
        self.client.cookies.set("sonicmap_session", session_id_a)

        # Train User A via API
        train_res = self.client.post("/api/personal/train")
        self.assertEqual(train_res.status_code, 200)
        self.assertEqual(train_res.json()["status"], "trained")

        # Map returns User A's tracks
        map_res = self.client.get("/api/personal/map")
        self.assertEqual(map_res.status_code, 200)
        points = map_res.json()["points"]
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertIn("User A", p["title"])

        # Recommendations for User A's track succeeds
        rec_res = self.client.get(f"/api/personal/recommendations/{t_a_list[0].id}")
        self.assertEqual(rec_res.status_code, 200)
        self.assertIn("recommendations", rec_res.json())

        # Recommendations querying User B's track MUST return 404
        rec_b_res = self.client.get(f"/api/personal/recommendations/{t_b_list[0].id}")
        self.assertEqual(rec_b_res.status_code, 404)

    def test_global_music_map_regression(self):
        """Test that Global Music Map endpoints remain 100% operational and unchanged."""
        overview_res = self.client.get("/api/overview")
        self.assertEqual(overview_res.status_code, 200)
        ov = overview_res.json()
        self.assertGreaterEqual(ov["total_tracks"], 1000)
        self.assertEqual(ov["cluster_count"], 8)

        map_res = self.client.get("/api/map?limit=100")
        self.assertEqual(map_res.status_code, 200)
        points = map_res.json()["points"]
        self.assertGreaterEqual(len(points), 100)

        songs_res = self.client.get("/api/songs?limit=10")
        self.assertEqual(songs_res.status_code, 200)
        self.assertEqual(len(songs_res.json()["songs"]), 10)

        rec_res = self.client.get("/api/recommend/0")
        self.assertEqual(rec_res.status_code, 200)
        self.assertIsInstance(rec_res.json(), list)
        self.assertGreater(len(rec_res.json()), 0)


if __name__ == "__main__":
    unittest.main()
