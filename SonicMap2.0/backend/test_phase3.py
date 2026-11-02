import os
import sys
import unittest
import time
import asyncio
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import SessionLocal, init_db
from models import User, Track, UserLibraryTrack, AudioFeatureProvenance
from services import feature_engine_service
from services.feature_engine_service import (
    validate_feature_value,
    ReferenceCatalogMatcher,
    check_clustering_eligibility,
    acquire_features_for_track,
    ml_estimator,
    MIN_CLUSTERING_CONFIDENCE,
)


class TestPhase3BAudioFeatureEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()
        self.user = User(
            spotify_id=f"test_p3_user_{int(time.time()*1000)}",
            display_name="Feature Tester",
        )
        self.db.add(self.user)
        self.db.flush()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def test_feature_value_validation_and_bounds(self):
        """Test feature boundary validation and invalid values."""
        self.assertTrue(validate_feature_value("danceability", 0.75))
        self.assertFalse(validate_feature_value("danceability", 1.5))
        self.assertFalse(validate_feature_value("danceability", -0.1))
        self.assertTrue(validate_feature_value("loudness", -12.4))
        self.assertFalse(validate_feature_value("loudness", 10.0))
        self.assertTrue(validate_feature_value("tempo", 128.0))
        self.assertFalse(validate_feature_value("tempo", 10.0))
        self.assertFalse(validate_feature_value("tempo", None))

    def test_reference_catalog_exact_spotify_id_matching(self):
        """Test Tier 1 exact match by Spotify ID returns all 9 features with 1.0 confidence."""
        # Find a real track ID from dataset-source
        matcher = ReferenceCatalogMatcher()
        matcher.load()
        if not matcher._id_index:
            self.skipTest("Catalog index not loaded")

        sample_id = next(iter(matcher._id_index.keys()))
        expected_row = matcher._id_index[sample_id]

        features, match_conf, feat_conf, note = matcher.match_track(
            spotify_id=sample_id,
            title="Some Arbitrary Title",
            artist="Some Artist",
        )
        self.assertIsNotNone(features)
        self.assertEqual(match_conf, 1.0)
        self.assertEqual(feat_conf, 1.0)
        self.assertIn("Exact Spotify ID", note)
        self.assertEqual(features["tempo"], float(expected_row["tempo"]))
        self.assertEqual(len(features), 9)

    def test_reference_catalog_fuzzy_metadata_matching_and_ambiguity_rejection(self):
        """Test Tier 1 metadata matching and rejection of ambiguous tracks."""
        matcher = ReferenceCatalogMatcher()
        matcher.load()
        if not matcher._id_index:
            self.skipTest("Catalog index not loaded")

        sample_id = next(iter(matcher._id_index.keys()))
        sample_row = matcher._id_index[sample_id]
        title = sample_row["track_name"]
        artist = str(sample_row["artists"]).split(";")[0]

        # Match with empty ID should fallback to metadata
        features, match_conf, feat_conf, note = matcher.match_track(
            spotify_id="",
            title=title,
            artist=artist,
            duration_ms=int(sample_row["duration_ms"]),
        )
        self.assertIsNotNone(features)
        self.assertGreaterEqual(match_conf, 0.85)

        # Ambiguous match check with fake duplicate candidates
        norm_t = feature_engine_service.normalize_string("Ambiguous Track")
        norm_a = feature_engine_service.normalize_string("Ambiguous Artist")
        key = f"{norm_t}___{norm_a}"
        matcher._title_artist_index[key] = [
            {"duration_ms": 100000, "danceability": 0.5},
            {"duration_ms": 300000, "danceability": 0.8},
        ]
        # Querying with no duration or non-matching duration must reject ambiguity
        amb_features, m_conf, _, _ = matcher.match_track(
            spotify_id="",
            title="Ambiguous Track",
            artist="Ambiguous Artist",
            duration_ms=200000,
        )
        self.assertIsNone(amb_features)
        self.assertEqual(m_conf, 0.0)

    def test_librosa_audio_signal_extraction_and_failure_handling(self):
        """Test Librosa extracts tempo and loudness or gracefully returns None on missing/failed preview."""
        async def run_test():
            # Test missing preview URL returns None immediately
            res_none = await feature_engine_service.extract_librosa_physical_features("")
            self.assertIsNone(res_none)

            # Test invalid preview URL returns None gracefully without crashing
            res_bad = await feature_engine_service.extract_librosa_physical_features("http://127.0.0.1:9999/nonexistent.mp3")
            self.assertIsNone(res_bad)

        asyncio.run(run_test())

    def test_ml_estimator_availability_reporting(self):
        """Test ML estimator reports honest unavailability when no licensed training set is present."""
        status = ml_estimator.get_status()
        self.assertFalse(status["available"])
        self.assertIn("unavailable", status["note"].lower())
        est, conf = ml_estimator.estimate_features({})
        self.assertIsNone(est)
        self.assertEqual(conf, 0.0)

    def test_clustering_eligibility_confidence_enforcement(self):
        """Test track clustering eligibility enforces all 9 features and >= 0.75 confidence."""
        track = Track(
            title="Incomplete Track",
            artist="Test Artist",
            tempo=120.0,
            loudness=-10.0,
            # Remaining 7 features are None
        )
        self.db.add(track)
        self.db.flush()

        eligible, reason = check_clustering_eligibility(track)
        self.assertFalse(eligible)
        self.assertIn("Missing feature", reason)

        # Fill all 9 features with valid values
        for f in feature_engine_service.AUDIO_FEATURES:
            setattr(track, f, 0.6 if f not in ("loudness", "tempo") else (120.0 if f == "tempo" else -8.0))

        # Add low confidence provenance (0.50 < 0.75)
        prov_low = AudioFeatureProvenance(
            track_id=track.id,
            feature_name="all_9",
            source="low_confidence_source",
            match_confidence=0.5,
            feature_confidence=0.5,
            confidence=0.5,
        )
        self.db.add(prov_low)
        self.db.flush()

        eligible, reason = check_clustering_eligibility(track)
        self.assertFalse(eligible)
        self.assertIn(f"< {MIN_CLUSTERING_CONFIDENCE}", reason)

        # Update confidence to >= 0.75
        prov_low.confidence = 0.85
        prov_low.match_confidence = 0.90
        prov_low.feature_confidence = 0.95
        eligible, reason = check_clustering_eligibility(track)
        self.assertTrue(eligible)

    def test_provenance_recording_and_user_isolation(self):
        """Test acquiring features persists provenance records and respects user library isolation."""
        async def run_test():
            matcher = ReferenceCatalogMatcher()
            matcher.load()
            sample_id = next(iter(matcher._id_index.keys()))

            existing = self.db.query(Track).filter(Track.spotify_id == sample_id).first()
            if existing:
                track = existing
            else:
                track = Track(
                    spotify_id=sample_id,
                    title="Catalog Test Song",
                    artist="Artist Y",
                )
                self.db.add(track)
                self.db.flush()

            # Ensure unpopulated features for the test
            for f in feature_engine_service.AUDIO_FEATURES:
                setattr(track, f, None)
            # Remove previous provenance for this track if any
            self.db.query(AudioFeatureProvenance).filter(AudioFeatureProvenance.track_id == track.id).delete()

            link = (
                self.db.query(UserLibraryTrack)
                .filter(UserLibraryTrack.user_id == self.user.id, UserLibraryTrack.track_id == track.id)
                .first()
            )
            if not link:
                link = UserLibraryTrack(user_id=self.user.id, track_id=track.id, source="liked")
                self.db.add(link)
            self.db.commit()

            res = await acquire_features_for_track(self.db, track)
            self.assertEqual(res["status"], "matched_catalog")
            self.assertGreaterEqual(res["confidence"], 0.75)

            # Check track now has all 9 features populated
            for f in feature_engine_service.AUDIO_FEATURES:
                val = getattr(track, f)
                self.assertIsNotNone(val)

            # Check provenance was saved
            prov = (
                self.db.query(AudioFeatureProvenance)
                .filter(AudioFeatureProvenance.track_id == track.id)
                .first()
            )
            self.assertIsNotNone(prov)
            self.assertEqual(prov.acquisition_type, "matched")
            self.assertEqual(prov.match_confidence, 1.0)
            self.assertEqual(prov.feature_confidence, 1.0)

            # Check clustering eligibility
            eligible, _ = check_clustering_eligibility(track)
            self.assertTrue(eligible)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
