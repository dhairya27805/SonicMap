import os
import sys
import unittest
import time

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import Base, SessionLocal, engine, init_db
from models import (
    User,
    UserToken,
    UserSession,
    Track,
    AudioFeatureProvenance,
    UserLibraryTrack,
    Playlist,
    PlaylistItem,
    PersonalModelMetadata,
)
from routers.auth import get_session
from services import ml_service, recommendation_service
import pandas as pd
from state import DATA_PATH


class TestPhase1Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        # Ensure Global ML model is loaded if not already
        if ml_service.STATE["df"] is None:
            raw_df = pd.read_csv(DATA_PATH)
            ml_service.fit_model(raw_df, source="Spotify dataset")

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def test_database_initialization_and_models(self):
        """Test tables exist and basic query runs without error."""
        users_count = self.db.query(User).count()
        self.assertGreaterEqual(users_count, 0)

    def test_user_creation_and_token_storage(self):
        """Test creating user, user_token, and user_session."""
        spotify_id = f"test_user_{int(time.time()*1000)}"
        user = User(
            spotify_id=spotify_id,
            display_name="Tester",
            email="tester@example.com",
            product="premium",
        )
        self.db.add(user)
        self.db.flush()

        token = UserToken(
            user_id=user.id,
            access_token="fake_access_token_123",
            refresh_token="fake_refresh_token_456",
            expires_at=time.time() + 3600,
            scope="user-read-email",
        )
        self.db.add(token)

        session_id = f"sess_{spotify_id}"
        session = UserSession(
            session_id=session_id,
            user_id=user.id,
            expires_at=time.time() + 7200,
        )
        self.db.add(session)
        self.db.commit()

        # Verify retrieval via get_session
        fetched_session = get_session(session_id)
        self.assertIsNotNone(fetched_session)
        self.assertEqual(fetched_session["user"]["id"], spotify_id)
        self.assertEqual(fetched_session["access_token"], "fake_access_token_123")
        self.assertEqual(fetched_session["refresh_token"], "fake_refresh_token_456")

    def test_user_data_isolation(self):
        """Test that user library items and playlists are isolated between users."""
        user_a = User(spotify_id=f"user_a_{int(time.time()*1000)}", display_name="User A")
        user_b = User(spotify_id=f"user_b_{int(time.time()*1000)}", display_name="User B")
        self.db.add_all([user_a, user_b])
        self.db.flush()

        track = Track(
            spotify_id=f"track_{int(time.time()*1000)}",
            title="Isolated Track",
            artist="Artist X",
            danceability=0.75,
            energy=0.82,
        )
        self.db.add(track)
        self.db.flush()

        # Add track to user_a library
        lib_a = UserLibraryTrack(user_id=user_a.id, track_id=track.id, source="liked")
        # Add playlist to user_b
        pl_b = Playlist(user_id=user_b.id, name="User B Playlist", playlist_type="mood")
        self.db.add_all([lib_a, pl_b])
        self.db.commit()

        # Assert user_a library tracks do not belong to user_b
        tracks_a = self.db.query(UserLibraryTrack).filter(UserLibraryTrack.user_id == user_a.id).all()
        tracks_b = self.db.query(UserLibraryTrack).filter(UserLibraryTrack.user_id == user_b.id).all()
        self.assertEqual(len(tracks_a), 1)
        self.assertEqual(len(tracks_b), 0)

        playlists_a = self.db.query(Playlist).filter(Playlist.user_id == user_a.id).all()
        playlists_b = self.db.query(Playlist).filter(Playlist.user_id == user_b.id).all()
        self.assertEqual(len(playlists_a), 0)
        self.assertEqual(len(playlists_b), 1)

    def test_global_music_map_ml_endpoints(self):
        """Ensure Phase 1 changes didn't affect global ML pipeline."""
        overview = ml_service.get_overview()
        self.assertIn("total_tracks", overview)
        self.assertIn("clusters", overview)
        self.assertEqual(len(overview["clusters"]), 8)

        points = ml_service.get_map_points(limit=100)
        self.assertGreaterEqual(len(points), 100)
        self.assertIn("x", points[0])
        self.assertIn("y", points[0])
        self.assertIn("cluster", points[0])

        recs = recommendation_service.get_playlist("chill", limit=5)
        self.assertEqual(len(recs), 5)


if __name__ == "__main__":
    unittest.main()
