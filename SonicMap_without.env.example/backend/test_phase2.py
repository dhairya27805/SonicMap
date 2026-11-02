import os
import sys
import unittest
import time
import asyncio
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import SessionLocal, init_db
from models import User, Track, UserLibraryTrack, Playlist, PlaylistItem, UserToken, UserSession
from services import spotify_import_service
from routers.auth import get_session


class TestPhase2SpotifyImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()
        # Create test user
        self.user = User(
            spotify_id=f"test_sp_user_{int(time.time()*1000)}",
            display_name="Spotify Tester",
            email="tester@example.com",
            product="premium",
        )
        self.db.add(self.user)
        self.db.flush()

        self.token = UserToken(
            user_id=self.user.id,
            access_token="test_access_token_abc",
            refresh_token="test_refresh_token_xyz",
            expires_at=time.time() + 3600,
            scope="user-library-read user-top-read",
        )
        self.db.add(self.token)
        self.db.commit()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def test_track_deduplication_and_extraction(self):
        """Test track data extraction and database upserting without duplicates."""
        mock_item = {
            "track": {
                "id": "spotify_track_123",
                "name": "Midnight Waves",
                "artists": [{"name": "Synth Artist A"}, {"name": "Producer B"}],
                "album": {"name": "Neon Dreams", "images": [{"url": "https://img.test/cover.jpg"}]},
                "duration_ms": 210000,
                "preview_url": "https://p.scdn.co/mp3-preview/test",
            }
        }
        tinfo = spotify_import_service._extract_track_data(mock_item)
        self.assertIsNotNone(tinfo)
        self.assertEqual(tinfo["title"], "Midnight Waves")
        self.assertEqual(tinfo["artist"], "Synth Artist A, Producer B")
        self.assertEqual(tinfo["spotify_id"], "spotify_track_123")

        # First insert
        t1 = spotify_import_service._upsert_track(self.db, tinfo)
        self.db.commit()
        # Second insert with same spotify_id
        t2 = spotify_import_service._upsert_track(self.db, tinfo)
        self.db.commit()
        self.assertEqual(t1.id, t2.id)

    def test_saved_track_import_with_pagination(self):
        """Test importing saved tracks with mocked paginated response."""
        async def run_test():
            page_1 = {
                "items": [
                    {
                        "track": {
                            "id": "t_01",
                            "name": "Song 1",
                            "artists": [{"name": "Artist 1"}],
                            "album": {"name": "Album 1", "images": []},
                            "duration_ms": 180000,
                        }
                    }
                ],
                "next": "https://api.spotify.com/v1/me/tracks?offset=1&limit=1",
            }
            page_2 = {
                "items": [
                    {
                        "track": {
                            "id": "t_02",
                            "name": "Song 2",
                            "artists": [{"name": "Artist 2"}],
                            "album": {"name": "Album 2", "images": []},
                            "duration_ms": 200000,
                        }
                    }
                ],
                "next": None,
            }

            async def mock_get(client, url, token, params=None):
                if "offset=1" in url or (params and params.get("offset") == 1):
                    return page_2
                return page_1

            with patch("services.spotify_import_service._spotify_get", side_effect=mock_get):
                imported = await spotify_import_service.import_saved_tracks(
                    None, "dummy_token", self.db, self.user.id, max_tracks=5
                )
                self.assertEqual(imported, 2)

                # Verify in database
                tracks = (
                    self.db.query(UserLibraryTrack)
                    .filter(UserLibraryTrack.user_id == self.user.id, UserLibraryTrack.source == "liked")
                    .all()
                )
                self.assertEqual(len(tracks), 2)

        asyncio.run(run_test())

    def test_playlist_import_and_track_association(self):
        """Test importing user playlists and associating tracks with Feb 2026 /items schema and item key."""
        async def run_test():
            playlists_resp = {
                "items": [
                    {
                        "id": "sp_pl_999",
                        "name": "Roadtrip Beats",
                        "description": "Upbeat vibes",
                        "public": True,
                    }
                ],
                "next": None,
            }
            # February 2026 schema: top-level items object, and 'item' key for track
            items_resp = {
                "items": [
                    {
                        "item": {
                            "id": "road_01",
                            "name": "Highway Song",
                            "artists": [{"name": "Drive Band"}],
                            "album": {"name": "Drive LP", "images": []},
                        }
                    }
                ]
            }

            called_urls = []

            async def mock_get(client, url, token, params=None, return_status=False):
                called_urls.append(url)
                if "playlists/sp_pl_999/items" in url:
                    return (items_resp, 200) if return_status else items_resp
                return (playlists_resp, 200) if return_status else playlists_resp

            with patch("services.spotify_import_service._spotify_get", side_effect=mock_get):
                pls, count = await spotify_import_service.import_playlists_and_tracks(
                    None, "dummy_token", self.db, self.user.id
                )
                self.assertEqual(pls, 1)
                self.assertEqual(count, 1)
                self.assertTrue(any("playlists/sp_pl_999/items" in u for u in called_urls))
                # Confirm legacy endpoint /tracks is NEVER called
                self.assertFalse(any("tracks" in u.split("/")[-1] for u in called_urls))

                pl_record = (
                    self.db.query(Playlist)
                    .filter(Playlist.user_id == self.user.id, Playlist.spotify_playlist_id == "sp_pl_999")
                    .first()
                )
                self.assertIsNotNone(pl_record)
                self.assertEqual(pl_record.name, "Roadtrip Beats")
                self.assertEqual(len(pl_record.items), 1)

        asyncio.run(run_test())

    def test_playlist_import_403_and_empty_handling(self):
        """Test that 403 inaccessible playlist items and empty playlists preserve metadata without error."""
        async def run_test():
            playlists_resp = {
                "items": [
                    {
                        "id": "inaccessible_pl_1",
                        "name": "Non-Owned Public Playlist",
                        "description": "User does not own",
                        "public": True,
                    },
                    {
                        "id": "empty_pl_2",
                        "name": "Brand New Empty Playlist",
                        "description": "Zero tracks",
                        "public": False,
                    },
                ],
                "next": None,
            }

            async def mock_get(client, url, token, params=None, return_status=False):
                if "playlists/inaccessible_pl_1/items" in url:
                    # 403 Forbidden under Feb 2026 quota/access policy
                    return (None, 403) if return_status else None
                if "playlists/empty_pl_2/items" in url:
                    # 200 OK with empty items
                    return ({"items": []}, 200) if return_status else {"items": []}
                return (playlists_resp, 200) if return_status else playlists_resp

            with patch("services.spotify_import_service._spotify_get", side_effect=mock_get):
                pls, count = await spotify_import_service.import_playlists_and_tracks(
                    None, "dummy_token", self.db, self.user.id
                )
                self.assertEqual(pls, 2)
                self.assertEqual(count, 0)

                # Verify both playlist records exist and have their metadata preserved
                pl1 = (
                    self.db.query(Playlist)
                    .filter(Playlist.user_id == self.user.id, Playlist.spotify_playlist_id == "inaccessible_pl_1")
                    .first()
                )
                self.assertIsNotNone(pl1)
                self.assertEqual(pl1.name, "Non-Owned Public Playlist")
                self.assertEqual(len(pl1.items), 0)

                pl2 = (
                    self.db.query(Playlist)
                    .filter(Playlist.user_id == self.user.id, Playlist.spotify_playlist_id == "empty_pl_2")
                    .first()
                )
                self.assertIsNotNone(pl2)
                self.assertEqual(pl2.name, "Brand New Empty Playlist")
                self.assertEqual(len(pl2.items), 0)

        asyncio.run(run_test())

    def test_recently_played_and_top_tracks_import(self):
        """Test recently played and top tracks ingestion."""
        async def run_test():
            recent_resp = {
                "items": [
                    {
                        "track": {
                            "id": "rec_01",
                            "name": "Recent Track",
                            "artists": [{"name": "Recent Artist"}],
                            "album": {"name": "Recent Album", "images": []},
                        }
                    }
                ]
            }
            top_resp = {
                "items": [
                    {
                        "id": "top_01",
                        "name": "Top Track",
                        "artists": [{"name": "Top Artist"}],
                        "album": {"name": "Top Album", "images": []},
                    }
                ]
            }

            async def mock_get(client, url, token, params=None):
                if "recently-played" in url:
                    return recent_resp
                if "top/tracks" in url:
                    return top_resp
                return None

            with patch("services.spotify_import_service._spotify_get", side_effect=mock_get):
                rec_count = await spotify_import_service.import_recently_played(
                    None, "dummy_token", self.db, self.user.id
                )
                top_count = await spotify_import_service.import_top_tracks(
                    None, "dummy_token", self.db, self.user.id
                )
                self.assertEqual(rec_count, 1)
                self.assertEqual(top_count, 1)

        asyncio.run(run_test())

    def test_partial_failure_handling_and_status(self):
        """Test full import when one endpoint fails (e.g. 403 or network error) while others succeed."""
        async def run_test():
            def side_effect(client, url, token, params=None):
                if "recently-played" in url:
                    raise Exception("Endpoint deprecated or 403 Forbidden")
                if "me/tracks" in url:
                    return {
                        "items": [
                            {
                                "track": {
                                    "id": "saved_ok",
                                    "name": "Saved OK",
                                    "artists": [{"name": "Artist"}],
                                    "album": {"name": "Album", "images": []},
                                }
                            }
                        ],
                        "next": None,
                    }
                return {"items": [], "next": None}

            with patch("services.spotify_import_service._spotify_get", side_effect=side_effect):
                res = await spotify_import_service.run_full_library_import(
                    "dummy_token", self.db, self.user.id
                )
                self.assertEqual(res["status"], "completed")
                self.assertEqual(res["stats"]["liked"], 1)
                self.assertTrue(len(res["errors"]) > 0)
                self.assertIn("Recently played", res["errors"][0])

        asyncio.run(run_test())

    def test_user_library_isolation(self):
        """Verify another user cannot see this user's imported tracks."""
        other_user = User(
            spotify_id=f"other_user_{int(time.time()*1000)}",
            display_name="Other User",
        )
        self.db.add(other_user)
        self.db.flush()

        track = Track(title="Private Track", artist="Secret Artist")
        self.db.add(track)
        self.db.flush()

        link = UserLibraryTrack(user_id=self.user.id, track_id=track.id, source="liked")
        self.db.add(link)
        self.db.commit()

        # Query other user's tracks
        other_tracks = (
            self.db.query(UserLibraryTrack)
            .filter(UserLibraryTrack.user_id == other_user.id)
            .all()
        )
        self.assertEqual(len(other_tracks), 0)


if __name__ == "__main__":
    unittest.main()
