// Run with: yarn test   (Node's built-in test runner, no extra dependencies)
import assert from "node:assert/strict";
import { test } from "node:test";
import { formatMs, getSpotifyId, spotifyTrackUrl, trackArtist, trackTitle } from "./trackUtils.js";

const ID = "4uLU6hMCjMI75M1A2tKUQC";

test("getSpotifyId reads every SonicMap track shape", () => {
  assert.equal(getSpotifyId({ spotify_id: ID }), ID); // library / map / playlists / discovery
  assert.equal(getSpotifyId({ track_id: ID }), ID); // global dataset rows
  assert.equal(getSpotifyId({ uri: `spotify:track:${ID}` }), ID); // mood playlist uris
  assert.equal(getSpotifyId({ spotify_id: ` ${ID} ` }), ID);
});

test("getSpotifyId rejects tracks that are not on Spotify", () => {
  assert.equal(getSpotifyId(null), null);
  assert.equal(getSpotifyId({ spotify_id: "" }), null); // iTunes / MusicBrainz discovery rows
  assert.equal(getSpotifyId({ spotify_id: "sp_local_fixture_id" }), null);
  assert.equal(getSpotifyId({ uri: `spotify:album:${ID}` }), null);
  assert.equal(getSpotifyId({ track_id: 12345 }), null);
});

test("display helpers", () => {
  assert.equal(spotifyTrackUrl(ID), `https://open.spotify.com/track/${ID}`);
  assert.equal(spotifyTrackUrl(null), "https://open.spotify.com/");
  assert.equal(trackTitle({ track_name: "Song" }), "Song");
  assert.equal(trackTitle({}), "Unknown track");
  assert.equal(trackArtist({ artists: [{ name: "A" }, { name: "B" }] }), "A, B");
  assert.equal(formatMs(0), "0:00");
  assert.equal(formatMs(61000), "1:01");
  assert.equal(formatMs(225999), "3:45");
});
