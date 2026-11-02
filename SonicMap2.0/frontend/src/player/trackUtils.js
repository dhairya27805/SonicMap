const SPOTIFY_ID_RE = /^[A-Za-z0-9]{22}$/;
const TRACK_URI_RE = /^spotify:track:([A-Za-z0-9]{22})$/;

// Real Spotify track ID for a SonicMap track object (library, map, dataset,
// discovery or playlist item), or null when the track is not on Spotify.
export function getSpotifyId(track) {
  if (!track) return null;
  for (const candidate of [track.spotify_id, track.track_id, track.uri]) {
    if (typeof candidate !== "string") continue;
    const value = candidate.trim();
    if (SPOTIFY_ID_RE.test(value)) return value;
    const match = value.match(TRACK_URI_RE);
    if (match) return match[1];
  }
  return null;
}

export function spotifyTrackUrl(spotifyId) {
  return spotifyId ? `https://open.spotify.com/track/${spotifyId}` : "https://open.spotify.com/";
}

export function trackTitle(track) {
  const value = track?.title ?? track?.track_name ?? track?.name;
  return typeof value === "string" && value ? value : "Unknown track";
}

export function trackArtist(track) {
  const value = track?.artist ?? track?.artists ?? track?.artist_name;
  if (Array.isArray(value)) return value.map((a) => a?.name ?? a).join(", ");
  return typeof value === "string" && value ? value : "Unknown artist";
}

export function formatMs(ms) {
  const total = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}
