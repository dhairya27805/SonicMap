import { useCallback, useEffect, useState } from "react";
import { apiErrorMessage, apiRequest } from "../api";
import { PlayAllButton, PlayButton } from "../player/PlayButton";
import { formatMs, getSpotifyId, trackArtist, trackTitle } from "../player/trackUtils";

const PLAYLIST_TYPES = [
  ["chill_relax", "Chill & Relax"],
  ["high_energy", "High Energy"],
  ["workout", "Workout"],
  ["focus", "Focus"],
  ["discover_weekly", "Discover Weekly"],
  ["late_night", "Late Night"],
  ["similar_sounds", "Similar Sounds"],
  ["explore_new", "Explore Something New"],
];

// Phase 6 SonicMap Playlists (stored in SonicMap's DB), now playable in the player.
export default function SonicPlaylists({ spotifyUser }) {
  const [playlists, setPlaylists] = useState(null); // null = not loaded yet
  const [selected, setSelected] = useState(null);
  const [playlistType, setPlaylistType] = useState("chill_relax");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");

  const loadPlaylists = useCallback(
    () =>
      apiRequest("/sonic-playlists")
        .then((data) => setPlaylists(Array.isArray(data?.playlists) ? data.playlists : []))
        .catch((err) => {
          setPlaylists([]);
          setError(apiErrorMessage(err));
        }),
    []
  );

  const openPlaylist = useCallback(async (playlistId) => {
    setError("");
    try {
      setSelected(await apiRequest(`/sonic-playlists/${playlistId}`));
    } catch (err) {
      setError(apiErrorMessage(err));
    }
  }, []);

  useEffect(() => {
    if (spotifyUser) loadPlaylists();
  }, [spotifyUser, loadPlaylists]);

  async function generate() {
    setGenerating(true);
    setError("");
    setInfo("");
    try {
      const result = await apiRequest("/sonic-playlists/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ playlist_type: playlistType, target_size: 20, save: true }),
      });
      if (result.saved && result.playlist_id) {
        const skipped = result.skipped_tracks ? ` (${result.skipped_tracks} without a Spotify ID were not saved)` : "";
        setInfo(`Saved “${result.playlist_label || "playlist"}” with ${result.saved_tracks} tracks${skipped}.`);
        await Promise.all([loadPlaylists(), openPlaylist(result.playlist_id)]);
      } else {
        setInfo(result.message || "No playlist could be generated from your library yet.");
      }
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setGenerating(false);
    }
  }

  async function removePlaylist(playlistId) {
    setError("");
    try {
      await apiRequest(`/sonic-playlists/${playlistId}`, { method: "DELETE" });
      if (selected?.id === playlistId) setSelected(null);
      await loadPlaylists();
    } catch (err) {
      setError(apiErrorMessage(err));
    }
  }

  if (!spotifyUser) {
    return (
      <div className="empty-state compact" data-testid="sonic-playlists-signed-out">
        Connect Spotify to generate and play SonicMap Playlists from your own library.
      </div>
    );
  }

  const tracks = selected?.tracks || [];
  const loading = playlists === null;
  const playlistList = playlists || [];
  const playableCount = tracks.filter((t) => getSpotifyId(t)).length;

  return (
    <div className="sonic-playlists" data-testid="sonic-playlists">
      <div className="sonic-playlists-toolbar">
        <select
          className="dark-select"
          value={playlistType}
          onChange={(e) => setPlaylistType(e.target.value)}
          aria-label="Playlist type"
          data-testid="sonic-playlist-type-select"
        >
          {PLAYLIST_TYPES.map(([value, text]) => (
            <option key={value} value={value}>
              {text}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="primary-button"
          onClick={generate}
          disabled={generating}
          data-testid="sonic-playlist-generate-button"
        >
          {generating ? "Generating…" : "Generate & save"}
        </button>
      </div>

      {error && (
        <p className="inline-error" data-testid="sonic-playlists-error">
          {error}
        </p>
      )}
      {info && (
        <p className="muted-copy" data-testid="sonic-playlists-info">
          {info}
        </p>
      )}

      <div className="sonic-playlists-grid">
        <div className="sonic-playlist-list" data-testid="sonic-playlist-list">
          {loading ? (
            <div className="empty-state compact">
              <div className="loader" /> Loading playlists…
            </div>
          ) : playlistList.length ? (
            playlistList.map((pl) => (
              <div
                key={pl.id}
                className={`sonic-playlist-item ${selected?.id === pl.id ? "selected" : ""}`}
                data-testid={`sonic-playlist-item-${pl.id}`}
              >
                <button
                  type="button"
                  className="sonic-playlist-open"
                  onClick={() => openPlaylist(pl.id)}
                  data-testid={`sonic-playlist-open-${pl.id}`}
                >
                  <strong>{pl.name}</strong>
                  <span>{pl.track_count} tracks</span>
                </button>
                <button
                  type="button"
                  className="icon-button"
                  title="Delete playlist"
                  onClick={() => removePlaylist(pl.id)}
                  data-testid={`sonic-playlist-delete-${pl.id}`}
                >
                  ×
                </button>
              </div>
            ))
          ) : (
            <div className="empty-state compact" data-testid="sonic-playlists-empty">
              No SonicMap Playlists yet. Generate one above.
            </div>
          )}
        </div>

        <div className="sonic-playlist-detail" data-testid="sonic-playlist-detail">
          {selected ? (
            <>
              <div className="playlist-result-heading">
                <div>
                  <h3 data-testid="sonic-playlist-detail-name">{selected.name}</h3>
                  <span>
                    {tracks.length} tracks · {playableCount} playable on Spotify
                  </span>
                </div>
                <PlayAllButton tracks={tracks} label={selected.name} testId="sonic-playlist-play-all" />
              </div>
              <div className="song-list">
                {tracks.map((track, index) => (
                  <div className="song-row" key={track.playlist_item_id || index} data-testid="sonic-playlist-track-row">
                    <div className="song-index">{String(index + 1).padStart(2, "0")}</div>
                    <div className="song-avatar">
                      {track.image_url ? <img src={track.image_url} alt="" className="cover-img" /> : <span>♫</span>}
                    </div>
                    <div className="song-info">
                      <strong>{trackTitle(track)}</strong>
                      <span>
                        {trackArtist(track)}
                        {track.duration_ms ? ` · ${formatMs(track.duration_ms)}` : ""}
                      </span>
                    </div>
                    <PlayButton
                      tracks={tracks}
                      index={index}
                      label={selected.name}
                      testId={`sonic-playlist-track-play-${index}`}
                    />
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state compact">Select a playlist to see and play its tracks.</div>
          )}
        </div>
      </div>
    </div>
  );
}
