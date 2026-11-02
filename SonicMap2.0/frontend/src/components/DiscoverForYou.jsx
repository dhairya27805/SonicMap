import { useEffect, useState } from "react";
import { apiErrorMessage, apiRequest } from "../api";
import { PlayAllButton, PlayButton } from "../player/PlayButton";
import { getSpotifyId, trackArtist, trackTitle } from "../player/trackUtils";

const MODES = [
  ["similar_to_library", "Similar to my music"],
  ["different", "Something different"],
];

const SOURCE_LABELS = {
  reference_catalog: "Spotify catalogue",
  itunes: "iTunes metadata",
  musicbrainz: "MusicBrainz metadata",
};

// Phase 5 discovery results (outside the user's library), now playable.
export default function DiscoverForYou({ spotifyUser }) {
  const [mode, setMode] = useState("similar_to_library");
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [result, setResult] = useState({ key: null, tracks: [], error: "" });

  const requestKey = spotifyUser ? `${mode}|${submittedQuery}` : null;
  const loading = Boolean(requestKey) && result.key !== requestKey;
  const tracks = loading ? [] : result.tracks;
  const error = loading ? "" : result.error;

  useEffect(() => {
    if (!requestKey) return undefined;
    let cancelled = false;
    const params = new URLSearchParams({ mode, limit: "15" });
    if (submittedQuery) params.set("q", submittedQuery);
    apiRequest(`/discovery/recommendations?${params.toString()}`)
      .then((data) => {
        if (cancelled) return;
        const list = Array.isArray(data?.recommendations) ? data.recommendations : [];
        setResult({ key: requestKey, tracks: list, error: "" });
      })
      .catch((err) => {
        if (!cancelled) setResult({ key: requestKey, tracks: [], error: apiErrorMessage(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [requestKey, mode, submittedQuery]);

  if (!spotifyUser) {
    return (
      <div className="empty-state compact" data-testid="discover-for-you-signed-out">
        Connect Spotify to get discovery picks from outside your library.
      </div>
    );
  }

  const label = MODES.find(([m]) => m === mode)?.[1] || "Discover";

  return (
    <div className="discover-for-you" data-testid="discover-for-you">
      <div className="discover-for-you-head">
        <div>
          <div className="section-kicker">DISCOVER FOR YOU</div>
          <h3>Outside your library</h3>
        </div>
        <div className="discover-for-you-controls">
          {MODES.map(([value, text]) => (
            <button
              key={value}
              type="button"
              className={`chip-button ${mode === value ? "selected" : ""}`}
              onClick={() => setMode(value)}
              data-testid={`discover-mode-${value}`}
            >
              {text}
            </button>
          ))}
          <form
            className="search-form compact"
            onSubmit={(e) => {
              e.preventDefault();
              setSubmittedQuery(query.trim());
            }}
          >
            <span>⌕</span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Optional keyword…"
              data-testid="discover-for-you-search-input"
            />
            <button type="submit" data-testid="discover-for-you-search-button">
              Go
            </button>
          </form>
          <PlayAllButton tracks={tracks} label={`Discover · ${label}`} testId="discover-for-you-play-all" />
        </div>
      </div>

      {loading ? (
        <div className="empty-state compact">
          <div className="loader" /> Finding new music…
        </div>
      ) : error ? (
        <div className="empty-state compact" data-testid="discover-for-you-error">
          {error}
        </div>
      ) : tracks.length ? (
        <div className="song-list">
          {tracks.map((track, index) => (
            <div className="song-row" key={`${track.spotify_id || track.title}-${index}`} data-testid="discover-for-you-row">
              <div className="song-index">{String(index + 1).padStart(2, "0")}</div>
              <div className="song-avatar">
                {track.image_url ? <img src={track.image_url} alt="" className="cover-img" /> : <span>♫</span>}
              </div>
              <div className="song-info">
                <strong>{trackTitle(track)}</strong>
                <span>
                  {trackArtist(track)}
                  {track.album ? ` · ${track.album}` : ""}
                </span>
              </div>
              <span className={`cluster-tag ${getSpotifyId(track) ? "" : "muted-tag"}`}>
                {getSpotifyId(track) ? SOURCE_LABELS[track.source] || "Spotify" : "Not on Spotify"}
              </span>
              <PlayButton
                tracks={tracks}
                index={index}
                label={`Discover · ${label}`}
                testId={`discover-for-you-play-button-${index}`}
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state compact" data-testid="discover-for-you-empty">
          No discovery picks yet. Sync your library and acquire audio features first.
        </div>
      )}
    </div>
  );
}
