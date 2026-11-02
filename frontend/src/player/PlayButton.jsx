import { usePlayer } from "./playerContext";
import { getSpotifyId } from "./trackUtils";

// Play / pause a single track; the surrounding list becomes the queue.
export function PlayButton({ tracks, index, label, testId }) {
  const p = usePlayer();
  const spotifyId = getSpotifyId(tracks?.[index]);

  if (!spotifyId) {
    return (
      <button
        type="button"
        className="icon-button play-button unavailable"
        disabled
        title="Not on Spotify — this track has no Spotify track ID, so it can't be played"
        aria-label="Not available on Spotify"
        data-testid={testId}
        data-playable="false"
      >
        ⊘
      </button>
    );
  }

  const isCurrent = p.currentSpotifyId === spotifyId;
  const playing = isCurrent && p.state && !p.state.paused;

  return (
    <button
      type="button"
      className={`icon-button play-button ${isCurrent ? "current" : ""}`}
      onClick={() => (isCurrent ? p.togglePlay() : p.playTracks(tracks, index, label))}
      disabled={p.busy && !isCurrent}
      title={playing ? "Pause" : "Play on Spotify"}
      aria-label={playing ? "Pause" : "Play on Spotify"}
      data-testid={testId}
      data-playable="true"
    >
      {playing ? "❚❚" : "▶"}
    </button>
  );
}

export function PlayAllButton({ tracks, label, testId, children = "Play all" }) {
  const p = usePlayer();
  const playableCount = (tracks || []).filter((t) => getSpotifyId(t)).length;

  return (
    <button
      type="button"
      className="primary-button play-all-button"
      onClick={() => p.playTracks(tracks, 0, label)}
      disabled={!playableCount || p.busy}
      title={playableCount ? `Play ${playableCount} track(s) on Spotify` : "No tracks with a Spotify ID"}
      data-testid={testId}
    >
      ▶ {children}
    </button>
  );
}
