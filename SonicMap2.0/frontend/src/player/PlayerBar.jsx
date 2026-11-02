import { useEffect, useState } from "react";
import { usePlayer } from "./playerContext";
import { formatMs, spotifyTrackUrl } from "./trackUtils";

const PHASE_COPY = {
  disconnected: {
    title: "Connect Spotify to listen",
    detail: "Playback runs through Spotify's official Web Playback SDK.",
    action: "connect",
  },
  checking: { title: "Connecting SonicMap Web Player…", detail: "Starting Spotify's in-browser player." },
  connecting: { title: "Connecting SonicMap Web Player…", detail: "Starting Spotify's in-browser player." },
  ready: {
    title: "Web Player ready",
    detail: "Press ▶ on any track in My Library, My Music Map, Discover or SonicMap Playlists.",
  },
  premium_required: {
    title: "Spotify Premium required",
    detail: "Spotify only allows in-browser playback for Premium accounts. You can still open tracks in Spotify.",
  },
  missing_scopes: {
    title: "Playback permission missing",
    detail: "Reconnect Spotify to grant SonicMap streaming permission.",
    action: "reconnect",
  },
  unsupported: {
    title: "Browser not supported",
    detail: "This browser can't run Spotify's Web Player (it needs DRM / EME). Try desktop Chrome, Edge or Firefox.",
  },
  auth_error: {
    title: "Spotify session expired",
    detail: "Reconnect Spotify to keep listening.",
    action: "reconnect",
  },
  error: { title: "Player unavailable", detail: "Spotify's player could not be started." },
};

const REPEAT_LABELS = ["Repeat off", "Repeat all", "Repeat one"];

export default function PlayerBar({ onConnect }) {
  const p = usePlayer();
  const { state } = p;
  const [now, setNow] = useState(() => Date.now());
  const [scrub, setScrub] = useState(null);
  const [mutedFrom, setMutedFrom] = useState(null);

  const playing = Boolean(state && !state.paused);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, [playing]);

  const duration = state?.duration || 0;
  const elapsed = state ? state.position + (playing ? Math.max(0, now - state.updatedAt) : 0) : 0;
  const position = scrub ?? Math.min(duration, elapsed);
  const controlsEnabled = p.isReady && Boolean(state);
  const copy = PHASE_COPY[p.phase] || PHASE_COPY.error;
  const track = state?.track;
  const openId = track?.id || p.lastRequested?.spotifyId;

  function commitScrub() {
    if (scrub === null) return;
    p.seek(scrub);
    setScrub(null);
  }

  function toggleMute() {
    if (p.volume > 0) {
      setMutedFrom(p.volume);
      p.setVolume(0);
    } else {
      p.setVolume(mutedFrom || 0.6);
      setMutedFrom(null);
    }
  }

  return (
    <div className="player-dock" data-testid="player-bar" data-phase={p.phase}>
      {p.message && (
        <div className={`player-message tone-${p.message.tone}`} role="status" data-testid="player-message">
          <span data-testid="player-message-text">{p.message.text}</span>
          <button type="button" onClick={p.dismissMessage} aria-label="Dismiss" data-testid="player-message-dismiss">
            ×
          </button>
        </div>
      )}

      <div className="player-bar">
        <div className="player-track">
          <div className={`player-art ${playing ? "is-playing" : ""}`} data-testid="player-artwork">
            {track?.image ? <img src={track.image} alt="" /> : <span>♫</span>}
          </div>
          <div className="player-track-text">
            {track ? (
              <>
                <strong data-testid="player-track-title">{track.name}</strong>
                <span data-testid="player-track-artist">{track.artists}</span>
              </>
            ) : (
              <>
                <strong data-testid="player-status-title">{copy.title}</strong>
                <span data-testid="player-status-detail">{copy.detail}</span>
              </>
            )}
          </div>
          {copy.action && !track && (
            <button type="button" className="player-pill" onClick={onConnect} data-testid="player-connect-button">
              {copy.action === "connect" ? "Connect Spotify" : "Reconnect"}
            </button>
          )}
        </div>

        <div className="player-center">
          <div className="player-controls">
            <button
              type="button"
              className={`player-btn small ${state?.shuffle ? "active" : ""}`}
              onClick={p.toggleShuffle}
              disabled={!controlsEnabled}
              title={state?.shuffle ? "Shuffle on" : "Shuffle off"}
              aria-pressed={Boolean(state?.shuffle)}
              data-testid="player-shuffle-button"
            >
              ⤨
            </button>
            <button
              type="button"
              className="player-btn"
              onClick={p.previous}
              disabled={!controlsEnabled || !state?.canPrevious}
              title="Previous"
              data-testid="player-previous-button"
            >
              ⏮
            </button>
            <button
              type="button"
              className="player-btn main"
              onClick={p.togglePlay}
              disabled={!controlsEnabled}
              title={playing ? "Pause" : "Play"}
              aria-label={playing ? "Pause" : "Play"}
              data-testid="player-play-pause-button"
            >
              {playing ? "❚❚" : "▶"}
            </button>
            <button
              type="button"
              className="player-btn"
              onClick={p.next}
              disabled={!controlsEnabled || !state?.canNext}
              title="Next"
              data-testid="player-next-button"
            >
              ⏭
            </button>
            <button
              type="button"
              className={`player-btn small ${state?.repeatMode ? "active" : ""}`}
              onClick={p.cycleRepeat}
              disabled={!controlsEnabled}
              title={REPEAT_LABELS[state?.repeatMode || 0]}
              data-testid="player-repeat-button"
            >
              {state?.repeatMode === 2 ? "↻1" : "↻"}
            </button>
          </div>

          <div className="player-progress">
            <span data-testid="player-elapsed">{formatMs(position)}</span>
            <input
              type="range"
              min={0}
              max={Math.max(duration, 1)}
              step={1000}
              value={Math.min(position, Math.max(duration, 1))}
              disabled={!controlsEnabled || !state?.canSeek}
              onChange={(e) => setScrub(Number(e.target.value))}
              onPointerUp={commitScrub}
              onKeyUp={commitScrub}
              onBlur={commitScrub}
              aria-label="Seek"
              style={{ "--fill": `${duration ? (position / duration) * 100 : 0}%` }}
              data-testid="player-seek-slider"
            />
            <span data-testid="player-duration">{formatMs(duration)}</span>
          </div>
        </div>

        <div className="player-right">
          <button
            type="button"
            className="player-btn small"
            onClick={toggleMute}
            disabled={!p.isReady}
            title={p.volume > 0 ? "Mute" : "Unmute"}
            data-testid="player-mute-button"
          >
            {p.volume === 0 ? "🔇" : p.volume < 0.5 ? "🔉" : "🔊"}
          </button>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(p.volume * 100)}
            onChange={(e) => p.setVolume(Number(e.target.value) / 100)}
            disabled={!p.isReady}
            aria-label="Volume"
            className="player-volume"
            style={{ "--fill": `${Math.round(p.volume * 100)}%` }}
            data-testid="player-volume-slider"
          />
          <a
            className="player-open"
            href={spotifyTrackUrl(openId)}
            target="_blank"
            rel="noreferrer"
            title="Open in Spotify"
            data-testid="player-open-spotify-link"
          >
            Open in Spotify ↗
          </a>
        </div>
      </div>
    </div>
  );
}
