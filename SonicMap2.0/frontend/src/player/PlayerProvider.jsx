import { useCallback, useEffect, useMemo, useState } from "react";
import { apiErrorMessage, apiRequest } from "../api";
import { PlayerContext } from "./playerContext";
import { getSpotifyId, trackArtist, trackTitle } from "./trackUtils";

const SDK_SRC = "https://sdk.scdn.co/spotify-player.js";
const VOLUME_KEY = "sonicmap.player.volume";
const SDK_EVENTS = [
  "ready",
  "not_ready",
  "player_state_changed",
  "initialization_error",
  "authentication_error",
  "account_error",
  "playback_error",
  "autoplay_failed",
];

// One SDK script and one Spotify.Player per browser tab. Kept on window so it
// survives React StrictMode double-mounting and Vite hot reloads — this is
// what guarantees there is never a duplicate "SonicMap Web Player" device.
const registry = (window.__SONICMAP_PLAYER__ ??= {
  sdkPromise: null,
  creating: null,
  player: null,
  deviceId: null,
  handlers: null,
});

function emit(event, payload) {
  registry.handlers?.[event]?.(payload);
}

function loadSpotifySdk() {
  if (window.Spotify?.Player) return Promise.resolve(window.Spotify);
  if (registry.sdkPromise) return registry.sdkPromise;

  registry.sdkPromise = new Promise((resolve, reject) => {
    window.onSpotifyWebPlaybackSDKReady = () => resolve(window.Spotify);
    if (!document.querySelector(`script[src="${SDK_SRC}"]`)) {
      const script = document.createElement("script");
      script.src = SDK_SRC;
      script.async = true;
      script.onerror = () => {
        registry.sdkPromise = null;
        script.remove();
        reject(new Error("Could not load Spotify's Web Playback SDK (network or content blocker)."));
      };
      document.body.appendChild(script);
    }
  });
  return registry.sdkPromise;
}

function getOrCreatePlayer(volume) {
  if (registry.player) return Promise.resolve(registry.player);
  if (!registry.creating) {
    registry.creating = loadSpotifySdk()
      .then(async (Spotify) => {
        const player = new Spotify.Player({
          name: "SonicMap Web Player",
          volume,
          // The SDK calls this on connect and whenever the token expires; the
          // backend refreshes server-side and returns only a short-lived token.
          getOAuthToken: (callback) => {
            apiRequest("/playback/token")
              .then((data) => callback(data.access_token))
              .catch((err) => emit("token_error", err));
          },
        });
        SDK_EVENTS.forEach((event) => player.addListener(event, (payload) => emit(event, payload)));
        registry.player = player;
        const connected = await player.connect();
        if (!connected) emit("initialization_error", { message: "Spotify player could not connect." });
        return player;
      })
      .catch((err) => {
        registry.creating = null;
        throw err;
      });
  }
  return registry.creating;
}

function destroyPlayer() {
  if (registry.player) registry.player.disconnect();
  registry.player = null;
  registry.creating = null;
  registry.deviceId = null;
}

function readStoredVolume() {
  const stored = Number(window.localStorage.getItem(VOLUME_KEY));
  return Number.isFinite(stored) && stored >= 0 && stored <= 1 && window.localStorage.getItem(VOLUME_KEY) !== null
    ? stored
    : 0.6;
}

function normalizeState(state) {
  if (!state) return null;
  const track = state.track_window?.current_track;
  const disallows = state.disallows || {};
  return {
    paused: state.paused,
    position: state.position || 0,
    duration: state.duration || track?.duration_ms || 0,
    shuffle: Boolean(state.shuffle),
    repeatMode: state.repeat_mode || 0, // 0 off, 1 context, 2 track
    canPrevious: !disallows.skipping_prev,
    canNext: !disallows.skipping_next,
    canSeek: !disallows.seeking,
    track: track
      ? {
          // linked_from is set when Spotify relinked the requested track for this market
          requestedId: track.linked_from?.id || track.id,
          id: track.id,
          name: track.name,
          artists: (track.artists || []).map((a) => a.name).join(", "),
          album: track.album?.name || "",
          image: track.album?.images?.[0]?.url || "",
        }
      : null,
    updatedAt: Date.now(),
  };
}

const NOT_READY_COPY = {
  disconnected: "Connect Spotify to play music inside SonicMap.",
  checking: "The SonicMap Web Player is still connecting — try again in a moment.",
  connecting: "The SonicMap Web Player is still connecting — try again in a moment.",
  premium_required:
    "Spotify Premium is required for in-browser playback. Use “Open in Spotify” to listen in the Spotify app.",
  missing_scopes: "Reconnect Spotify to grant SonicMap playback permission.",
  unsupported: "This browser can't run Spotify's Web Player.",
  auth_error: "Your Spotify session expired. Reconnect Spotify to keep listening.",
  error: "The Spotify player is unavailable right now.",
};

export default function PlayerProvider({ children }) {
  // undefined = auth not checked yet, null = signed out, object = Spotify user
  const [user, setUser] = useState(undefined);
  const [phase, setPhase] = useState("disconnected");
  const [deviceId, setDeviceId] = useState(registry.deviceId);
  const [state, setState] = useState(null);
  const [volume, setVolumeState] = useState(readStoredVolume);
  const [message, setMessage] = useState(null);
  const [lastRequested, setLastRequested] = useState(null);
  const [busy, setBusy] = useState(false);

  const userId = user === undefined ? undefined : user?.id ?? null;

  // Signing out tears down the single SDK player immediately.
  const updateUser = useCallback((nextUser) => {
    setUser(nextUser);
    if (nextUser === null) {
      destroyPlayer();
      setDeviceId(null);
      setState(null);
      setPhase("disconnected");
    }
  }, []);

  // SDK events -> React state. Handlers only use stable setters.
  useEffect(() => {
    const handlers = {
      ready: ({ device_id }) => {
        registry.deviceId = device_id;
        setDeviceId(device_id);
        setPhase("ready");
      },
      not_ready: () => {
        registry.deviceId = null;
        setDeviceId(null);
        setPhase((p) => (p === "ready" ? "connecting" : p));
      },
      player_state_changed: (s) => setState(normalizeState(s)),
      initialization_error: ({ message: text }) => {
        setPhase("unsupported");
        setMessage({ tone: "error", text: `Spotify player failed to start: ${text}` });
      },
      authentication_error: () => setPhase("auth_error"),
      account_error: () => {
        // Spotify's SDK raises this for non-Premium accounts.
        setPhase("premium_required");
        destroyPlayer();
      },
      playback_error: ({ message: text }) =>
        setMessage({
          tone: "warn",
          text: `Spotify couldn't play this track (${text}). It may be unavailable in your region.`,
        }),
      autoplay_failed: () =>
        setMessage({ tone: "info", text: "Your browser blocked autoplay — press play to start listening." }),
      token_error: (err) => {
        if (err?.detail?.code === "premium_required") {
          setPhase("premium_required");
          destroyPlayer();
        } else {
          setPhase("auth_error");
        }
      },
    };
    registry.handlers = handlers;
    return () => {
      if (registry.handlers === handlers) registry.handlers = null;
    };
  }, []);

  // Connect / disconnect the single SDK player as the Spotify session changes.
  useEffect(() => {
    if (!userId) return undefined;
    let cancelled = false;

    (async () => {
      setPhase(registry.deviceId ? "ready" : "checking");
      try {
        const status = await apiRequest("/playback/status");
        if (cancelled) return;
        if (!status.authenticated) {
          setPhase("disconnected");
          return;
        }
        if (status.reason === "missing_scopes") {
          setPhase("missing_scopes");
          return;
        }
        if (status.reason === "premium_required") {
          setPhase("premium_required");
          return;
        }
        if (!registry.deviceId) setPhase("connecting");
        await getOrCreatePlayer(readStoredVolume());
        if (!cancelled && registry.deviceId) {
          setDeviceId(registry.deviceId);
          setPhase("ready");
        }
      } catch (err) {
        if (cancelled) return;
        setPhase("error");
        setMessage({ tone: "error", text: apiErrorMessage(err) });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [userId]);

  const handlePlaybackError = useCallback((err) => {
    const code = err?.detail?.code;
    if (code === "premium_required") {
      setPhase("premium_required");
      destroyPlayer();
    } else if (code === "spotify_auth" || code === "not_connected") {
      setPhase("auth_error");
    }
    setMessage({ tone: code === "rate_limited" ? "warn" : "error", text: apiErrorMessage(err) });
  }, []);

  const playTracks = useCallback(
    async (tracks, startIndex = 0, label = "") => {
      const list = (tracks || []).map((track) => ({ track, id: getSpotifyId(track) }));
      const chosen = list[startIndex] || list[0];
      const playable = list.filter((entry) => entry.id);

      if (chosen && !chosen.id) {
        setMessage({
          tone: "warn",
          text: `“${trackTitle(chosen.track)}” has no Spotify track ID, so it can't be played on Spotify.`,
        });
        return false;
      }
      if (!playable.length) {
        setMessage({ tone: "warn", text: "None of these tracks are available on Spotify." });
        return false;
      }

      const offset = Math.max(0, playable.indexOf(chosen));
      const first = playable[offset];
      setLastRequested({
        spotifyId: first.id,
        title: trackTitle(first.track),
        artist: trackArtist(first.track),
        label,
      });

      if (phase !== "ready" || !deviceId) {
        setMessage({ tone: phase === "premium_required" ? "warn" : "info", text: NOT_READY_COPY[phase] });
        return false;
      }

      // Must run inside the click handler for mobile / Safari autoplay rules.
      registry.player?.activateElement?.();
      setBusy(true);
      try {
        const result = await apiRequest("/playback/play", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            device_id: deviceId,
            uris: playable.map((entry) => entry.id),
            offset_index: offset,
          }),
        });
        const notes = [];
        if (result.unavailable?.length) {
          notes.push(`${result.unavailable.length} track(s) unavailable on Spotify in your market were skipped`);
        }
        const missing = list.length - playable.length;
        if (missing > 0) notes.push(`${missing} track(s) without a Spotify ID were left out`);
        setMessage(
          notes.length
            ? { tone: "warn", text: `Playing ${label ? `“${label}”` : "your selection"} — ${notes.join("; ")}.` }
            : null
        );
        return true;
      } catch (err) {
        handlePlaybackError(err);
        return false;
      } finally {
        setBusy(false);
      }
    },
    [phase, deviceId, handlePlaybackError]
  );

  const togglePlay = useCallback(() => registry.player?.togglePlay(), []);
  const next = useCallback(() => registry.player?.nextTrack(), []);
  const previous = useCallback(() => registry.player?.previousTrack(), []);

  const seek = useCallback((positionMs) => {
    if (!registry.player) return;
    registry.player.seek(positionMs);
    setState((s) => (s ? { ...s, position: positionMs, updatedAt: Date.now() } : s));
  }, []);

  const setVolume = useCallback((value) => {
    const v = Math.min(1, Math.max(0, value));
    setVolumeState(v);
    window.localStorage.setItem(VOLUME_KEY, String(v));
    registry.player?.setVolume(v);
  }, []);

  const toggleShuffle = useCallback(async () => {
    if (!deviceId || !state) return;
    try {
      await apiRequest(
        `/playback/shuffle?state=${!state.shuffle}&device_id=${encodeURIComponent(deviceId)}`,
        { method: "PUT" }
      );
    } catch (err) {
      handlePlaybackError(err);
    }
  }, [deviceId, state, handlePlaybackError]);

  const cycleRepeat = useCallback(async () => {
    if (!deviceId || !state) return;
    const nextMode = ["context", "track", "off"][state.repeatMode] || "off";
    try {
      await apiRequest(`/playback/repeat?state=${nextMode}&device_id=${encodeURIComponent(deviceId)}`, {
        method: "PUT",
      });
    } catch (err) {
      handlePlaybackError(err);
    }
  }, [deviceId, state, handlePlaybackError]);

  const value = useMemo(
    () => ({
      phase,
      deviceId,
      state,
      volume,
      busy,
      message,
      lastRequested,
      isReady: phase === "ready" && Boolean(deviceId),
      currentSpotifyId: state?.track?.requestedId || null,
      setUser: updateUser,
      playTracks,
      togglePlay,
      next,
      previous,
      seek,
      setVolume,
      toggleShuffle,
      cycleRepeat,
      dismissMessage: () => setMessage(null),
    }),
    [
      phase,
      deviceId,
      state,
      volume,
      busy,
      message,
      lastRequested,
      updateUser,
      playTracks,
      togglePlay,
      next,
      previous,
      seek,
      setVolume,
      toggleShuffle,
      cycleRepeat,
    ]
  );

  return <PlayerContext.Provider value={value}>{children}</PlayerContext.Provider>;
}
