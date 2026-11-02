
import { useCallback, useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import "./App.css";
import "./player/player.css";
import { API, apiErrorMessage, apiRequest } from "./api";
import PlayerProvider from "./player/PlayerProvider";
import PlayerBar from "./player/PlayerBar";
import { PlayAllButton, PlayButton } from "./player/PlayButton";
import { usePlayer } from "./player/playerContext";
import DiscoverForYou from "./components/DiscoverForYou";
import SonicPlaylists from "./components/SonicPlaylists";

function asArray(data, key) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.[key])) return data[key];
  return [];
}

function displayText(value, fallback = "—") {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "object") {
    return value.name ?? value.label ?? value.count ?? fallback;
  }
  return String(value);
}

function getClusterCount(overview) {
  if (!overview) return 8;

  const explicit =
    overview.n_clusters ??
    overview.num_clusters ??
    overview.cluster_count;

  if (Number.isFinite(Number(explicit))) return Number(explicit);

  const clusters = overview.clusters;

  if (Array.isArray(clusters)) return clusters.length;
  if (typeof clusters === "number") return clusters;
  if (clusters && typeof clusters === "object") {
    if (Number.isFinite(Number(clusters.count))) {
      return Number(clusters.count);
    }
    return 1;
  }

  return 8;
}

function getSongCount(overview) {
  if (!overview) return "—";

  const value =
    overview.total_songs ??
    overview.track_count ??
    overview.total_tracks ??
    overview.n_songs ??
    overview.songs;

  if (typeof value === "number" || typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) return value.length;
  if (value && typeof value === "object") {
    return value.count ?? value.total ?? "—";
  }

  return "—";
}

function getFeatureCount(overview) {
  if (!overview) return "—";

  const value =
    overview.n_features ??
    overview.feature_count ??
    overview.features;

  if (Array.isArray(value)) return value.length;
  if (typeof value === "number" || typeof value === "string") {
    return value;
  }

  return "—";
}

function getSilhouette(overview) {
  const value =
    overview?.silhouette ??
    overview?.silhouette_score;

  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(3);
  }

  return "—";
}

function SongRow({ song, onRecommend, index, queue, queueLabel, testIdPrefix = "song" }) {
  const title = displayText(
    song.title ?? song.track_name ?? song.name,
    "Unknown track"
  );

  const artist = displayText(
    song.artist ?? song.artists ?? song.artist_name,
    "Unknown artist"
  );

  const cluster = displayText(song.cluster, "—");

  return (
    <div className="song-row">
      <div className="song-index">
        {String(index + 1).padStart(2, "0")}
      </div>

      <div className="song-avatar">
        {song.image_url ? (
          <img src={song.image_url} alt="" style={{ width: "100%", height: "100%", borderRadius: "8px", objectFit: "cover" }} />
        ) : (
          <span>♫</span>
        )}
      </div>

      <div className="song-info">
        <strong>{title}</strong>
        <span>{artist}</span>
      </div>

      <span className="cluster-tag">
        Cluster {cluster}
      </span>

      {Number.isFinite(Number(song.similarity)) && (
        <span
          className="cluster-tag"
          title="Cosine similarity based on standardized audio features"
          style={{ borderColor: "rgba(167, 139, 250, 0.45)", color: "#c4b5fd" }}
        >
          {Number(song.similarity).toFixed(1)}% match
        </span>
      )}

      {queue && (
        <PlayButton
          tracks={queue}
          index={index}
          label={queueLabel}
          testId={`${testIdPrefix}-play-button-${index}`}
        />
      )}

      <button
        className="icon-button"
        title="Find similar songs"
        onClick={() => onRecommend(song)}
        data-testid={`${testIdPrefix}-similar-button-${index}`}
      >
        ↗
      </button>
    </div>
  );
}

function StatCard({ label, value, detail, icon }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <span className="stat-icon">{icon}</span>
      </div>

      <div className="stat-value">
        {displayText(value)}
      </div>

      <div className="stat-detail">{detail}</div>
    </div>
  );
}

// The PlayerProvider sits above the whole page and never unmounts, so the
// Spotify player (one SDK device) persists across every section/navigation.
export default function App() {
  return (
    <PlayerProvider>
      <SonicMapApp />
    </PlayerProvider>
  );
}

function SonicMapApp() {
  const playback = usePlayer();
  const { setUser: setPlaybackUser } = playback;
  const plotRef = useRef(null);
  const myMapPlotRef = useRef(null);

  const [overview, setOverview] = useState(null);
  const [points, setPoints] = useState([]);
  const [songs, setSongs] = useState([]);
  const [query, setQuery] = useState("");
  const [cluster, setCluster] = useState(-1);
  const [mood, setMood] = useState("discovery");
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [selectedSong, setSelectedSong] = useState(null);
  const [playlist, setPlaylist] = useState([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  // --- Spotify auth + playback state ---
  const [authChecked, setAuthChecked] = useState(false);
  const [spotifyUser, setSpotifyUser] = useState(null);
  const [connecting, setConnecting] = useState(false);
  const [creatingPlaylist, setCreatingPlaylist] = useState(false);
  const [spotifyPlaylistUrl, setSpotifyPlaylistUrl] = useState("");

  // --- My Library State (Phase 2) ---
  const [libraryTracks, setLibraryTracks] = useState([]);
  const [libraryTotal, setLibraryTotal] = useState(0);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryQuery, setLibraryQuery] = useState("");
  const [librarySource, setLibrarySource] = useState("");
  const [syncStatus, setSyncStatus] = useState(null);
  const [syncing, setSyncing] = useState(false);

  // --- My Music Map State (Phase 4) ---
  const [personalStatus, setPersonalStatus] = useState(null);
  const [personalClusters, setPersonalClusters] = useState([]);
  const [personalMapPoints, setPersonalMapPoints] = useState([]);
  const [personalCluster, setPersonalCluster] = useState(-1);
  const [personalQuery, setPersonalQuery] = useState("");
  const [personalSelectedTrack, setPersonalSelectedTrack] = useState(null);
  const [personalRecommendations, setPersonalRecommendations] = useState([]);
  const [personalRecLoading, setPersonalRecLoading] = useState(false);
  const [personalLoading, setPersonalLoading] = useState(false);
  const [personalTraining, setPersonalTraining] = useState(false);
  const [acquiringFeatures, setAcquiringFeatures] = useState(false);

  const totalSongs = getSongCount(overview);
  const totalClusters = getClusterCount(overview);
  const featureCount = getFeatureCount(overview);
  const silhouette = getSilhouette(overview);

  const loadLibrary = useCallback(async (search = "", sourceFilter = "") => {
    if (!spotifyUser) return;
    setLibraryLoading(true);
    try {
      const params = new URLSearchParams({ limit: "50", offset: "0" });
      if (search) params.set("q", search);
      if (sourceFilter) params.set("source", sourceFilter);
      const data = await apiRequest(`/library/tracks?${params.toString()}`);
      setLibraryTracks(data.tracks || []);
      setLibraryTotal(data.total || 0);
    } catch {
      // Non-fatal if library is empty or not yet synced
    } finally {
      setLibraryLoading(false);
    }
  }, [spotifyUser]);

  const loadPersonalStatus = useCallback(async () => {
    if (!spotifyUser) return;
    try {
      const data = await apiRequest("/personal/status");
      setPersonalStatus(data);
      if (Array.isArray(data.clusters)) {
        setPersonalClusters(data.clusters);
      }
    } catch {
      // Non-fatal if unauthenticated
    }
  }, [spotifyUser]);

  const loadPersonalMap = useCallback(async (selectedCluster = -1) => {
    if (!spotifyUser) return;
    setPersonalLoading(true);
    try {
      const data = await apiRequest(`/personal/map?cluster=${selectedCluster}&limit=2000`);
      setPersonalMapPoints(asArray(data, "points"));
    } catch {
      setPersonalMapPoints([]);
    } finally {
      setPersonalLoading(false);
    }
  }, [spotifyUser]);

  async function trainPersonalModel() {
    if (!spotifyUser || personalTraining) return;
    setPersonalTraining(true);
    setError("");
    setNotice("Training your personalized KMeans model & 2D PCA projection...");
    try {
      const result = await apiRequest("/personal/train", { method: "POST" });
      if (result.status === "trained") {
        setNotice(`Personalized model trained successfully! (${result.cluster_count} clusters created from ${result.eligible_tracks} tracks).`);
        await Promise.all([loadPersonalStatus(), loadPersonalMap(personalCluster)]);
      } else if (result.status === "insufficient_tracks") {
        setError(result.message || "Insufficient eligible tracks to train.");
        await loadPersonalStatus();
      }
    } catch (err) {
      setError(`Failed to train personalized model: ${err.message}`);
    } finally {
      setPersonalTraining(false);
    }
  }

  function showLibrarySimilar(song) {
    const point = personalMapPoints.find((pt) => pt.id === song.id);
    if (!point) {
      setError(
        `“${song.title}” isn't on My Music Map yet. Similar tracks need all 9 audio features and a trained personal model (clear any cluster filter if one is active).`
      );
      return;
    }
    showPersonalRecommendations(point);
    requestAnimationFrame(() => {
      document.getElementById("my-music-map")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  async function showPersonalRecommendations(track) {
    if (!track?.id) return;
    setPersonalSelectedTrack(track);
    setPersonalRecLoading(true);
    setError("");
    try {
      const data = await apiRequest(`/personal/recommendations/${track.id}?limit=8`);
      setPersonalRecommendations(asArray(data, "recommendations"));
    } catch (err) {
      setError(`Could not retrieve similar tracks from your library: ${apiErrorMessage(err)}`);
      setPersonalRecommendations([]);
    } finally {
      setPersonalRecLoading(false);
    }
  }

  async function triggerFeatureAcquisition() {
    if (!spotifyUser || acquiringFeatures) return;
    setAcquiringFeatures(true);
    setError("");
    setNotice("Acquiring 9 audio features for your imported library...");
    try {
      const res = await apiRequest("/library/features/acquire", { method: "POST" });
      setNotice(res.message || "Feature acquisition started in the background.");
      setTimeout(async () => {
        await Promise.all([loadPersonalStatus(), loadPersonalMap(personalCluster)]);
        setAcquiringFeatures(false);
      }, 2500);
    } catch (err) {
      setError(`Feature acquisition failed: ${err.message}`);
      setAcquiringFeatures(false);
    }
  }

  const checkSyncStatus = useCallback(async () => {
    if (!spotifyUser) return;
    try {
      const data = await apiRequest("/library/status");
      setSyncStatus(data);
      if (data.status === "running") {
        setSyncing(true);
      } else {
        setSyncing(false);
      }
    } catch {
      // Ignore if unauthenticated
    }
  }, [spotifyUser]);

  async function triggerLibrarySync() {
    if (!spotifyUser || syncing) return;
    setSyncing(true);
    setError("");
    try {
      await apiRequest("/library/sync", { method: "POST" });
      setNotice("Spotify library synchronization started in the background.");
      // Start polling
      checkSyncStatus();
    } catch (err) {
      setError(`Failed to start library sync: ${err.message}`);
      setSyncing(false);
    }
  }

  // Poll sync status only while a library sync is actually running.
  // (Previously `!syncing && !spotifyUser` kept polling forever once signed in and
  // reloaded the library + My Music Map every 2.5 s, making the page jump.)
  useEffect(() => {
    if (!syncing || !spotifyUser) return;
    const interval = setInterval(async () => {
      try {
        const data = await apiRequest("/library/status");
        setSyncStatus(data);
        if (data.status !== "running") {
          setSyncing(false);
          loadLibrary(libraryQuery, librarySource);
          loadPersonalStatus();
          loadPersonalMap(personalCluster);
        }
      } catch {
        // Stop polling on error
        setSyncing(false);
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [syncing, spotifyUser, libraryQuery, librarySource, loadLibrary, loadPersonalStatus, loadPersonalMap, personalCluster]);

  useEffect(() => {
    if (spotifyUser) {
      checkSyncStatus();
      loadLibrary(libraryQuery, librarySource);
      loadPersonalStatus();
      loadPersonalMap(personalCluster);
    }
  }, [spotifyUser, librarySource, checkSyncStatus, loadLibrary, loadPersonalStatus, loadPersonalMap, personalCluster]);

  const loadOverview = useCallback(async () => {
    const data = await apiRequest("/overview");
    setOverview(data);
  }, []);

  const loadMap = useCallback(async (selectedCluster = -1) => {
    const data = await apiRequest(
      `/map?limit=4000&cluster=${selectedCluster}`
    );

    setPoints(asArray(data, "points"));
  }, []);

  const loadSongs = useCallback(async (search = "") => {
    const data = await apiRequest(
      `/songs?q=${encodeURIComponent(search)}&limit=30`
    );

    setSongs(asArray(data, "songs"));
  }, []);

  const loadApp = useCallback(async () => {
    setBusy(true);
    setError("");

    try {
      await Promise.all([
        loadOverview(),
        loadMap(cluster),
        loadSongs(query),
      ]);
    } catch (err) {
      setError(
        `Could not load SonicMap data. Check that the backend is running. ${err.message}`
      );
    } finally {
      setBusy(false);
    }
  }, [cluster, query, loadMap, loadOverview, loadSongs]);

  useEffect(() => {
    loadApp();
  }, [loadApp]);

  // Check Spotify auth status on load, and surface OAuth redirect results.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const spotifyError = params.get("spotify_error");
    const connected = params.get("connected");

    if (spotifyError) {
      setError(`Spotify connection failed: ${decodeURIComponent(spotifyError)}`);
    } else if (connected) {
      setNotice("Spotify account connected.");
    }

    if (spotifyError || connected) {
      window.history.replaceState({}, "", window.location.pathname);
    }

    (async () => {
      try {
        const data = await apiRequest("/auth/me");
        setSpotifyUser(data.authenticated ? data.user : null);
      } catch {
        setSpotifyUser(null);
      } finally {
        setAuthChecked(true);
      }
    })();
  }, []);

  // Hand the Spotify session to the persistent player (Phase 7).
  useEffect(() => {
    setPlaybackUser(authChecked ? spotifyUser : undefined);
  }, [authChecked, spotifyUser, setPlaybackUser]);

  function connectSpotify() {
    setConnecting(true);
    window.location.href = `${API}/auth/login`;
  }

  async function disconnectSpotify() {
    try {
      await apiRequest("/auth/logout", { method: "POST" });
    } catch {
      // Ignore; we clear local state regardless.
    } finally {
      setSpotifyUser(null);
      setSpotifyPlaylistUrl("");
      setNotice("Disconnected from Spotify.");
    }
  }

  async function createAndPlayMoodPlaylist() {
    if (!spotifyUser) {
      setError("Connect your Spotify account first.");
      return;
    }

    setCreatingPlaylist(true);
    setError("");
    setNotice(`Building a ${mood} playlist from your music map...`);
    setSpotifyPlaylistUrl("");

    try {
      const result = await apiRequest(
        `/playback/create-mood-playlist?mood=${encodeURIComponent(mood)}&limit=15`,
        { method: "POST" }
      );

      setSpotifyPlaylistUrl(result.playlist_url || "");

      const unresolvedNote = result.unresolved_tracks?.length
        ? ` (${result.unresolved_tracks.length} track(s) couldn't be resolved on Spotify and were skipped.)`
        : "";

      setNotice(
        `Created "${mood}" playlist with ${result.track_count} tracks on your Spotify account.${unresolvedNote}`
      );

      // Auto-play in the persistent SonicMap player when it is ready (Premium only).
      if (playback.isReady && result.uris?.length) {
        const started = await playback.playTracks(
          result.uris.map((uri) => ({ uri })),
          0,
          `SonicMap · ${mood} mix`
        );
        if (started) setNotice((prev) => `${prev} Now playing in SonicMap.`);
      }
    } catch (err) {
      setError(`Could not create playlist: ${apiErrorMessage(err)}`);
      setNotice("");
    } finally {
      setCreatingPlaylist(false);
    }
  }

  useEffect(() => {
    const element = plotRef.current;

    if (!element || points.length === 0) return;

    const trace = {
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      text: points.map(
        (p) =>
          `<b>${displayText(p.title, "Unknown track")}</b><br>` +
          `${displayText(p.artist, "Unknown artist")}<br>` +
          `Cluster ${displayText(p.cluster)}`
      ),
      customdata: points.map((p) => p.id),
      mode: "markers",
      type: "scattergl",
      marker: {
        size: 7,
        opacity: 0.78,
        color: points.map((p) =>
          typeof p.cluster === "number" ? p.cluster : 0
        ),
        colorscale: [
          [0, "#8b5cf6"],
          [0.25, "#ec4899"],
          [0.5, "#22d3ee"],
          [0.75, "#a3e635"],
          [1, "#fb923c"],
        ],
        showscale: false,
        line: { width: 0 },
      },
      hovertemplate: "%{text}<extra></extra>",
    };

    const handleClick = (event) => {
      const clicked = event.points?.[0];
      if (!clicked) return;

      const song = points.find((p) => p.id === clicked.customdata);
      if (song) setSelectedSong(song);
    };

    Plotly.react(
      element,
      [trace],
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 12, r: 12, b: 42, l: 42 },
        xaxis: {
          title: "Music dimension 1",
          color: "#89869d",
          gridcolor: "rgba(255,255,255,0.06)",
          zeroline: false,
        },
        yaxis: {
          title: "Music dimension 2",
          color: "#89869d",
          gridcolor: "rgba(255,255,255,0.06)",
          zeroline: false,
        },
        font: {
          family: "Inter, sans-serif",
          color: "#b5b1c8",
          size: 11,
        },
        hoverlabel: {
          bgcolor: "#201b30",
          bordercolor: "#8b5cf6",
          font: { color: "#ffffff" },
        },
        showlegend: false,
        autosize: true,
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
      }
    ).then(() => {
      if (plotRef.current === element) {
        element.on("plotly_click", handleClick);
      }
    }).catch((err) => {
      console.error("Could not render music map:", err);
    });

    return () => {
      Plotly.purge(element);
    };
  }, [points]);

  useEffect(() => {
    const element = myMapPlotRef.current;
    if (!element || personalMapPoints.length === 0) return;

    const filtered = personalQuery
      ? personalMapPoints.filter(
          (p) =>
            p.title.toLowerCase().includes(personalQuery.toLowerCase()) ||
            p.artist.toLowerCase().includes(personalQuery.toLowerCase())
        )
      : personalMapPoints;

    if (filtered.length === 0) {
      Plotly.purge(element);
      return;
    }

    const trace = {
      x: filtered.map((p) => p.x),
      y: filtered.map((p) => p.y),
      text: filtered.map((p) => {
        const cName = personalClusters.find((c) => c.id === p.cluster)?.name || `Cluster ${p.cluster}`;
        return (
          `<b>${displayText(p.title, "Unknown track")}</b><br>` +
          `${displayText(p.artist, "Unknown artist")}<br>` +
          `Group: ${cName}<br>` +
          `Energy: ${p.energy} · Valence: ${p.valence} · BPM: ${p.tempo}`
        );
      }),
      customdata: filtered.map((p) => p.id),
      mode: "markers",
      type: "scattergl",
      marker: {
        size: 8,
        opacity: 0.85,
        color: filtered.map((p) => (typeof p.cluster === "number" ? p.cluster : 0)),
        colorscale: [
          [0, "#ec4899"],
          [0.25, "#8b5cf6"],
          [0.5, "#06b6d4"],
          [0.75, "#10b981"],
          [1, "#f59e0b"],
        ],
        showscale: false,
        line: { width: 1, color: "rgba(255,255,255,0.25)" },
      },
      hovertemplate: "%{text}<extra></extra>",
    };

    const handleClick = (event) => {
      const clicked = event.points?.[0];
      if (!clicked) return;
      const song = filtered.find((p) => p.id === clicked.customdata);
      if (song) {
        setPersonalSelectedTrack(song);
        showPersonalRecommendations(song);
      }
    };

    Plotly.react(
      element,
      [trace],
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 12, r: 12, b: 42, l: 42 },
        xaxis: {
          title: "Personal audio dimension 1 (PCA)",
          color: "#89869d",
          gridcolor: "rgba(255,255,255,0.06)",
          zeroline: false,
        },
        yaxis: {
          title: "Personal audio dimension 2 (PCA)",
          color: "#89869d",
          gridcolor: "rgba(255,255,255,0.06)",
          zeroline: false,
        },
        font: {
          family: "Inter, sans-serif",
          color: "#b5b1c8",
          size: 11,
        },
        hoverlabel: {
          bgcolor: "#201b30",
          bordercolor: "#ec4899",
          font: { color: "#ffffff" },
        },
        showlegend: false,
        autosize: true,
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
      }
    ).then(() => {
      if (myMapPlotRef.current === element) {
        element.on("plotly_click", handleClick);
      }
    }).catch((err) => {
      console.error("Could not render personal music map:", err);
    });

    return () => {
      Plotly.purge(element);
    };
  }, [personalMapPoints, personalQuery, personalClusters]);

  async function searchSongs(event) {
    event.preventDefault();
    setError("");

    try {
      await loadSongs(query);
    } catch (err) {
      setError(`Search failed. ${err.message}`);
    }
  }

  
  async function showRecommendations(song) {
    const songId = song?.id;

    if (songId === undefined || songId === null) {
      setError("This track has no valid ID.");
      setNotice("");
      return;
    }

    setSelectedSong(song);
    setRecommendations([]);
    setRecommendationLoading(true);
    setError("");
    setNotice(`Finding tracks similar to ${song.title || song.track_name || "your selected song"}...`);

    // Bring the recommendation panel into view immediately.
    requestAnimationFrame(() => {
      document.getElementById("recommendations")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });

    try {
      const data = await apiRequest(`/recommend/${encodeURIComponent(songId)}?limit=8`);
      console.log("Recommendation API response:", data);

      let result = [];
      if (Array.isArray(data)) {
        result = data;
      } else if (Array.isArray(data?.recommendations)) {
        result = data.recommendations;
      } else if (Array.isArray(data?.songs)) {
        result = data.songs;
      } else if (Array.isArray(data?.similar_tracks)) {
        result = data.similar_tracks;
      } else if (Array.isArray(data?.results)) {
        result = data.results;
      }

      setRecommendations(result);
      setNotice(
        result.length
          ? `Found ${result.length} similar tracks.`
          : "The API returned no recommendations. Check the backend response."
      );
    } catch (err) {
      console.error("Recommendation request failed:", err);
      setError(`Recommendation failed: ${err.message}`);
      setNotice("");
    } finally {
      setRecommendationLoading(false);
    }
  }

  async function generatePlaylist() {
    setError("");

    try {
      const data = await apiRequest(
        `/playlist?mood=${encodeURIComponent(mood)}&limit=15`
      );

      setPlaylist(asArray(data, "songs"));
    } catch (err) {
      setError(`Could not generate playlist. ${err.message}`);
    }
  }

  return (
    <div className="app-shell has-player">
      <aside className="sidebar">
        <a href="#overview" className="brand">
          <div className="brand-mark">S</div>
          <div>
            <strong>SonicMap</strong>
            <span>MUSIC INTELLIGENCE</span>
          </div>
        </a>

        <div className="side-label">WORKSPACE</div>

        <a className="nav-item active" href="#overview">
          <span>◈</span> Overview
        </a>

        <a className="nav-item" href="#music-map">
          <span>⌘</span> Global Music Map
        </a>

        <a className="nav-item" href="#my-music-map">
          <span>✧</span> My Music Map
        </a>

        <a className="nav-item" href="#my-library">
          <span>♥</span> My Library
        </a>

        <a className="nav-item" href="#discover">
          <span>♫</span> Discover tracks
        </a>

        <a className="nav-item" href="#playlists">
          <span>▤</span> Mood playlists
        </a>

        <a className="nav-item" href="#sonic-playlists" data-testid="nav-sonic-playlists">
          <span>≋</span> SonicMap Playlists
        </a>

        <div className="sidebar-bottom">
          <div className="engine-card">
            <div className="engine-status">
              <span className="status-dot" /> ENGINE STATUS
            </div>
            <strong>
              {busy
                ? "Connecting..."
                : error
                  ? "Check connection"
                  : "Systems online"}
            </strong>
            <span>K-Means · PCA · Cosine similarity</span>
          </div>

          <div className="engine-card" style={{ marginTop: "10px" }}>
            <div className="engine-status">
              <span
                className="status-dot"
                style={{ background: spotifyUser ? "#1DB954" : "#6b7280" }}
              />
              SPOTIFY
            </div>
            {!authChecked ? (
              <strong>Checking...</strong>
            ) : spotifyUser ? (
              <>
                <strong>{spotifyUser.display_name}</strong>
                <span data-testid="sidebar-player-status">
                  {playback.phase === "ready"
                    ? "Web Player connected · playback enabled"
                    : playback.phase === "premium_required"
                      ? "Premium required for playback · open tracks in Spotify"
                      : playback.phase === "checking" || playback.phase === "connecting"
                        ? "Connecting Web Player…"
                        : "Playback unavailable — see player"}
                </span>
                <button className="text-link" onClick={disconnectSpotify} style={{ marginTop: "8px" }}>
                  Disconnect
                </button>
              </>
            ) : (
              <button
                className="text-link"
                onClick={connectSpotify}
                disabled={connecting}
                style={{ marginTop: "4px" }}
              >
                {connecting ? "Redirecting..." : "Connect Spotify ↗"}
              </button>
            )}
          </div>

          <div className="sidebar-footer">BUILT FOR THE CURIOUS</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>Overview</strong>
          </div>

          <div className="topbar-right">
            <span className="live-indicator">
              <span /> LIVE ENGINE
            </span>
            <div className="profile-avatar">SM</div>
          </div>
        </header>

        <section className="welcome-section" id="overview">
          <div>
            <div className="eyebrow">
              <span /> YOUR MUSIC, REIMAGINED
            </div>

            <h1>
              Find your next <span>favorite sound.</span>
            </h1>

            <p>
              Explore the hidden connections between tracks. Discover music
              by how it sounds, not just what genre it's labelled.
            </p>
          </div>

          {spotifyUser ? (
            <div className="upload-button" style={{ cursor: "default" }}>
              <span>✓</span>
              Connected as {spotifyUser.display_name}
            </div>
          ) : (
            <button className="upload-button" onClick={connectSpotify} disabled={connecting}>
              <span>♫</span>
              {connecting ? "Redirecting..." : "Connect Spotify"}
            </button>
          )}
        </section>

        {error && (
          <div className="alert error-alert">
            <span>!</span>
            <div>
              <strong>Something needs attention</strong>
              <p>{error}</p>
            </div>
            <button onClick={() => setError("")}>×</button>
          </div>
        )}

        {notice && (
          <div className="alert success-alert">
            <span>✓</span>
            <p>{notice}</p>
            <button onClick={() => setNotice("")}>×</button>
          </div>
        )}

        <section className="stats-grid">
          <StatCard
            label="TRACKS ANALYZED"
            value={totalSongs}
            detail="Songs in current dataset"
            icon="♫"
          />

          <StatCard
            label="MUSIC CLUSTERS"
            value={totalClusters}
            detail="Sound-based groupings"
            icon="◈"
          />

          <StatCard
            label="AUDIO FEATURES"
            value={featureCount}
            detail="Dimensions used for analysis"
            icon="⌁"
          />

          <StatCard
            label="SILHOUETTE SCORE"
            value={silhouette}
            detail="Cluster separation measure"
            icon="◎"
          />
        </section>

        <section className="content-grid" id="music-map">
          <div className="panel map-panel">
            <div className="panel-heading">
              <div>
                <div className="section-kicker">VISUAL EXPLORER</div>
                <h2>The music universe</h2>
                <p>
                  Each dot is a track. Nearby dots have similar audio
                  characteristics.
                </p>
              </div>

              <select
                value={cluster}
                onChange={(e) => setCluster(Number(e.target.value))}
                className="dark-select"
                aria-label="Filter by cluster"
              >
                <option value={-1}>All clusters</option>
                {Array.from(
                  { length: totalClusters },
                  (_, i) => (
                    <option value={i} key={i}>
                      Cluster {i}
                    </option>
                  )
                )}
              </select>
            </div>

            <div className="map-legend">
              <span>
                <i className="legend-purple" /> Sound clusters
              </span>
              <span className="map-count">
                {points.length.toLocaleString()} tracks plotted
              </span>
            </div>

            <div className="plot-container">
              {busy && points.length === 0 ? (
                <div className="empty-state">
                  <div className="loader" />
                  Building your music universe...
                </div>
              ) : points.length === 0 ? (
                <div className="empty-state">
                  No tracks found for this cluster.
                </div>
              ) : (
                <div ref={plotRef} className="music-plot" />
              )}
            </div>

            {selectedSong && (
              <div className="selected-track">
                <div className="song-avatar large">♫</div>

                <div className="song-info">
                  <strong>
                    {displayText(selectedSong.title, "Unknown track")}
                  </strong>
                  <span>
                    {displayText(selectedSong.artist, "Unknown artist")}
                  </span>
                </div>

                <button
                  className="primary-button"
                  onClick={() => showRecommendations(selectedSong)}
                >
                  Find similar
                </button>
              </div>
            )}
          </div>

          <div className="right-column">
            <div className="panel insight-panel">
              <div className="section-kicker">BEHIND THE SCENES</div>
              <h2>How SonicMap thinks</h2>

              <p className="muted-copy">
                Three techniques turn raw audio features into new discoveries.
              </p>

              <div className="method-row">
                <div className="method-icon purple">◉</div>
                <div>
                  <strong>K-Means clustering</strong>
                  <span>Groups tracks by audio similarity.</span>
                </div>
              </div>

              <div className="method-row">
                <div className="method-icon pink">⤢</div>
                <div>
                  <strong>Principal components</strong>
                  <span>Projects complex features onto a 2D map.</span>
                </div>
              </div>

              <div className="method-row">
                <div className="method-icon cyan">⌁</div>
                <div>
                  <strong>Cosine similarity</strong>
                  <span>Finds tracks with a similar sound profile.</span>
                </div>
              </div>
            </div>

            <div className="panel upload-panel">
              <div className="upload-icon">♫</div>
              <div>
                <strong>Play it on Spotify</strong>
                <p>
                  {spotifyUser
                    ? playback.phase === "premium_required"
                      ? "Connected — mood playlists are created on your account, but in-browser playback needs Spotify Premium."
                      : playback.isReady
                        ? "Web Player connected — press ▶ on any track, or generate a mood playlist below."
                        : "Connected — the SonicMap Web Player is starting (in-browser playback requires Spotify Premium)."
                    : "Connect your Spotify account to turn mood playlists into real, playable Spotify playlists."}
                </p>
              </div>

              {!spotifyUser && (
                <button className="text-link" onClick={connectSpotify} disabled={connecting}>
                  Connect Spotify ↗
                </button>
              )}
            </div>
          </div>
        </section>

        {/* --- MY MUSIC MAP SECTION (PHASE 4) --- */}
        <section className="content-grid" id="my-music-map" style={{ marginTop: "32px" }}>
          <div className="panel map-panel">
            <div className="panel-heading">
              <div>
                <div className="eyebrow" style={{ color: "#ec4899" }}>
                  <span style={{ background: "#ec4899" }} /> PERSONAL MUSIC MAP
                </div>
                <h2>My Music Map</h2>
                <p>
                  Personalized KMeans clustering & 2D PCA projected strictly from your private imported Spotify tracks.
                </p>
              </div>

              <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
                <select
                  value={personalCluster}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setPersonalCluster(val);
                    loadPersonalMap(val);
                  }}
                  className="dark-select"
                  aria-label="Filter by personal cluster"
                  disabled={!spotifyUser || (personalStatus?.eligible_tracks || 0) < 4}
                >
                  <option value={-1}>All clusters ({personalClusters.length})</option>
                  {personalClusters.map((c) => (
                    <option value={c.id} key={c.id}>
                      Cluster {c.id}: {c.name} ({c.count})
                    </option>
                  ))}
                </select>

                {spotifyUser && (
                  <button
                    className="primary-button"
                    onClick={trainPersonalModel}
                    disabled={personalTraining || (personalStatus?.eligible_tracks || 0) < 4}
                    style={{ padding: "8px 16px", fontSize: "12px", whiteSpace: "nowrap" }}
                    title="Retrain your personalized KMeans and PCA models"
                  >
                    {personalTraining ? "Training..." : "Retrain Map ↻"}
                  </button>
                )}
              </div>
            </div>

            {/* Explanation of eligible vs awaiting features */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "12px",
                padding: "10px 14px",
                margin: "12px 0 16px",
                background: "rgba(236, 72, 153, 0.08)",
                border: "1px solid rgba(236, 72, 153, 0.2)",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#f472b6",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "14px", flexWrap: "wrap" }}>
                <span>
                  <strong>{personalMapPoints.length}</strong> tracks plotted
                </span>
                <span>•</span>
                <span>
                  <strong>{personalStatus?.eligible_tracks || 0}</strong> eligible for clustering
                </span>
                <span>•</span>
                <span style={{ color: (personalStatus?.awaiting_features || 0) > 0 ? "#fbbf24" : "#a78bfa" }}>
                  <strong>{personalStatus?.awaiting_features || 0}</strong> awaiting audio features
                </span>
                {personalStatus?.silhouette_score !== null && personalStatus?.silhouette_score !== undefined && (
                  <>
                    <span>•</span>
                    <span>
                      Silhouette: <strong>{Number(personalStatus.silhouette_score).toFixed(3)}</strong>
                    </span>
                  </>
                )}
                {personalStatus?.model_version && (
                  <>
                    <span>•</span>
                    <span style={{ color: "#9ca3af" }}>Version: {personalStatus.model_version}</span>
                  </>
                )}
              </div>

              {(personalStatus?.awaiting_features || 0) > 0 && (
                <button
                  className="tag-button"
                  onClick={triggerFeatureAcquisition}
                  disabled={acquiringFeatures}
                  style={{
                    padding: "4px 10px",
                    background: "rgba(251, 191, 36, 0.15)",
                    border: "1px solid rgba(251, 191, 36, 0.35)",
                    color: "#fbbf24",
                    fontSize: "11px",
                    cursor: "pointer",
                    borderRadius: "4px",
                  }}
                >
                  {acquiringFeatures ? "Acquiring..." : "Acquire Missing Features ⟳"}
                </button>
              )}
            </div>

            {/* In-Map Search */}
            {spotifyUser && (personalStatus?.eligible_tracks || 0) >= 4 && (
              <div style={{ marginBottom: "12px" }}>
                <form
                  className="search-form"
                  onSubmit={(e) => e.preventDefault()}
                  style={{ maxWidth: "420px" }}
                >
                  <span>⌕</span>
                  <input
                    value={personalQuery}
                    onChange={(e) => setPersonalQuery(e.target.value)}
                    placeholder="Search personal map by title or artist..."
                  />
                  {personalQuery && (
                    <button type="button" onClick={() => setPersonalQuery("")}>
                      Clear
                    </button>
                  )}
                </form>
              </div>
            )}

            {/* Plot Container */}
            <div className="plot-container">
              {!spotifyUser ? (
                <div className="empty-state" style={{ minHeight: "335px" }}>
                  <div style={{ fontSize: "32px", marginBottom: "8px" }}>🔒</div>
                  <h3>Spotify Not Connected</h3>
                  <p className="muted-copy">
                    Connect your Spotify account to import your tracks and view your personal music map.
                  </p>
                  <button className="primary-button" onClick={connectSpotify} style={{ marginTop: "12px" }}>
                    Connect Spotify
                  </button>
                </div>
              ) : personalTraining || (personalLoading && personalMapPoints.length === 0) ? (
                <div className="empty-state" style={{ minHeight: "335px" }}>
                  <div className="loader" />
                  <span>Computing personalized KMeans & PCA projection...</span>
                </div>
              ) : (personalStatus?.eligible_tracks || 0) < 4 ? (
                <div className="empty-state" style={{ minHeight: "335px", padding: "20px" }}>
                  <div style={{ fontSize: "32px", marginBottom: "8px" }}>📊</div>
                  <h3>Personal Map Requires ≥ 4 Eligible Tracks</h3>
                  <p className="muted-copy" style={{ maxWidth: "540px" }}>
                    Your library has <strong>{personalStatus?.eligible_tracks || 0}</strong> eligible track(s).
                    Clustering requires complete 9-feature vectors (confidence ≥ 0.75).
                    Currently <strong>{personalStatus?.awaiting_features || 0}</strong> imported tracks are awaiting feature acquisition.
                  </p>
                  <div style={{ display: "flex", gap: "10px", marginTop: "14px" }}>
                    <button
                      className="primary-button"
                      onClick={triggerFeatureAcquisition}
                      disabled={acquiringFeatures}
                    >
                      {acquiringFeatures ? "Acquiring..." : "Acquire Features Now"}
                    </button>
                    <button
                      className="subtle-button"
                      onClick={triggerLibrarySync}
                      disabled={syncing}
                    >
                      {syncing ? "Syncing..." : "Sync Spotify Library"}
                    </button>
                  </div>
                </div>
              ) : personalMapPoints.length === 0 ? (
                <div className="empty-state" style={{ minHeight: "335px" }}>
                  No tracks found for this cluster or search filter.
                </div>
              ) : (
                <div ref={myMapPlotRef} className="music-plot" />
              )}
            </div>

            {/* Selected Track Details */}
            {personalSelectedTrack && (
              <div className="selected-track" style={{ marginTop: "16px", flexWrap: "wrap", gap: "16px" }}>
                <div className="song-avatar large">
                  {personalSelectedTrack.image_url ? (
                    <img
                      src={personalSelectedTrack.image_url}
                      alt=""
                      style={{ width: "100%", height: "100%", borderRadius: "8px", objectFit: "cover" }}
                    />
                  ) : (
                    <span>♫</span>
                  )}
                </div>

                <div className="song-info" style={{ flex: "1 1 200px" }}>
                  <strong>{displayText(personalSelectedTrack.title, "Unknown track")}</strong>
                  <span>{displayText(personalSelectedTrack.artist, "Unknown artist")}</span>
                  <div style={{ marginTop: "4px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                    <span
                      className="cluster-tag"
                      style={{
                        background: "rgba(236, 72, 153, 0.15)",
                        borderColor: "rgba(236, 72, 153, 0.4)",
                        color: "#f472b6",
                      }}
                    >
                      Cluster {personalSelectedTrack.cluster}: {personalClusters.find(c => c.id === personalSelectedTrack.cluster)?.name || `Group ${personalSelectedTrack.cluster + 1}`}
                    </span>
                    <span className="muted-copy" style={{ fontSize: "11px", alignSelf: "center" }}>
                      Energy: {personalSelectedTrack.energy} · Valence: {personalSelectedTrack.valence} · BPM: {personalSelectedTrack.tempo}
                    </span>
                  </div>
                </div>

                <PlayButton
                  tracks={[personalSelectedTrack, ...personalRecommendations]}
                  index={0}
                  label="My Music Map"
                  testId="my-map-selected-play-button"
                />
                <button
                  className="primary-button"
                  onClick={() => showPersonalRecommendations(personalSelectedTrack)}
                  disabled={personalRecLoading}
                  style={{ padding: "8px 18px", fontSize: "13px" }}
                >
                  {personalRecLoading ? "Finding..." : "Find Similar in Library"}
                </button>
              </div>
            )}
          </div>

          {/* Right Column: Personal ML Clusters & Similar Recommendations */}
          <div className="right-column">
            {/* Cluster Profiles Panel */}
            <div className="panel insight-panel">
              <div className="section-kicker">PERSONAL CLUSTER PROFILES</div>
              <h2>Sound Groups</h2>
              <p className="muted-copy">
                Clusters dynamically discovered from your personal listening habits.
              </p>

              {personalClusters.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "12px" }}>
                  {personalClusters.map((c) => (
                    <div
                      key={c.id}
                      className="method-row"
                      style={{
                        cursor: "pointer",
                        background: personalCluster === c.id ? "rgba(236, 72, 153, 0.12)" : "transparent",
                        borderRadius: "8px",
                        padding: "8px",
                        transition: "background 0.2s",
                      }}
                      onClick={() => {
                        const newCluster = personalCluster === c.id ? -1 : c.id;
                        setPersonalCluster(newCluster);
                        loadPersonalMap(newCluster);
                      }}
                    >
                      <div className="method-icon pink" style={{ fontSize: "12px" }}>
                        C{c.id}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <strong style={{ fontSize: "13px" }}>{c.name}</strong>
                          <span style={{ fontSize: "12px", color: "#ec4899" }}>{c.count} tracks</span>
                        </div>
                        {c.top_features && c.top_features.length > 0 && (
                          <span style={{ fontSize: "11px", color: "#9ca3af" }}>
                            Top features: {c.top_features.join(", ")}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state compact" style={{ padding: "20px" }}>
                  <p className="muted-copy">Cluster profiles will appear once your personal model is trained.</p>
                </div>
              )}
            </div>

            {/* Personal Similar Recommendations Panel */}
            <div className="panel recommendations-panel" style={{ marginTop: "20px" }}>
              <div className="panel-heading" style={{ marginBottom: "12px" }}>
                <div>
                  <div className="section-kicker">PERSONAL SIMILARITY</div>
                  <h2>
                    {personalSelectedTrack
                      ? `Similar to ${personalSelectedTrack.title}`
                      : "Library Recommendations"}
                  </h2>
                  <p className="muted-copy">
                    Private library tracks ranked by 9-feature cosine similarity.
                  </p>
                </div>
              </div>

              {personalRecLoading ? (
                <div className="empty-state compact">
                  <div className="loader" />
                  <span>Computing cosine similarities...</span>
                </div>
              ) : personalRecommendations.length > 0 ? (
                <div className="song-list">
                  {personalRecommendations.map((song, index) => (
                    <div key={song.id || index} className="recommendation-item">
                      <div className="recommendation-rank">#{index + 1}</div>
                      <div className="song-avatar">
                        {song.image_url ? (
                          <img
                            src={song.image_url}
                            alt=""
                            style={{ width: "100%", height: "100%", borderRadius: "8px", objectFit: "cover" }}
                          />
                        ) : (
                          <span>♫</span>
                        )}
                      </div>
                      <div className="song-info">
                        <strong>{song.title}</strong>
                        <span>{song.artist} {song.album ? `· ${song.album}` : ""}</span>
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "2px" }}>
                        <span
                          className="similarity-badge"
                          style={{
                            background: "rgba(236, 72, 153, 0.2)",
                            color: "#f472b6",
                            padding: "2px 8px",
                            borderRadius: "12px",
                            fontSize: "11px",
                            fontWeight: 600,
                          }}
                        >
                          {song.similarity}% match
                        </span>
                        <span style={{ fontSize: "10px", color: "#9ca3af" }}>
                          Cluster {song.cluster}
                        </span>
                      </div>
                      <PlayButton
                        tracks={personalRecommendations}
                        index={index}
                        label={personalSelectedTrack ? `Similar to ${personalSelectedTrack.title}` : "My Music Map"}
                        testId={`my-map-rec-play-button-${index}`}
                      />
                    </div>
                  ))}
                </div>
              ) : personalSelectedTrack ? (
                <div className="empty-state compact" style={{ padding: "16px" }}>
                  No other similar tracks found in your eligible library.
                </div>
              ) : (
                <div className="empty-state compact" style={{ padding: "16px" }}>
                  <p className="muted-copy">Click any track on My Music Map to view similar tracks from your own library.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* --- MY LIBRARY SECTION (PHASE 2) --- */}
        <section className="panel discovery-panel" id="my-library" style={{ marginTop: "32px" }}>
          <div className="panel-heading" style={{ flexWrap: "wrap", gap: "16px" }}>
            <div>
              <div className="eyebrow" style={{ color: "#ec4899" }}>
                <span style={{ background: "#ec4899" }} /> PERSONAL SPOTIFY REPOSITORY
              </div>
              <h2>My Library</h2>
              <p>
                Your imported Spotify tracks, playlists, top tracks, and recent listening history.
              </p>
            </div>

            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              {syncStatus?.last_synced_at && (
                <span className="muted-copy" style={{ fontSize: "12px" }}>
                  Last synced: {syncStatus.last_synced_at}
                </span>
              )}
              {spotifyUser ? (
                <button
                  className="primary-button"
                  onClick={triggerLibrarySync}
                  disabled={syncing}
                  style={{ padding: "8px 18px", fontSize: "13px" }}
                >
                  {syncing ? (
                    <span>Syncing {syncStatus?.progress || 0}%...</span>
                  ) : (
                    <span>Sync Library ↻</span>
                  )}
                </button>
              ) : (
                <button
                  className="upload-button"
                  onClick={connectSpotify}
                  style={{ padding: "8px 16px", fontSize: "13px" }}
                >
                  Connect Spotify to Import
                </button>
              )}
            </div>
          </div>

          {/* Sync Progress Bar */}
          {syncing && (
            <div style={{ margin: "16px 0", background: "rgba(255,255,255,0.06)", borderRadius: "8px", overflow: "hidden", height: "8px" }}>
              <div
                style={{
                  height: "100%",
                  width: `${syncStatus?.progress || 10}%`,
                  background: "linear-gradient(90deg, #8b5cf6, #ec4899)",
                  transition: "width 0.4s ease",
                }}
              />
            </div>
          )}

          {syncStatus?.message && (
            <div className="status-banner" style={{ marginBottom: "16px", padding: "10px 14px", background: "rgba(167, 139, 250, 0.08)", borderRadius: "8px", fontSize: "13px" }}>
              {syncStatus.message}
            </div>
          )}

          {/* Search and Source Filter Controls */}
          {spotifyUser && (
            <div style={{ display: "flex", gap: "14px", margin: "18px 0", flexWrap: "wrap", alignItems: "center" }}>
              <form
                className="search-form"
                style={{ flex: "1 1 280px" }}
                onSubmit={(e) => {
                  e.preventDefault();
                  loadLibrary(libraryQuery, librarySource);
                }}
              >
                <span>⌕</span>
                <input
                  value={libraryQuery}
                  onChange={(e) => setLibraryQuery(e.target.value)}
                  placeholder="Filter imported library by title, artist, album..."
                />
                <button type="submit">Filter</button>
              </form>

              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                {[
                  ["", "All Sources"],
                  ["liked", "♥ Liked Tracks"],
                  ["playlist", "▤ Playlists"],
                  ["recently_played", "⏱ Recently Played"],
                  ["top_tracks", "★ Top Tracks"],
                ].map(([val, label]) => (
                  <button
                    key={val}
                    className={`tag-button ${librarySource === val ? "active" : ""}`}
                    onClick={() => {
                      setLibrarySource(val);
                      loadLibrary(libraryQuery, val);
                    }}
                    style={{
                      padding: "6px 14px",
                      borderRadius: "20px",
                      border: librarySource === val ? "1px solid #a78bfa" : "1px solid rgba(255,255,255,0.1)",
                      background: librarySource === val ? "rgba(167, 139, 250, 0.2)" : "transparent",
                      color: librarySource === val ? "#f4f1ff" : "#9690a8",
                      fontSize: "12px",
                      cursor: "pointer",
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Track List / Empty State */}
          {!spotifyUser ? (
            <div className="empty-state" style={{ padding: "40px 20px", textAlign: "center" }}>
              <div style={{ fontSize: "32px", marginBottom: "12px" }}>🔒</div>
              <h3>Spotify Account Not Connected</h3>
              <p className="muted-copy">
                Connect your Spotify account to import your liked tracks, playlists, top songs, and recent playback history into SonicMap.
              </p>
              <button
                className="primary-button"
                onClick={connectSpotify}
                style={{ marginTop: "16px" }}
              >
                Connect Spotify Now
              </button>
            </div>
          ) : libraryLoading && libraryTracks.length === 0 ? (
            <div className="empty-state compact">
              Loading your library tracks...
            </div>
          ) : libraryTracks.length > 0 ? (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px", fontSize: "13px", color: "#9690a8" }}>
                <span>Showing {libraryTracks.length} of {libraryTotal} imported songs</span>
                <span>Category: {librarySource || "All sources"}</span>
                <PlayAllButton tracks={libraryTracks} label="My Library" testId="library-play-all-button" />
              </div>
              <div className="song-list">
                {libraryTracks.map((song, index) => (
                  <div key={song.id || index} className="song-row">
                    <div className="song-index">{String(index + 1).padStart(2, "0")}</div>
                    <div className="song-avatar">
                      {song.image_url ? (
                        <img src={song.image_url} alt="" style={{ width: "100%", height: "100%", borderRadius: "8px", objectFit: "cover" }} />
                      ) : (
                        <span>♫</span>
                      )}
                    </div>
                    <div className="song-info">
                      <strong>{song.title}</strong>
                      <span>{song.artist} {song.album ? `· ${song.album}` : ""}</span>
                    </div>
                    <span
                      className="cluster-tag"
                      style={{
                        background: "rgba(236, 72, 153, 0.12)",
                        borderColor: "rgba(236, 72, 153, 0.35)",
                        color: "#f472b6",
                        textTransform: "capitalize",
                      }}
                    >
                      {song.source?.replace("_", " ") || "imported"}
                    </span>
                    <PlayButton
                      tracks={libraryTracks}
                      index={index}
                      label="My Library"
                      testId={`library-play-button-${index}`}
                    />
                    <button
                      className="icon-button"
                      title="Find similar songs in your library"
                      onClick={() => showLibrarySimilar(song)}
                      data-testid={`library-similar-button-${index}`}
                    >
                      ↗
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-state" style={{ padding: "36px 20px", textAlign: "center" }}>
              <div style={{ fontSize: "28px", marginBottom: "8px" }}>♫</div>
              <h3>No tracks found in My Library</h3>
              <p className="muted-copy">
                {libraryQuery || librarySource
                  ? "No songs match your current search or source filter."
                  : "Click 'Sync Library ↻' above to import your liked songs, playlists, and listening history from Spotify."}
              </p>
              {!libraryQuery && !librarySource && (
                <button
                  className="primary-button"
                  onClick={triggerLibrarySync}
                  disabled={syncing}
                  style={{ marginTop: "14px" }}
                >
                  {syncing ? "Syncing..." : "Sync Spotify Library Now"}
                </button>
              )}
            </div>
          )}
        </section>

        <section className="panel discovery-panel" id="discover">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">YOUR NEXT OBSESSION</div>
              <h2>Explore tracks</h2>
              <p>
                Search the dataset and discover tracks beyond your usual rotation.
              </p>
            </div>

            <form className="search-form" onSubmit={searchSongs}>
              <span>⌕</span>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search tracks or artists..."
              />
              <button type="submit">Search</button>
            </form>
          </div>

          <div className="song-list">
            {songs.length ? (
              songs.slice(0, 12).map((song, index, shown) => (
                <SongRow
                  key={song.id ?? song.track_id ?? index}
                  song={song}
                  index={index}
                  onRecommend={showRecommendations}
                  queue={shown}
                  queueLabel="Explore tracks"
                  testIdPrefix="explore"
                />
              ))
            ) : (
              <div className="empty-state compact">
                No tracks to show. Try another search.
              </div>
            )}
          </div>

          <DiscoverForYou spotifyUser={spotifyUser} />
        </section>

        {selectedSong && (
          <section id="recommendations" className="panel recommendations-panel">
            <div className="panel-heading">
              <div>
                <div className="section-kicker">SOUND-ALIKE DISCOVERY</div>
                <h2>
                  Because you selected{" "}
                  {displayText(selectedSong?.title, "this track")}
                </h2>
                <p>Tracks with similar audio feature profiles.</p>
              </div>

              <button
                className="subtle-button"
                onClick={() => {
                  setRecommendations([]);
                  setSelectedSong(null);
                  setNotice("");
                  setError("");
                }}
              >
                Close ×
              </button>
            </div>

            <div className="song-list">
              {recommendationLoading ? (
                <div className="empty-state compact">
                  <div className="loader" />
                  Finding similar tracks...
                </div>
              ) : recommendations.length > 0 ? (
                recommendations.map((song, index) => (
                  <div key={song.id ?? song.track_id ?? index} className="recommendation-item">
                    <SongRow
                      song={song}
                      index={index}
                      onRecommend={showRecommendations}
                      queue={recommendations}
                      queueLabel={`Similar to ${displayText(selectedSong?.title, "this track")}`}
                      testIdPrefix="recommendation"
                    />
                    {Array.isArray(song.feature_comparison) && song.feature_comparison.length > 0 && (
                      <div
                        className="feature-comparison"
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: "8px",
                          padding: "0 16px 14px 58px",
                        }}
                      >
                        {song.feature_comparison.map((item) => (
                          <span
                            key={item.feature}
                            title={`Selected: ${item.selected}; recommended: ${item.recommended}`}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "6px",
                              padding: "5px 9px",
                              borderRadius: "999px",
                              border: "1px solid rgba(167, 139, 250, 0.25)",
                              background: "rgba(139, 92, 246, 0.08)",
                              color: "#cbd5e1",
                              fontSize: "11px",
                            }}
                          >
                            <span>{String(item.feature).replaceAll("_", " ")}</span>
                            <strong style={{ color: "#c4b5fd", fontWeight: 600 }}>
                              {Number(item.difference).toFixed(2)} apart
                            </strong>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="empty-state compact">
                  No recommendations available. Check the error or status message above.
                </div>
              )}
            </div>
          </section>
        )}

        <section className="panel playlist-panel" id="playlists">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">CURATED BY ALGORITHM</div>
              <h2>Set the mood</h2>
              <p>
                Generate a playlist using the audio qualities you feel like
                listening to.
              </p>
            </div>
          </div>

          <div className="mood-options">
            {[
              ["discovery", "✦", "Discovery", "A little bit of everything"],
              ["energetic", "ϟ", "High energy", "Tracks that go hard"],
              ["chill", "☾", "Chill mode", "Acoustic and mellow sounds"],
              ["dance", "♬", "Dance floor", "Made to move"],
              ["happy", "☼", "Feel good", "Bright, positive sounds"],
            ].map(([value, icon, title, description]) => (
              <button
                key={value}
                className={`mood-card ${mood === value ? "selected" : ""}`}
                onClick={() => setMood(value)}
              >
                <span className="mood-icon">{icon}</span>
                <strong>{title}</strong>
                <small>{description}</small>
              </button>
            ))}
          </div>

          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
            <button
              className="primary-button generate-button"
              onClick={generatePlaylist}
            >
              Preview playlist <span>→</span>
            </button>

            <button
              className="primary-button generate-button"
              onClick={createAndPlayMoodPlaylist}
              disabled={!spotifyUser || creatingPlaylist}
              title={!spotifyUser ? "Connect Spotify first" : undefined}
            >
              {creatingPlaylist ? "Creating on Spotify..." : "Create & play on Spotify"}{" "}
              <span>♫</span>
            </button>
          </div>

          {spotifyPlaylistUrl && (
            <p className="muted-copy" style={{ marginTop: "10px" }}>
              <a href={spotifyPlaylistUrl} target="_blank" rel="noreferrer" className="text-link">
                Open this playlist in Spotify ↗
              </a>
            </p>
          )}

          {playlist.length > 0 && (
            <div className="playlist-results">
              <div className="playlist-result-heading">
                <h3>Your {mood} mix</h3>
                <span>{playlist.length} tracks</span>
                <PlayAllButton tracks={playlist} label={`${mood} mix`} testId="mood-preview-play-all" />
              </div>

              <div className="song-list">
                {playlist.map((song, index) => (
                  <SongRow
                    key={song.id ?? song.track_id ?? index}
                    song={song}
                    index={index}
                    onRecommend={showRecommendations}
                    queue={playlist}
                    queueLabel={`${mood} mix`}
                    testIdPrefix="mood-preview"
                  />
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="panel playlist-panel" id="sonic-playlists" data-testid="sonic-playlists-section">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">PERSONALIZED AI PLAYLISTS</div>
              <h2>SonicMap Playlists</h2>
              <p>
                Playlists generated from your own library clusters (plus discovery picks), saved in SonicMap and
                playable right here.
              </p>
            </div>
          </div>
          <SonicPlaylists spotifyUser={spotifyUser} />
        </section>

        <footer className="app-footer">
          <div className="footer-brand">
            <span>♫</span> SonicMap
          </div>
          <span>Discover the sound between genres.</span>
          <span>POWERED BY MACHINE LEARNING</span>
        </footer>
      </main>

      <PlayerBar onConnect={connectSpotify} />
    </div>
  );
}
