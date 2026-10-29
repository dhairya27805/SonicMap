
import { useCallback, useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import "./App.css";

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api").replace(/\/$/, "");

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

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

function SongRow({ song, onRecommend, index }) {
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
        <span>♫</span>
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

      <button
        className="icon-button"
        title="Find similar songs"
        onClick={() => onRecommend(song)}
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

export default function App() {
  const plotRef = useRef(null);

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
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");

  const totalSongs = getSongCount(overview);
  const totalClusters = getClusterCount(overview);
  const featureCount = getFeatureCount(overview);
  const silhouette = getSilhouette(overview);

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

  async function uploadDataset(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const result = await apiRequest("/upload", {
        method: "POST",
        body: formData,
      });

      setNotice(result.message || "Dataset uploaded successfully.");
      setCluster(-1);
      setRecommendations([]);
      setPlaylist([]);

      await loadOverview();
      await loadMap(-1);
      await loadSongs("");
      setQuery("");
    } catch (err) {
      setError(`Upload failed. ${err.message}`);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  return (
    <div className="app-shell">
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
          <span>⌘</span> Music map
        </a>

        <a className="nav-item" href="#discover">
          <span>♫</span> Discover tracks
        </a>

        <a className="nav-item" href="#playlists">
          <span>▤</span> Mood playlists
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

          <label className="upload-button">
            <span>↑</span>
            {uploading ? "Uploading..." : "Upload dataset"}
            <input
              type="file"
              accept=".csv"
              onChange={uploadDataset}
              disabled={uploading}
              hidden
            />
          </label>
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
              <div className="upload-icon">⇧</div>
              <div>
                <strong>Bring your own music data</strong>
                <p>
                  Upload a CSV with audio features to explore your own collection.
                </p>
              </div>

              <label className="text-link">
                Choose CSV ↗
                <input
                  type="file"
                  accept=".csv"
                  onChange={uploadDataset}
                  hidden
                  disabled={uploading}
                />
              </label>
            </div>
          </div>
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
              songs.slice(0, 12).map((song, index) => (
                <SongRow
                  key={song.id ?? song.track_id ?? index}
                  song={song}
                  index={index}
                  onRecommend={showRecommendations}
                />
              ))
            ) : (
              <div className="empty-state compact">
                No tracks to show. Try another search.
              </div>
            )}
          </div>
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

          <button
            className="primary-button generate-button"
            onClick={generatePlaylist}
          >
            Generate playlist <span>→</span>
          </button>

          {playlist.length > 0 && (
            <div className="playlist-results">
              <div className="playlist-result-heading">
                <h3>Your {mood} mix</h3>
                <span>{playlist.length} tracks</span>
              </div>

              <div className="song-list">
                {playlist.map((song, index) => (
                  <SongRow
                    key={song.id ?? song.track_id ?? index}
                    song={song}
                    index={index}
                    onRecommend={showRecommendations}
                  />
                ))}
              </div>
            </div>
          )}
        </section>

        <footer className="app-footer">
          <div className="footer-brand">
            <span>♫</span> SonicMap
          </div>
          <span>Discover the sound between genres.</span>
          <span>POWERED BY MACHINE LEARNING</span>
        </footer>
      </main>
    </div>
  );
}
