"""
Audio Feature Acquisition Engine for SonicMap (Phase 3B)

Architecture:
Tier 1: Reference Catalog Matching (Exact Spotify ID + Normalized Title/Artist/Duration)
Tier 2: Librosa Signal Analysis for Physical Descriptors (Tempo, Loudness, RMS energy proxy)
Tier 3: Modular ML Feature Estimator (Checks dataset availability, reports metrics or honest unavailability)

Rules:
- Never fabricate missing values.
- Keep missing features nullable.
- Separate match_confidence from feature_confidence.
- Minimum feature confidence >= 0.75 and all 9 features present for clustering eligibility.
"""

import io
import re
import math
import logging
import datetime
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from models import Track, AudioFeatureProvenance, UserLibraryTrack, utcnow
from state import BASE_DIR, AUDIO_FEATURES

logger = logging.getLogger("sonicmap.feature_engine")

# Feature validation bounds based on standard audio feature domains
FEATURE_BOUNDS = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "loudness": (-60.0, 5.0),  # standard dB range
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (30.0, 300.0),  # BPM range
}

MIN_CLUSTERING_CONFIDENCE = 0.75


def validate_feature_value(name: str, val: float | None) -> bool:
    """Validate that a feature value is within realistic domain boundaries."""
    if val is None or not math.isfinite(val):
        return False
    bounds = FEATURE_BOUNDS.get(name)
    if bounds:
        low, high = bounds
        return low <= val <= high
    return True


def normalize_string(s) -> str:
    """Clean string for safe metadata matching (lowercase, strip remaster/deluxe suffixes)."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).lower().strip()
    # Strip (remastered...), [deluxe...], (feat...), etc.
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


class ReferenceCatalogMatcher:
    """
    Tier 1 Matcher:
    Matches imported tracks against verified open reference datasets (e.g. Kaggle 114k Spotify Tracks dataset).
    """

    def __init__(self, catalog_path: Path | None = None):
        self.catalog_path = catalog_path or (BASE_DIR.parent / "dataset-source" / "dataset.csv")
        self._df = None
        self._id_index = {}
        self._title_artist_index = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not self.catalog_path.exists():
            logger.warning(f"Reference catalog not found at {self.catalog_path}. Tier 1 matching disabled.")
            return

        cols = [
            "track_id", "artists", "track_name", "album_name", "duration_ms",
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"
        ]
        try:
            df = pd.read_csv(self.catalog_path, usecols=cols)
            df = df.dropna(subset=AUDIO_FEATURES + ["track_id"])
            self._df = df

            # Build fast lookup indexes
            for _, row in df.iterrows():
                tid = str(row["track_id"]).strip()
                if tid and tid not in self._id_index:
                    self._id_index[tid] = row

                    # Normalized title + primary artist key (deduplicating same track_id across genres)
                    norm_title = normalize_string(row.get("track_name"))
                    artists_raw = str(row.get("artists") or "").split(";")[0].split(",")[0]
                    norm_artist = normalize_string(artists_raw)
                    if norm_title and norm_artist:
                        key = f"{norm_title}___{norm_artist}"
                        if key not in self._title_artist_index:
                            self._title_artist_index[key] = []
                        self._title_artist_index[key].append(row)

            self._loaded = True
            logger.info(f"Reference catalog loaded with {len(self._id_index)} indexed tracks.")
        except Exception as exc:
            logger.error(f"Failed to load reference catalog: {exc}")

    def match_track(self, spotify_id: str | None, title: str, artist: str, duration_ms: int | None = None) -> tuple[dict | None, float, float, str]:
        """
        Attempts to match a track.
        Returns: (features_dict, match_confidence, feature_confidence, source_note) or (None, 0, 0, "")
        """
        self.load()
        if not self._loaded:
            return None, 0.0, 0.0, "Reference catalog unavailable"

        # 1. Exact Spotify ID match
        if spotify_id and spotify_id in self._id_index:
            row = self._id_index[spotify_id]
            features = {f: float(row[f]) for f in AUDIO_FEATURES}
            return features, 1.0, 1.0, "Tier 1: Exact Spotify ID match"

        # 2. Normalized Title + Primary Artist Match
        norm_title = normalize_string(title)
        primary_artist = artist.split(",")[0].split(";")[0]
        norm_artist = normalize_string(primary_artist)
        key = f"{norm_title}___{norm_artist}"

        candidates = self._title_artist_index.get(key, [])
        if not candidates:
            return None, 0.0, 0.0, "No catalog match"

        # Reject ambiguous matches if multiple conflicting candidates exist with no duration to disambiguate
        if len(candidates) > 1 and duration_ms:
            # Filter candidates within +- 4 seconds (4000 ms)
            close = [c for c in candidates if abs(c.get("duration_ms", 0) - duration_ms) <= 4000]
            if len(close) == 1:
                row = close[0]
                features = {f: float(row[f]) for f in AUDIO_FEATURES}
                return features, 0.88, 1.0, "Tier 1: Disambiguated metadata & duration match"
            elif len(close) > 1:
                return None, 0.0, 0.0, "Ambiguous metadata match rejected"
        elif len(candidates) == 1:
            row = candidates[0]
            # Verify duration if available
            cand_dur = row.get("duration_ms")
            if duration_ms and cand_dur and abs(cand_dur - duration_ms) > 12000:
                return None, 0.0, 0.0, "Duration mismatch rejected"

            features = {f: float(row[f]) for f in AUDIO_FEATURES}
            return features, 0.85, 1.0, "Tier 1: Metadata match"

        return None, 0.0, 0.0, "Ambiguous catalog match"


# Singleton catalog instance
catalog_matcher = ReferenceCatalogMatcher()


async def extract_librosa_physical_features(preview_url: str) -> dict | None:
    """
    Tier 2: Extracts measurable signal descriptors from legal 30-second audio previews using Librosa.
    Calculates:
      - tempo (BPM)
      - loudness (integrated RMS converted to standard dB)
      - energy proxy (RMS energy normalized)
    NOTE: Does NOT fabricate valence or danceability; those remain null.
    """
    if not preview_url:
        return None

    try:
        import librosa
        import soundfile
    except ImportError:
        logger.warning("Librosa or soundfile not installed. Tier 2 extraction unavailable.")
        return None

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.get(preview_url)
            if res.status_code != 200:
                logger.info(f"Could not download preview audio ({res.status_code}): {preview_url}")
                return None
            audio_bytes = res.content

        # Load audio using soundfile / librosa from in-memory stream
        with io.BytesIO(audio_bytes) as bio:
            y, sr = librosa.load(bio, sr=22050, mono=True, duration=30.0)

        if len(y) == 0:
            return None

        # 1. Tempo calculation (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo_val = float(tempo)

        # 2. Loudness (RMS dB calculation)
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))
        # dB = 20 * log10(rms) clipped to [-60.0, 0.0]
        if rms_mean > 0:
            db_loudness = max(-60.0, min(0.0, 20.0 * math.log10(rms_mean)))
        else:
            db_loudness = -60.0

        # 3. Measurable signal energy proxy (normalized RMS intensity)
        # Note: Clearly labeled as signal RMS energy proxy, not Spotify's closed proprietary formula
        raw_energy = min(1.0, max(0.0, rms_mean * 3.5))

        return {
            "tempo": tempo_val,
            "loudness": db_loudness,
            "energy": raw_energy,
        }
    except Exception as exc:
        logger.warning(f"Librosa extraction failed on {preview_url}: {exc}")
        return None


class MLEstimationService:
    """
    Tier 3: Machine Learning Audio Feature Estimator.
    Requires validated labelled training dataset.
    Reports honest model metrics and status.
    """

    def __init__(self):
        self.is_model_trained = False
        self.validation_metrics = {}

    def get_status(self) -> dict:
        return {
            "available": self.is_model_trained,
            "model_type": "Multi-output Random Forest Regressor" if self.is_model_trained else "None",
            "training_dataset_status": "No independently validated multi-feature audio preview training set bundled.",
            "metrics": self.validation_metrics,
            "note": "Per strict requirements, estimation is reported as unavailable until licensed paired audio dataset is verified.",
        }

    def estimate_features(self, audio_features_so_far: dict) -> tuple[dict | None, float]:
        """
        Estimates remaining psychoacoustic features only if a trained model is ready.
        Otherwise returns (None, 0.0) honestly.
        """
        if not self.is_model_trained:
            return None, 0.0
        # If model is ready in future phases, predict and return confidence
        return None, 0.0


ml_estimator = MLEstimationService()


def check_clustering_eligibility(track: Track) -> tuple[bool, str]:
    """
    Evaluates whether a track satisfies all clustering eligibility rules:
    - All 9 features present (not null).
    - Every feature has a recorded confidence >= 0.75.
    - All values pass feature-specific validation bounds.
    """
    for feat in AUDIO_FEATURES:
        val = getattr(track, feat, None)
        if val is None:
            return False, f"Missing feature: {feat}"
        if not validate_feature_value(feat, val):
            return False, f"Feature '{feat}' value {val} out of valid bounds"

    # Check provenance confidence
    if not track.provenance:
        return False, "No feature provenance recorded"

    for prov in track.provenance:
        if prov.confidence < MIN_CLUSTERING_CONFIDENCE:
            return False, f"Feature '{prov.feature_name}' confidence {prov.confidence:.2f} < {MIN_CLUSTERING_CONFIDENCE}"

    return True, "Eligible for personalized clustering"


async def acquire_features_for_track(db: Session, track: Track) -> dict:
    """
    Executes the 3-tier acquisition workflow for a single track:
    1. Try Tier 1 Reference Catalog Match.
    2. If missing, try Tier 2 Librosa preview analysis.
    3. If still missing, check Tier 3 ML Estimation.
    4. Record provenance and update track fields safely.
    """
    # 1. Check if already has complete, high-confidence features
    eligible, reason = check_clustering_eligibility(track)
    if eligible:
        return {"track_id": track.id, "status": "already_complete", "details": reason}

    # Tier 1: Reference Catalog
    features, match_conf, feat_conf, note = catalog_matcher.match_track(
        spotify_id=track.spotify_id,
        title=track.title,
        artist=track.artist,
        duration_ms=track.duration_ms,
    )

    if features and match_conf >= 0.75:
        # All 9 features acquired from catalog
        for f, val in features.items():
            if validate_feature_value(f, val):
                setattr(track, f, val)

        combined_conf = match_conf * feat_conf
        prov = AudioFeatureProvenance(
            track_id=track.id,
            feature_name="all_9",
            source="reference_catalog",
            acquisition_type="matched",
            match_confidence=match_conf,
            feature_confidence=feat_conf,
            confidence=combined_conf,
            notes=note,
        )
        db.add(prov)
        db.commit()
        return {"track_id": track.id, "status": "matched_catalog", "confidence": combined_conf}

    # Tier 2: Librosa Signal Analysis for Physical Descriptors (if preview_url available)
    if track.preview_url:
        librosa_res = await extract_librosa_physical_features(track.preview_url)
        if librosa_res:
            # Set measurable features
            for f, val in librosa_res.items():
                if validate_feature_value(f, val):
                    setattr(track, f, val)
                prov = AudioFeatureProvenance(
                    track_id=track.id,
                    feature_name=f,
                    source="librosa_audio_preview",
                    acquisition_type="measured",
                    match_confidence=1.0,
                    feature_confidence=0.85 if f in ("tempo", "loudness") else 0.70,
                    confidence=0.85 if f in ("tempo", "loudness") else 0.70,
                    notes=f"Physical signal measurement from preview audio ({f})",
                )
                db.add(prov)
            db.commit()
            return {"track_id": track.id, "status": "partial_librosa", "features_acquired": list(librosa_res.keys())}

    # Tier 3: ML estimation
    est_status = ml_estimator.get_status()
    return {
        "track_id": track.id,
        "status": "unacquired",
        "reason": "No catalog match and no preview audio available for signal analysis.",
        "ml_estimation_status": est_status["note"],
    }
