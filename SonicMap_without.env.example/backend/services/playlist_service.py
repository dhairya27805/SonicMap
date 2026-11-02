"""
SonicMap Playlist Generation & Management Service (Phase 6)

Architecture:
- Uses Phase 4 personalized KMeans clusters and cosine similarity for library-based selection.
- Uses Phase 5 discovery service for external track augmentation.
- Stores playlists using existing Playlist and PlaylistItem DB models.
- Never fabricates feature values.
- Gracefully handles insufficient eligible tracks.

Playlist types supported:
  chill_relax, high_energy, workout, focus, discover_weekly, late_night,
  similar_sounds, explore_new, mood, cluster, seed_track, custom
"""

import logging
import math
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Playlist, PlaylistItem, Track, UserLibraryTrack, utcnow
from state import AUDIO_FEATURES, RANDOM_STATE
from services.feature_engine_service import check_clustering_eligibility
from services.personal_ml_service import get_eligible_user_tracks, get_personal_model
from services.discovery_service import (
    get_discovery_recommendations,
    get_user_excluded_signatures,
    reference_catalogue,
)

logger = logging.getLogger("sonicmap.playlist")

# ── Preset audio-feature profiles for each playlist type ─────────────────────
PLAYLIST_PROFILES = {
    "chill_relax": {
        "label": "Chill & Relax",
        "description": "Low energy, acoustic, and peaceful sounds for winding down.",
        "target": {"energy": (0.0, 0.45), "acousticness": (0.55, 1.0), "valence": (0.2, 0.7)},
        "mood_key": "chill",
    },
    "high_energy": {
        "label": "High Energy",
        "description": "Loud, fast, and high-energy tracks.",
        "target": {"energy": (0.7, 1.0), "tempo": (120.0, 300.0)},
        "mood_key": "energetic",
    },
    "workout": {
        "label": "Workout",
        "description": "High danceability and tempo to keep you moving.",
        "target": {"energy": (0.75, 1.0), "danceability": (0.65, 1.0), "tempo": (125.0, 300.0)},
        "mood_key": "energetic",
    },
    "focus": {
        "label": "Focus",
        "description": "Instrumental, low speechiness — ideal for deep work.",
        "target": {"instrumentalness": (0.4, 1.0), "speechiness": (0.0, 0.1), "energy": (0.2, 0.6)},
        "mood_key": "focus",
    },
    "discover_weekly": {
        "label": "Discover Weekly",
        "description": "A balanced mix of your library sounds and fresh discoveries.",
        "target": {},
        "mood_key": None,
    },
    "late_night": {
        "label": "Late Night",
        "description": "Low valence, moody, and atmospheric.",
        "target": {"valence": (0.0, 0.4), "energy": (0.3, 0.75), "acousticness": (0.2, 0.8)},
        "mood_key": "dark",
    },
    "similar_sounds": {
        "label": "Similar Sounds",
        "description": "Sounds consistent with your library's overall audio character.",
        "target": {},
        "mood_key": None,
    },
    "explore_new": {
        "label": "Explore Something New",
        "description": "Deliberately ventures into under-represented sound territory.",
        "target": {},
        "mood_key": "different",
    },
}

MIN_TRACKS_FOR_PLAYLIST = 3


# ── Helper: track feature vector ─────────────────────────────────────────────

def _track_vector(track: Track) -> np.ndarray | None:
    """Returns the 9-feature numpy vector for a Track, or None if incomplete."""
    try:
        vals = [float(getattr(track, f)) for f in AUDIO_FEATURES]
        if any(v is None for v in vals):
            return None
        return np.array(vals)
    except (TypeError, AttributeError):
        return None


# ── Helper: filter library tracks by feature ranges ──────────────────────────

def _filter_by_ranges(tracks: list, ranges: dict) -> list:
    """Return tracks whose feature values fall inside the given (min, max) ranges."""
    if not ranges:
        return tracks
    result = []
    for t in tracks:
        ok = True
        for feat, (lo, hi) in ranges.items():
            val = getattr(t, feat, None)
            if val is None or not (lo <= float(val) <= hi):
                ok = False
                break
        if ok:
            result.append(t)
    return result


# ── Helper: deduplicate tracks by DB id ──────────────────────────────────────

def _dedup(tracks: list, seen_ids: set) -> list:
    out = []
    for t in tracks:
        tid = t if isinstance(t, int) else t.id
        if tid not in seen_ids:
            seen_ids.add(tid)
            out.append(t)
    return out


# ── Helper: sort library tracks by cosine similarity to a target vector ───────

def _sort_by_similarity(tracks: list, target_vec: np.ndarray) -> list:
    """Return tracks sorted by cosine similarity to target_vec (descending)."""
    pairs = []
    for t in tracks:
        v = _track_vector(t)
        if v is None:
            pairs.append((0.0, t))
            continue
        sim = float(cosine_similarity(target_vec.reshape(1, -1), v.reshape(1, -1))[0][0])
        pairs.append((sim, t))
    pairs.sort(key=lambda x: -x[0])
    return [t for _, t in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

async def generate_playlist_tracks(
    db: Session,
    user_id: int,
    playlist_type: str = "chill_relax",
    cluster_id: int = -1,
    seed_track_id: int | None = None,
    mood: str | None = None,
    energy_min: float = 0.0,
    energy_max: float = 1.0,
    danceability_min: float = 0.0,
    danceability_max: float = 1.0,
    tempo_min: float = 30.0,
    tempo_max: float = 300.0,
    target_size: int = 20,
    include_discoveries: bool = True,
) -> dict:
    """
    Generate a list of track dicts for a playlist.
    Returns both library tracks and optional discovery tracks.
    Never fabricates feature values.
    """
    target_size = max(1, min(target_size, 100))
    profile = PLAYLIST_PROFILES.get(playlist_type, PLAYLIST_PROFILES["similar_sounds"])
    eligible_tracks = get_eligible_user_tracks(db, user_id)
    model = get_personal_model(db, user_id)

    # Start with all eligible library tracks
    library_pool = list(eligible_tracks)

    # Apply user's filter constraints
    user_ranges = {
        "energy": (energy_min, energy_max),
        "danceability": (danceability_min, danceability_max),
        "tempo": (tempo_min, tempo_max),
    }
    library_pool = _filter_by_ranges(library_pool, user_ranges)

    # Merge profile feature ranges
    profile_ranges = {k: v for k, v in profile["target"].items() if k not in ("tempo", "energy", "danceability")}
    # Also apply profile ranges on top of user ranges where not overridden
    if "energy" in profile["target"]:
        lo, hi = profile["target"]["energy"]
        library_pool = _filter_by_ranges(library_pool, {"energy": (max(energy_min, lo), min(energy_max, hi))})
    if profile_ranges:
        library_pool = _filter_by_ranges(library_pool, profile_ranges)
    if "tempo" in profile["target"] and profile["target"]["tempo"][0] > 30.0:
        lo, hi = profile["target"]["tempo"]
        library_pool = _filter_by_ranges(library_pool, {"tempo": (max(tempo_min, lo), min(tempo_max, hi))})

    # Cluster filter
    if cluster_id >= 0 and model:
        df = model["df"]
        cluster_ids_in_cluster = set(df[df["cluster"] == cluster_id]["id"].tolist())
        library_pool = [t for t in library_pool if t.id in cluster_ids_in_cluster]

    # Seed-track: cosine-similarity ranking from a specific track
    target_vec = None
    if seed_track_id is not None:
        seed = db.query(Track).filter(Track.id == seed_track_id).first()
        if seed:
            target_vec = _track_vector(seed)
            if target_vec is not None:
                library_pool = _sort_by_similarity(library_pool, target_vec)
                # Exclude seed track itself from pool
                library_pool = [t for t in library_pool if t.id != seed_track_id]
    elif model and len(model["df"]) > 0:
        target_vec = model["df"][AUDIO_FEATURES].mean().to_numpy()
        library_pool = _sort_by_similarity(library_pool, target_vec)

    # Allocation: take up to (target_size * 0.6) library tracks, rest from discoveries
    max_library = max(1, math.ceil(target_size * 0.6)) if include_discoveries else target_size
    selected_library = library_pool[:max_library]
    seen_ids: set = {t.id for t in selected_library}

    shortfall_reason = None
    if len(eligible_tracks) < MIN_TRACKS_FOR_PLAYLIST:
        shortfall_reason = (
            f"Only {len(eligible_tracks)} eligible tracks in your library "
            f"(requires {MIN_TRACKS_FOR_PLAYLIST}+ tracks with all 9 audio features). "
            "Import more tracks and acquire audio features to grow this playlist."
        )

    # Serialise library tracks
    def _serialise_library(t: Track) -> dict:
        return {
            "id": t.id,
            "spotify_id": t.spotify_id or "",
            "title": t.title,
            "artist": t.artist,
            "album": t.album or "",
            "image_url": t.image_url or "",
            "preview_url": t.preview_url or "",
            "duration_ms": t.duration_ms or 0,
            "source": "library",
            "has_features": True,
            "features": {f: round(float(getattr(t, f)), 4) for f in AUDIO_FEATURES if getattr(t, f) is not None},
        }

    library_items = [_serialise_library(t) for t in selected_library]

    # Discovery tracks
    discovery_items = []
    if include_discoveries and len(library_items) < target_size:
        slots = target_size - len(library_items)
        try:
            discovery_mode = profile.get("mood_key") or "similar_to_library"
            if discovery_mode == "different":
                discovery_mode_param = "different"
            elif discovery_mode and discovery_mode != "similar_to_library":
                discovery_mode_param = "mood"
            else:
                discovery_mode_param = "similar_to_library"

            disc_result = await get_discovery_recommendations(
                db=db,
                user_id=user_id,
                mode=discovery_mode_param,
                cluster_id=cluster_id,
                mood=profile.get("mood_key"),
                page=1,
                limit=slots * 2,
            )
            for item in disc_result.get("recommendations", []):
                # Filter discovery tracks by user constraints (where features exist)
                if item.get("has_features") and item.get("features"):
                    feats = item["features"]
                    if not (energy_min <= feats.get("energy", 0.5) <= energy_max):
                        continue
                    if not (danceability_min <= feats.get("danceability", 0.5) <= danceability_max):
                        continue
                    if not (tempo_min <= feats.get("tempo", 120.0) <= tempo_max):
                        continue

                item["source"] = "discovery"
                item.setdefault("recommendation_reason", "Recommended based on your playlist profile")
                discovery_items.append(item)
                if len(discovery_items) >= slots:
                    break
        except Exception as exc:
            logger.warning(f"Discovery augmentation failed: {exc}")

    all_tracks = library_items + discovery_items

    return {
        "playlist_type": playlist_type,
        "playlist_label": profile["label"],
        "description": profile["description"],
        "target_size": target_size,
        "tracks_generated": len(all_tracks),
        "library_tracks": len(library_items),
        "discovery_tracks": len(discovery_items),
        "shortfall_reason": shortfall_reason,
        "tracks": all_tracks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLAYLIST CRUD
# ─────────────────────────────────────────────────────────────────────────────

def get_user_playlists(db: Session, user_id: int) -> list[dict]:
    """Return all playlists for a user (summary — no track details)."""
    playlists = (
        db.query(Playlist)
        .filter(Playlist.user_id == user_id)
        .order_by(Playlist.updated_at.desc())
        .all()
    )
    return [_playlist_summary(p) for p in playlists]


def get_playlist_detail(db: Session, user_id: int, playlist_id: int) -> dict | None:
    """Return full playlist details including ordered tracks."""
    p = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user_id).first()
    if not p:
        return None
    return _playlist_detail(p)


def create_empty_playlist(db: Session, user_id: int, name: str, description: str = "", playlist_type: str = "custom") -> dict:
    """Create a new empty playlist."""
    p = Playlist(
        user_id=user_id,
        name=name[:255],
        description=description,
        playlist_type=playlist_type,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _playlist_summary(p)


def update_playlist(db: Session, user_id: int, playlist_id: int, name: str | None = None, description: str | None = None) -> dict | None:
    """Rename or re-describe a playlist."""
    p = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user_id).first()
    if not p:
        return None
    if name is not None:
        p.name = name[:255]
    if description is not None:
        p.description = description
    p.updated_at = utcnow()
    db.commit()
    db.refresh(p)
    return _playlist_summary(p)


def delete_playlist(db: Session, user_id: int, playlist_id: int) -> bool:
    """Delete a playlist (cascade deletes PlaylistItems)."""
    p = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user_id).first()
    if not p:
        return False
    db.delete(p)
    db.commit()
    return True


def add_track_to_playlist(db: Session, user_id: int, playlist_id: int, track_id: int) -> dict | None:
    """Add a track to a playlist at the next available position."""
    p = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user_id).first()
    if not p:
        return None
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        return None
    # Check for duplicate
    existing = db.query(PlaylistItem).filter(
        PlaylistItem.playlist_id == playlist_id,
        PlaylistItem.track_id == track_id,
    ).first()
    if existing:
        return {"status": "already_present", "playlist_id": playlist_id, "track_id": track_id}

    max_pos = db.query(PlaylistItem).filter(PlaylistItem.playlist_id == playlist_id).count()
    item = PlaylistItem(playlist_id=playlist_id, track_id=track_id, position=max_pos)
    db.add(item)
    p.updated_at = utcnow()
    db.commit()
    return {"status": "added", "playlist_id": playlist_id, "track_id": track_id, "position": max_pos}


def remove_track_from_playlist(db: Session, user_id: int, playlist_id: int, track_id: int) -> bool:
    """Remove a track from a playlist and compact positions."""
    p = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user_id).first()
    if not p:
        return False
    item = db.query(PlaylistItem).filter(
        PlaylistItem.playlist_id == playlist_id,
        PlaylistItem.track_id == track_id,
    ).first()
    if not item:
        return False
    removed_pos = item.position
    db.delete(item)
    # Compact: shift all items above the removed position down by 1
    items_above = (
        db.query(PlaylistItem)
        .filter(PlaylistItem.playlist_id == playlist_id, PlaylistItem.position > removed_pos)
        .all()
    )
    for i in items_above:
        i.position -= 1
    p.updated_at = utcnow()
    db.commit()
    return True


def reorder_playlist_tracks(db: Session, user_id: int, playlist_id: int, ordered_track_ids: list[int]) -> dict | None:
    """
    Reorder playlist tracks to match the given ordered list of track_ids.
    All supplied track_ids must already be in the playlist.
    """
    p = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user_id).first()
    if not p:
        return None

    items = {item.track_id: item for item in p.items}
    # Validate all requested IDs are present
    for tid in ordered_track_ids:
        if tid not in items:
            return {"error": f"Track {tid} is not in this playlist."}

    # Temporarily set positions to very high values to avoid UNIQUE constraint conflicts
    for i, tid in enumerate(ordered_track_ids):
        items[tid].position = 10000 + i
    db.flush()

    for i, tid in enumerate(ordered_track_ids):
        items[tid].position = i

    p.updated_at = utcnow()
    db.commit()
    return _playlist_detail(p)


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISERS
# ─────────────────────────────────────────────────────────────────────────────

def _playlist_summary(p: Playlist) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description or "",
        "playlist_type": p.playlist_type,
        "track_count": len(p.items),
        "is_public": p.is_public,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        # Cover art: first track with an image
        "cover_image": next(
            (item.track.image_url for item in p.items if item.track and item.track.image_url),
            None,
        ),
    }


def _track_item_dict(item: PlaylistItem) -> dict:
    t = item.track
    if not t:
        return {"position": item.position, "track_id": item.track_id}
    return {
        "position": item.position,
        "playlist_item_id": item.id,
        "id": t.id,
        "spotify_id": t.spotify_id or "",
        "title": t.title,
        "artist": t.artist,
        "album": t.album or "",
        "image_url": t.image_url or "",
        "preview_url": t.preview_url or "",
        "duration_ms": t.duration_ms or 0,
        "has_features": all(getattr(t, f) is not None for f in AUDIO_FEATURES),
        "features": {f: round(float(getattr(t, f)), 4) for f in AUDIO_FEATURES if getattr(t, f) is not None},
    }


def _playlist_detail(p: Playlist) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description or "",
        "playlist_type": p.playlist_type,
        "is_public": p.is_public,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        "track_count": len(p.items),
        "cover_image": next(
            (item.track.image_url for item in p.items if item.track and item.track.image_url),
            None,
        ),
        "tracks": [_track_item_dict(item) for item in sorted(p.items, key=lambda x: x.position)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PERSIST GENERATED PLAYLIST TO DB
# ─────────────────────────────────────────────────────────────────────────────

def persist_generated_playlist(
    db: Session,
    user_id: int,
    name: str,
    playlist_type: str,
    description: str,
    tracks: list[dict],
) -> dict:
    """
    Save a generated playlist and its tracks to the database.
    Only saves library tracks that have a valid DB Track.id.
    Discovery-only tracks without a DB Track record are noted but not persisted.
    Returns the full playlist detail.
    """
    p = Playlist(
        user_id=user_id,
        name=name[:255],
        description=description,
        playlist_type=playlist_type,
    )
    db.add(p)
    db.flush()

    position = 0
    skipped = 0
    for item in tracks:
        track_id = item.get("id")
        if not track_id or item.get("source") == "discovery":
            # Discovery tracks (no DB Track.id) — skip for now (Phase 7 will handle)
            skipped += 1
            continue
        # Verify the track actually exists
        exists = db.query(Track.id).filter(Track.id == track_id).first()
        if not exists:
            skipped += 1
            continue
        # Dedup within this playlist
        already = db.query(PlaylistItem).filter(
            PlaylistItem.playlist_id == p.id,
            PlaylistItem.track_id == track_id,
        ).first()
        if already:
            continue
        db.add(PlaylistItem(playlist_id=p.id, track_id=track_id, position=position))
        position += 1

    db.commit()
    db.refresh(p)
    result = _playlist_detail(p)
    result["skipped_tracks"] = skipped
    return result

