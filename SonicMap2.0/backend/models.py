import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship

from database import Base


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String(128), unique=True, index=True, nullable=False)
    display_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    product = Column(String(64), nullable=True)  # premium | free | open
    avatar_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    # Relationships
    token = relationship("UserToken", back_populates="user", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    library_tracks = relationship("UserLibraryTrack", back_populates="user", cascade="all, delete-orphan")
    playlists = relationship("Playlist", back_populates="user", cascade="all, delete-orphan")
    model_metadata = relationship("PersonalModelMetadata", back_populates="user", cascade="all, delete-orphan")


class UserToken(Base):
    __tablename__ = "user_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    expires_at = Column(Float, nullable=False)  # Unix epoch timestamp
    scope = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="token")


class UserSession(Base):
    """Stores active browser sessions keyed by a secure random session token."""
    __tablename__ = "user_sessions"

    session_id = Column(String(128), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    expires_at = Column(Float, nullable=False)  # Unix epoch timestamp
    created_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")


class Track(Base):
    """Canonical track metadata and standard audio feature values."""
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String(128), unique=True, index=True, nullable=True)
    title = Column(String(512), nullable=False, index=True)
    artist = Column(String(512), nullable=False, index=True)
    album = Column(String(512), nullable=True)
    preview_url = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # 9 core audio features
    danceability = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    loudness = Column(Float, nullable=True)
    speechiness = Column(Float, nullable=True)
    acousticness = Column(Float, nullable=True)
    instrumentalness = Column(Float, nullable=True)
    liveness = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    # Relationships
    provenance = relationship("AudioFeatureProvenance", back_populates="track", cascade="all, delete-orphan")
    library_associations = relationship("UserLibraryTrack", back_populates="track", cascade="all, delete-orphan")
    playlist_items = relationship("PlaylistItem", back_populates="track", cascade="all, delete-orphan")


class AudioFeatureProvenance(Base):
    """Tracks origin, reliability, and confidence of audio features for each track."""
    __tablename__ = "audio_feature_provenance"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name = Column(String(64), nullable=False)  # 'all_9', 'tempo', 'danceability', etc.
    source = Column(String(128), nullable=False)  # 'reference_catalog_spotify_id', 'reference_catalog_metadata', 'librosa', 'ml_estimation'
    acquisition_type = Column(String(64), default="matched", nullable=False)  # 'matched', 'measured', 'estimated'
    match_confidence = Column(Float, default=1.0, nullable=False)  # 0.0 to 1.0 (accuracy of identity match)
    feature_confidence = Column(Float, default=1.0, nullable=False)  # 0.0 to 1.0 (accuracy of feature extraction/estimation)
    confidence = Column(Float, default=1.0, nullable=False)  # Overall combined confidence score
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    track = relationship("Track", back_populates="provenance")


class UserLibraryTrack(Base):
    """User's personal library items (Liked songs, playlist tracks, top tracks, recently played)."""
    __tablename__ = "user_library_tracks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    source = Column(String(64), default="liked", nullable=False)  # 'liked', 'playlist', 'top_short', 'recently_played'
    added_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="library_tracks")
    track = relationship("Track", back_populates="library_associations")

    __table_args__ = (
        UniqueConstraint("user_id", "track_id", "source", name="uq_user_track_source"),
    )


class Playlist(Base):
    """Playlists generated or saved in SonicMap."""
    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    playlist_type = Column(String(64), default="custom", nullable=False)  # 'cluster', 'similar', 'mood', 'energy', 'discovery', 'custom'
    is_public = Column(Boolean, default=False, nullable=False)
    spotify_playlist_id = Column(String(128), nullable=True)  # If exported or synced with Spotify
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="playlists")
    items = relationship("PlaylistItem", back_populates="playlist", cascade="all, delete-orphan", order_by="PlaylistItem.position")


class PlaylistItem(Base):
    """Ordered tracks within a playlist."""
    __tablename__ = "playlist_items"

    id = Column(Integer, primary_key=True, index=True)
    playlist_id = Column(Integer, ForeignKey("playlists.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    position = Column(Integer, nullable=False, default=0)
    added_at = Column(DateTime, default=utcnow, nullable=False)

    playlist = relationship("Playlist", back_populates="items")
    track = relationship("Track", back_populates="playlist_items")

    __table_args__ = (
        UniqueConstraint("playlist_id", "position", name="uq_playlist_position"),
    )


class PersonalModelMetadata(Base):
    """Tracks metadata and diagnostics for a user's personalized KMeans and PCA models."""
    __tablename__ = "personal_model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    cluster_count = Column(Integer, nullable=False, default=4)
    silhouette_score = Column(Float, nullable=True)
    track_count = Column(Integer, nullable=False, default=0)
    variance_ratio_pca1 = Column(Float, nullable=True)
    variance_ratio_pca2 = Column(Float, nullable=True)
    model_version = Column(String(32), default="v1.0", nullable=False)
    last_trained_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="model_metadata")
