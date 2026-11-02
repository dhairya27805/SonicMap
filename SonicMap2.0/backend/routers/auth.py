import base64
import hashlib
import secrets
import time
import urllib.parse

from fastapi import APIRouter, Cookie, HTTPException, Response
from fastapi.responses import RedirectResponse

from services import spotify_service
from state import (
    FRONTEND_URL,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    SESSIONS,
    SPOTIFY_AUTH_URL,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_REDIRECT_URI,
    SPOTIFY_SCOPES,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])

# Short-lived store for PKCE code_verifier + CSRF state, keyed by "state".
# Entries expire after 10 minutes and are removed once consumed.
_PKCE_STORE = {}
_PKCE_TTL_SECONDS = 600


def _cleanup_pkce_store():
    now = time.time()
    expired = [key for key, value in _PKCE_STORE.items() if now - value["created"] > _PKCE_TTL_SECONDS]
    for key in expired:
        _PKCE_STORE.pop(key, None)


def _generate_pkce_pair():
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("utf-8").rstrip("=")
    challenge_digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(challenge_digest).decode("utf-8").rstrip("=")
    return verifier, challenge


from database import SessionLocal
from models import User, UserToken, UserSession


def get_session(session_id: str | None):
    if not session_id:
        return None

    # First check DB
    try:
        with SessionLocal() as db:
            session_record = (
                db.query(UserSession)
                .filter(UserSession.session_id == session_id)
                .first()
            )
            if session_record:
                # If session has expired, clean up and return None
                if time.time() > session_record.expires_at:
                    db.delete(session_record)
                    db.commit()
                    return None

                user = session_record.user
                token = user.token if user else None
                if user and token:
                    return {
                        "access_token": token.access_token,
                        "refresh_token": token.refresh_token,
                        "expires_at": token.expires_at,
                        "scope": token.scope,
                        "user": {
                            "id": user.spotify_id,
                            "db_id": user.id,
                            "display_name": user.display_name,
                            "email": user.email,
                            "product": user.product,
                            "avatar": user.avatar_url,
                        },
                    }
    except Exception:
        pass

    # Fallback to in-memory SESSIONS if DB lookup fails or was not present
    return SESSIONS.get(session_id)


async def get_valid_access_token(session_id: str | None):
    """Return a valid (non-expired) access token for this session, refreshing if needed."""
    session = get_session(session_id)

    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated with Spotify.")

    if time.time() >= session["expires_at"]:
        try:
            refreshed = await spotify_service.refresh_access_token(session["refresh_token"])
        except spotify_service.SpotifyAuthError as exc:
            # Clean up session
            try:
                with SessionLocal() as db:
                    db.query(UserSession).filter(UserSession.session_id == session_id).delete()
                    db.commit()
            except Exception:
                pass
            SESSIONS.pop(session_id, None)
            raise HTTPException(
                status_code=401, detail=f"Spotify session expired: {exc}"
            ) from exc

        # Update in DB
        try:
            with SessionLocal() as db:
                session_record = (
                    db.query(UserSession)
                    .filter(UserSession.session_id == session_id)
                    .first()
                )
                if session_record and session_record.user and session_record.user.token:
                    user_token = session_record.user.token
                    user_token.access_token = refreshed["access_token"]
                    user_token.expires_at = refreshed["expires_at"]
                    if "refresh_token" in refreshed:
                        user_token.refresh_token = refreshed["refresh_token"]
                    db.commit()
        except Exception:
            pass

        session.update(refreshed)
        SESSIONS[session_id] = session

    return session["access_token"]


@router.get("/login")
def login():
    if not SPOTIFY_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail=(
                "SPOTIFY_CLIENT_ID is not configured on the server. "
                "Set it in backend/.env (see README setup instructions)."
            ),
        )

    _cleanup_pkce_store()

    verifier, challenge = _generate_pkce_pair()
    state_token = secrets.token_urlsafe(24)

    _PKCE_STORE[state_token] = {"verifier": verifier, "created": time.time()}

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "state": state_token,
        "scope": SPOTIFY_SCOPES,
    }

    url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url)


@router.get("/callback")
async def callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/?spotify_error={urllib.parse.quote(error)}")

    if not code or not state or state not in _PKCE_STORE:
        return RedirectResponse(f"{FRONTEND_URL}/?spotify_error=invalid_state")

    verifier = _PKCE_STORE.pop(state)["verifier"]

    try:
        token_payload = await spotify_service.exchange_code_for_token(code, verifier)
        profile = await spotify_service.get_current_user(token_payload["access_token"])
    except spotify_service.SpotifyAuthError as exc:
        return RedirectResponse(
            f"{FRONTEND_URL}/?spotify_error={urllib.parse.quote(str(exc))}"
        )

    session_id = secrets.token_urlsafe(32)
    user_info = {
        "id": profile.get("id"),
        "display_name": profile.get("display_name") or profile.get("id"),
        "email": profile.get("email"),
        "product": profile.get("product"),  # "premium" | "free" | "open"
        "avatar": (profile.get("images") or [{}])[0].get("url"),
    }

    # Persist user, token, and session to database
    try:
        with SessionLocal() as db:
            user = db.query(User).filter(User.spotify_id == user_info["id"]).first()
            if not user:
                user = User(
                    spotify_id=user_info["id"],
                    display_name=user_info["display_name"],
                    email=user_info["email"],
                    product=user_info["product"],
                    avatar_url=user_info["avatar"],
                )
                db.add(user)
                db.flush()
            else:
                user.display_name = user_info["display_name"]
                user.email = user_info["email"]
                user.product = user_info["product"]
                user.avatar_url = user_info["avatar"]

            # Upsert token
            token_record = db.query(UserToken).filter(UserToken.user_id == user.id).first()
            if not token_record:
                token_record = UserToken(
                    user_id=user.id,
                    access_token=token_payload["access_token"],
                    refresh_token=token_payload["refresh_token"],
                    expires_at=token_payload["expires_at"],
                    scope=token_payload.get("scope"),
                )
                db.add(token_record)
            else:
                token_record.access_token = token_payload["access_token"]
                token_record.refresh_token = token_payload["refresh_token"]
                token_record.expires_at = token_payload["expires_at"]
                if "scope" in token_payload:
                    token_record.scope = token_payload["scope"]

            # Create session
            session_expiry = time.time() + SESSION_MAX_AGE_SECONDS
            user_session_record = UserSession(
                session_id=session_id,
                user_id=user.id,
                expires_at=session_expiry,
            )
            db.add(user_session_record)
            db.commit()
    except Exception as exc:
        print(f"Warning: Failed to persist session to database: {exc}")

    # Also keep in in-memory SESSIONS for fast fallback
    SESSIONS[session_id] = {
        **token_payload,
        "user": user_info,
    }

    response = RedirectResponse(f"{FRONTEND_URL}/?connected=1")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        # secure=True should be enabled once served over HTTPS in production.
    )
    return response


@router.get("/me")
async def me(sonicmap_session: str | None = Cookie(default=None)):
    session = get_session(sonicmap_session)

    if session is None:
        return {"authenticated": False}

    return {"authenticated": True, "user": session["user"]}


@router.post("/logout")
def logout(response: Response, sonicmap_session: str | None = Cookie(default=None)):
    if sonicmap_session:
        try:
            with SessionLocal() as db:
                db.query(UserSession).filter(UserSession.session_id == sonicmap_session).delete()
                db.commit()
        except Exception:
            pass
        SESSIONS.pop(sonicmap_session, None)

    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"message": "Logged out."}
