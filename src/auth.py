"""
auth.py
=======
User authentication for the COVID-19 Information Retrieval system:
registration, login, logout, and Streamlit session-state management.

Passwords are never stored or compared in plaintext — bcrypt is used to
hash passwords at registration time and to verify them at login time.
"""

import logging
import re

import streamlit as st

from src import config, database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------
def is_valid_email(email: str) -> bool:
    """
    Validate an email address's format.

    Parameters
    ----------
    email : str

    Returns
    -------
    bool
    """
    return bool(email) and bool(_EMAIL_RE.match(email.strip()))


def is_valid_password(password: str) -> tuple:
    """
    Validate password strength.

    Parameters
    ----------
    password : str

    Returns
    -------
    tuple(bool, str)
        (is_valid, message). message is empty when is_valid is True.
    """
    if not password or len(password) < config.MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters long."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    return True, ""


# ---------------------------------------------------------------------------
# PASSWORD HASHING
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Parameters
    ----------
    password : str

    Returns
    -------
    str
        The bcrypt hash, decoded to a UTF-8 string for storage in MongoDB.

    Raises
    ------
    RuntimeError
        If hashing fails.
    """
    try:
        import bcrypt

        salt = bcrypt.gensalt(rounds=config.BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to hash password: {exc}") from exc


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.

    Parameters
    ----------
    password : str
    hashed_password : str

    Returns
    -------
    bool
    """
    try:
        import bcrypt

        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Password verification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# REGISTRATION / LOGIN LOGIC
# ---------------------------------------------------------------------------
def register_user(name: str, email: str, password: str, confirm_password: str) -> tuple:
    """
    Validate input and register a new user account.

    Parameters
    ----------
    name : str
    email : str
    password : str
    confirm_password : str

    Returns
    -------
    tuple(bool, str)
        (success, message)
    """
    name = (name or "").strip()
    email = (email or "").strip()

    if not name:
        return False, "Please enter your full name."
    if not is_valid_email(email):
        return False, "Please enter a valid email address."
    if password != confirm_password:
        return False, "Passwords do not match."

    valid, msg = is_valid_password(password)
    if not valid:
        return False, msg

    try:
        if database.find_user_by_email(email):
            return False, "An account with this email already exists. Please log in instead."

        hashed = hash_password(password)
        database.insert_user(name, email, hashed)
        logger.info("New user registered: %s", email)
        return True, "Account created successfully! You can now log in."

    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"Registration failed: {exc}"


def login_user(email: str, password: str) -> tuple:
    """
    Validate credentials and, on success, populate Streamlit session state
    with the logged-in user's public profile fields.

    Parameters
    ----------
    email : str
    password : str

    Returns
    -------
    tuple(bool, str)
        (success, message)
    """
    email = (email or "").strip()

    if not is_valid_email(email) or not password:
        return False, "Please enter a valid email and password."

    try:
        user = database.find_user_by_email(email)
        if not user or not verify_password(password, user["password"]):
            return False, "Invalid email or password."

        st.session_state[config.SESSION_STATE_USER_KEY] = {
            "user_id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "created_at": user.get("created_at"),
        }
        st.session_state[config.SESSION_STATE_PAGE_KEY] = "dashboard"
        logger.info("User logged in: %s", email)
        return True, f"Welcome back, {user['name']}!"

    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"Login failed: {exc}"


def logout_user() -> None:
    """Clear the logged-in user's session state, effectively logging them out."""
    st.session_state.pop(config.SESSION_STATE_USER_KEY, None)
    st.session_state[config.SESSION_STATE_PAGE_KEY] = "auth"


def get_current_user() -> dict:
    """
    Return the currently logged-in user's session dict, or None if no one
    is logged in.

    Returns
    -------
    dict or None
    """
    return st.session_state.get(config.SESSION_STATE_USER_KEY)


def is_logged_in() -> bool:
    """Return True if a user is currently logged in."""
    return get_current_user() is not None


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
def render_auth_page() -> None:
    """
    Render the Login / Register page. Should be called instead of the
    main dashboard whenever `is_logged_in()` is False.
    """
    st.markdown(
        f"<h1 style='text-align:center;'>{config.APP_ICON} {config.APP_TITLE}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#5B7280;'>"
        "Sign in to search, save, and get AI-assisted insights on COVID-19 research papers."
        "</p>",
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.4, 1])
    with center_col:
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            st.caption("No account? Register to create your account.")

            if submitted:
                success, message = login_user(email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with tab_register:
            with st.form("register_form"):
                name = st.text_input("Full Name", key="register_name")
                email_r = st.text_input("Email", key="register_email")
                password_r = st.text_input("Password", type="password", key="register_password")
                confirm_r = st.text_input("Confirm Password", type="password", key="register_confirm")
                st.caption(f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters, with a letter and a digit.")
                submitted_r = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submitted_r:
                success, message = register_user(name, email_r, password_r, confirm_r)
                if success:
                    st.success(message + " Switch to the Login tab to sign in.")
                else:
                    st.error(message)
