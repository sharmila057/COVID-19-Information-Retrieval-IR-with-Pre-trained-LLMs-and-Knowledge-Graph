"""
database.py
===========
MongoDB Atlas integration layer for the COVID-19 Information Retrieval
system's account features: user accounts, search history, and bookmarks.

This module is intentionally the *only* place that talks to MongoDB
directly — `auth.py`, `history.py`, `bookmark.py`, and `profile.py` all
go through the functions defined here, so the rest of the application
never constructs a query by hand.

Collections
-----------
users            : {_id, name, email, password (bcrypt hash), created_at}
search_history   : {_id, user_id, query, timestamp}
bookmarks        : {_id, user_id, paper_title, paper_id, saved_at}
"""

import datetime
import logging

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_client_cache = {}


def get_client():
    """
    Create (and cache) a MongoDB client connected to MongoDB Atlas using
    the URI from the `MONGO_URI` environment variable.

    Returns
    -------
    pymongo.MongoClient

    Raises
    ------
    RuntimeError
        If `MONGO_URI` is not configured or the connection cannot be
        established (e.g. wrong credentials, IP not allow-listed, no
        internet connection).
    """
    if "client" in _client_cache:
        return _client_cache["client"]

    if not config.MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is not set. Add it to your .env file — see .env.example "
            "and the README's 'MongoDB Atlas Setup' section."
        )

    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        client = MongoClient(config.MONGO_URI, server_api=ServerApi("1"), serverSelectionTimeoutMS=8000)
        # Fail fast with a clear error instead of hanging if the URI/credentials are wrong.
        client.admin.command("ping")
        _client_cache["client"] = client
        logger.info("Connected to MongoDB Atlas successfully.")
        return client

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to connect to MongoDB Atlas: {exc}") from exc


def get_db():
    """
    Return the application's MongoDB database handle.

    Returns
    -------
    pymongo.database.Database
    """
    client = get_client()
    return client[config.MONGO_DB_NAME]


def get_collection(name: str):
    """
    Return a specific MongoDB collection by name.

    Parameters
    ----------
    name : str

    Returns
    -------
    pymongo.collection.Collection
    """
    return get_db()[name]


def ensure_indexes():
    """
    Create the indexes required for correct/fast operation, if they don't
    already exist. Safe to call multiple times (idempotent).

    Raises
    ------
    RuntimeError
        If index creation fails.
    """
    try:
        users = get_collection(config.USERS_COLLECTION)
        users.create_index("email", unique=True)

        history = get_collection(config.SEARCH_HISTORY_COLLECTION)
        history.create_index("user_id")

        bookmarks = get_collection(config.BOOKMARKS_COLLECTION)
        bookmarks.create_index([("user_id", 1), ("paper_id", 1)], unique=True)

        logger.info("MongoDB indexes verified/created.")

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to create MongoDB indexes: {exc}") from exc


# ---------------------------------------------------------------------------
# USERS
# ---------------------------------------------------------------------------
def find_user_by_email(email: str) -> dict:
    """
    Look up a user document by (case-insensitive) email address.

    Parameters
    ----------
    email : str

    Returns
    -------
    dict or None
    """
    try:
        users = get_collection(config.USERS_COLLECTION)
        return users.find_one({"email": email.strip().lower()})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to query user by email: {exc}") from exc


def find_user_by_id(user_id) -> dict:
    """
    Look up a user document by its MongoDB ObjectId (or string form of it).

    Parameters
    ----------
    user_id : str or bson.ObjectId

    Returns
    -------
    dict or None
    """
    try:
        from bson import ObjectId

        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        users = get_collection(config.USERS_COLLECTION)
        return users.find_one({"_id": user_id})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to query user by id: {exc}") from exc


def insert_user(name: str, email: str, hashed_password: str) -> str:
    """
    Insert a new user document.

    Parameters
    ----------
    name : str
    email : str
    hashed_password : str
        A bcrypt hash — NEVER the plaintext password.

    Returns
    -------
    str
        The newly created user's _id as a string.

    Raises
    ------
    RuntimeError
        If insertion fails (e.g. duplicate email, connection error).
    """
    try:
        users = get_collection(config.USERS_COLLECTION)
        doc = {
            "name": name.strip(),
            "email": email.strip().lower(),
            "password": hashed_password,
            "created_at": datetime.datetime.utcnow(),
        }
        result = users.insert_one(doc)
        return str(result.inserted_id)

    except Exception as exc:  # noqa: BLE001
        from pymongo.errors import DuplicateKeyError

        if isinstance(exc, DuplicateKeyError):
            raise RuntimeError("An account with this email already exists.") from exc
        raise RuntimeError(f"Failed to create user account: {exc}") from exc


# ---------------------------------------------------------------------------
# SEARCH HISTORY
# ---------------------------------------------------------------------------
def add_search_history(user_id: str, query: str) -> None:
    """
    Record a search query for a user.

    Parameters
    ----------
    user_id : str
    query : str
    """
    try:
        history = get_collection(config.SEARCH_HISTORY_COLLECTION)
        history.insert_one({
            "user_id": str(user_id),
            "query": query,
            "timestamp": datetime.datetime.utcnow(),
        })
    except Exception as exc:  # noqa: BLE001
        # Search history is a convenience feature — never let a logging
        # failure break the actual retrieval the user asked for.
        logger.warning("Failed to save search history: %s", exc)


def get_search_history(user_id: str, limit: int = 100) -> list:
    """
    Fetch a user's search history, most recent first.

    Parameters
    ----------
    user_id : str
    limit : int

    Returns
    -------
    list[dict]
    """
    try:
        history = get_collection(config.SEARCH_HISTORY_COLLECTION)
        cursor = history.find({"user_id": str(user_id)}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to fetch search history: {exc}") from exc


def delete_search_history_item(history_id: str) -> None:
    """Delete a single search-history entry by its _id."""
    try:
        from bson import ObjectId

        history = get_collection(config.SEARCH_HISTORY_COLLECTION)
        history.delete_one({"_id": ObjectId(history_id)})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to delete search history item: {exc}") from exc


def clear_search_history(user_id: str) -> None:
    """Delete all search-history entries for a user."""
    try:
        history = get_collection(config.SEARCH_HISTORY_COLLECTION)
        history.delete_many({"user_id": str(user_id)})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to clear search history: {exc}") from exc


def count_search_history(user_id: str) -> int:
    """Return the total number of searches a user has made."""
    try:
        history = get_collection(config.SEARCH_HISTORY_COLLECTION)
        return history.count_documents({"user_id": str(user_id)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to count search history: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# BOOKMARKS
# ---------------------------------------------------------------------------
def add_bookmark(user_id: str, paper_id: str, paper_title: str) -> None:
    """
    Save a paper as a bookmark for a user. Silently no-ops if it is
    already bookmarked (enforced by a unique index on (user_id, paper_id)).

    Parameters
    ----------
    user_id : str
    paper_id : str
    paper_title : str
    """
    try:
        from pymongo.errors import DuplicateKeyError

        bookmarks = get_collection(config.BOOKMARKS_COLLECTION)
        try:
            bookmarks.insert_one({
                "user_id": str(user_id),
                "paper_id": str(paper_id),
                "paper_title": paper_title,
                "saved_at": datetime.datetime.utcnow(),
            })
        except DuplicateKeyError:
            pass  # already bookmarked — treat as success

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to save bookmark: {exc}") from exc


def remove_bookmark(user_id: str, paper_id: str) -> None:
    """Remove a bookmark for a user/paper pair."""
    try:
        bookmarks = get_collection(config.BOOKMARKS_COLLECTION)
        bookmarks.delete_one({"user_id": str(user_id), "paper_id": str(paper_id)})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to remove bookmark: {exc}") from exc


def get_bookmarks(user_id: str) -> list:
    """
    Fetch all of a user's bookmarked papers, most recently saved first.

    Parameters
    ----------
    user_id : str

    Returns
    -------
    list[dict]
    """
    try:
        bookmarks = get_collection(config.BOOKMARKS_COLLECTION)
        cursor = bookmarks.find({"user_id": str(user_id)}).sort("saved_at", -1)
        return list(cursor)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to fetch bookmarks: {exc}") from exc


def is_bookmarked(user_id: str, paper_id: str) -> bool:
    """Return True if a paper is already bookmarked by the user."""
    try:
        bookmarks = get_collection(config.BOOKMARKS_COLLECTION)
        return bookmarks.find_one({"user_id": str(user_id), "paper_id": str(paper_id)}) is not None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to check bookmark status: %s", exc)
        return False


def count_bookmarks(user_id: str) -> int:
    """Return the total number of papers a user has bookmarked."""
    try:
        bookmarks = get_collection(config.BOOKMARKS_COLLECTION)
        return bookmarks.count_documents({"user_id": str(user_id)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to count bookmarks: %s", exc)
        return 0
