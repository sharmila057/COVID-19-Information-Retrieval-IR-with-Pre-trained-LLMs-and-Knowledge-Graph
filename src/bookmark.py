"""
bookmark.py
===========
Lets a logged-in user save ("bookmark") papers from their search results
and manage those saved papers on a dedicated "Saved Papers" page.
"""

import logging

import streamlit as st

from src import database
from src.utils import truncate_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def toggle_bookmark(user_id: str, paper_id: str, paper_title: str) -> bool:
    """
    Bookmark a paper if it isn't already saved, or remove it if it is
    (a toggle operation driven by the ⭐ button next to each result).

    Parameters
    ----------
    user_id : str
    paper_id : str
    paper_title : str

    Returns
    -------
    bool
        The paper's new bookmark state (True = now bookmarked).
    """
    try:
        if database.is_bookmarked(user_id, paper_id):
            database.remove_bookmark(user_id, paper_id)
            return False
        else:
            database.add_bookmark(user_id, paper_id, paper_title)
            return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to update bookmark: {exc}")
        return database.is_bookmarked(user_id, paper_id)


def render_bookmark_button(user_id: str, paper_id: str, paper_title: str, key_suffix: str) -> None:
    """
    Render a ⭐ Save / ✅ Saved toggle button for a single search result.
    Intended to be called inline from app.py's results renderer, so it
    does not change the layout of the existing results table.

    Parameters
    ----------
    user_id : str
    paper_id : str
    paper_title : str
    key_suffix : str
        A unique suffix (e.g. the paper's rank or doc_id) to keep the
        Streamlit widget key unique across multiple results on the page.
    """
    try:
        already_saved = database.is_bookmarked(user_id, paper_id)
    except Exception:  # noqa: BLE001
        already_saved = False

    label = "✅ Saved" if already_saved else "⭐ Save"
    if st.button(label, key=f"bookmark_{key_suffix}"):
        new_state = toggle_bookmark(user_id, paper_id, paper_title)
        st.toast("Saved to Bookmarks!" if new_state else "Removed from Bookmarks.")
        st.rerun()


def render_bookmarks_page(user_id: str) -> None:
    """
    Render the "Saved Papers" (Bookmarks) Streamlit page: a list of all
    papers the user has bookmarked, each with a Remove button.

    Parameters
    ----------
    user_id : str
    """
    st.subheader("⭐ Saved Papers")

    try:
        bookmarks = database.get_bookmarks(user_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load bookmarks: {exc}")
        return

    if not bookmarks:
        st.info("You haven't saved any papers yet. Click **⭐ Save** on a search result to bookmark it.")
        return

    for item in bookmarks:
        paper_id = item.get("paper_id", "")
        title = item.get("paper_title", "Untitled")
        saved_at = item.get("saved_at")
        saved_str = saved_at.strftime("%Y-%m-%d %H:%M") if saved_at else ""

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{truncate_text(title, 150)}**")
            st.caption(f"Document ID: `{paper_id}`  •  Saved: {saved_str}")
        with col2:
            if st.button("🗑️ Remove", key=f"remove_bm_{paper_id}"):
                try:
                    database.remove_bookmark(user_id, paper_id)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to remove bookmark: {exc}")
        st.divider()
