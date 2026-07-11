"""
history.py
==========
Search-history tracking: automatically records every search a logged-in
user runs, and provides a Streamlit page to view, delete individual
entries, or clear all history.
"""

import logging

import streamlit as st

from src import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def record_search(user_id: str, query: str) -> None:
    """
    Save a search query to the user's history. Failures are logged but
    never raised, so a history-logging problem can never break an
    in-progress search.

    Parameters
    ----------
    user_id : str
    query : str
    """
    database.add_search_history(user_id, query)


def render_history_page(user_id: str) -> None:
    """
    Render the "Search History" Streamlit page: a reverse-chronological
    list of past queries with per-item delete buttons and a clear-all
    button.

    Parameters
    ----------
    user_id : str
    """
    st.subheader("📜 Search History")

    try:
        history_items = database.get_search_history(user_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load search history: {exc}")
        return

    if not history_items:
        st.info("You haven't searched anything yet. Head to **Search Research Papers** to get started.")
        return

    col_a, col_b = st.columns([4, 1])
    with col_b:
        if st.button("🗑️ Clear All", use_container_width=True):
            try:
                database.clear_search_history(user_id)
                st.success("Search history cleared.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to clear history: {exc}")

    for item in history_items:
        item_id = str(item["_id"])
        query_text = item.get("query", "")
        timestamp = item.get("timestamp")
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M") if timestamp else "Unknown time"

        row_col1, row_col2, row_col3 = st.columns([5, 2, 1])
        with row_col1:
            st.markdown(f"🔍 **{query_text}**")
        with row_col2:
            st.caption(ts_str)
        with row_col3:
            if st.button("Delete", key=f"del_hist_{item_id}"):
                try:
                    database.delete_search_history_item(item_id)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to delete entry: {exc}")
        st.divider()
