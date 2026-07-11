"""
profile.py
==========
Renders the logged-in user's Profile page: basic account details plus
usage statistics (total searches, total saved papers) pulled from
MongoDB via database.py.
"""

import logging

import streamlit as st

from src import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def render_profile_page(user: dict) -> None:
    """
    Render the "Profile" Streamlit page for the currently logged-in user.

    Parameters
    ----------
    user : dict
        The session-state user dict populated by auth.login_user():
        {'user_id', 'name', 'email', 'created_at'}.
    """
    st.subheader("👤 Profile")

    user_id = user.get("user_id")
    created_at = user.get("created_at")
    created_str = created_at.strftime("%B %d, %Y") if created_at else "Unknown"

    total_searches = database.count_search_history(user_id)
    total_saved = database.count_bookmarks(user_id)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div style='width:110px;height:110px;border-radius:50%;background:#1C7293;"
            f"display:flex;align-items:center;justify-content:center;font-size:42px;color:white;'>"
            f"{user.get('name', '?')[:1].upper()}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"### {user.get('name', 'Unknown')}")
        st.write(f"📧 {user.get('email', 'Unknown')}")
        st.write(f"📅 Account created: {created_str}")

    st.markdown("---")
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Total Searches", total_searches)
    stat_col2.metric("Total Saved Papers", total_saved)
