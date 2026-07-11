#!/usr/bin/env python3
"""
app.py
======
Streamlit web interface for the COVID-19 Information Retrieval system.

This file now wraps the ORIGINAL retrieval pipeline (unchanged) with:
    - User authentication (login / register / logout) backed by MongoDB Atlas
    - A modern dashboard with sidebar + card navigation
    - Automatic search-history logging
    - Per-result bookmarking ("Saved Papers")
    - A Gemini-powered AI Assistant that operates strictly on retrieved
      papers (Retrieval-Augmented Generation), never on its own knowledge
    - A user Profile page with account stats

Core retrieval pipeline per search query (UNCHANGED from the original app):
    1. Accept a free-text query.
    2. Dense retrieval: FAISS + SentenceTransformer -> Top-K candidates.
    3. Re-ranking: CrossEncoder -> Top-N results.
    4. Display Top-N papers with title, abstract snippet, and scores.
    5. Extract entities (spaCy) from the Top-N results.
    6. Build & visualize a knowledge graph (NetworkX) from those entities.

Run with:
    streamlit run app.py
"""

import logging

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_or_download
from src.entity_extractor import extract_entities_from_documents
from src.knowledge_graph import build_knowledge_graph, get_graph_statistics, visualize_knowledge_graph
from src.preprocessing import preprocess_corpus
from src.reranker import rerank
from src.retriever import DenseRetriever
from src.utils import format_score, truncate_text

# New feature modules (additive — none of the imports above were touched)
from src import auth, bookmark, gemini_service, history, profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")


# --------------------------------------------------------------------------
# CACHED RESOURCE LOADERS (UNCHANGED FROM THE ORIGINAL APP)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_corpus() -> pd.DataFrame:
    """Load and preprocess the corpus once per Streamlit session (cached)."""
    corpus_df, _, _ = load_or_download()
    return preprocess_corpus(corpus_df)


@st.cache_resource(show_spinner=False)
def _load_retriever(_corpus_df: pd.DataFrame) -> DenseRetriever:
    """Instantiate the DenseRetriever once and cache it across reruns."""
    return DenseRetriever(_corpus_df)


def _init_app_state():
    """Load the corpus and retriever, surfacing friendly errors on failure."""
    try:
        with st.spinner("Loading corpus ..."):
            corpus_df = _load_corpus()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load the TREC-COVID dataset: {exc}")
        st.stop()

    try:
        with st.spinner("Loading FAISS index and models ..."):
            retriever = _load_retriever(corpus_df)
    except FileNotFoundError:
        st.error(
            "FAISS index not found. Please run `python build_index.py` from the "
            "project root before launching the app."
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialise the retriever: {exc}")
        st.stop()

    return corpus_df, retriever


# --------------------------------------------------------------------------
# THEME / STYLING
# --------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject a professional 'Clinical Deep' navy/teal/coral colour theme."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #FFFFFF; }
        div[data-testid="stSidebar"] {
            background-color: #0B3D5C;
        }
        div[data-testid="stSidebar"] * { color: #EAF2F6 !important; }
        div[data-testid="stSidebar"] button {
            background-color: #1C7293 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        .dash-card {
            background-color: #EAF2F6;
            border-radius: 12px;
            padding: 22px 18px;
            text-align: center;
            border: 1px solid #DCE4E8;
        }
        .footer {
            text-align: center;
            color: #5B7280;
            font-size: 0.8rem;
            padding-top: 12px;
            border-top: 1px solid #DCE4E8;
            margin-top: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render a consistent footer with project information on every page."""
    st.markdown(
        f"<div class='footer'>{config.APP_ICON} {config.APP_TITLE} &nbsp;|&nbsp; "
        "Dense Retrieval + Cross-Encoder Re-ranking + Knowledge Graph + Gemini RAG Assistant "
        "&nbsp;|&nbsp; Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# SEARCH PAGE (core retrieval pipeline — logic UNCHANGED, presentation extended)
# --------------------------------------------------------------------------
def render_sidebar_settings():
    """Render sidebar retrieval controls and return user-configurable parameters."""
    st.sidebar.markdown("### ⚙️ Search Settings")
    top_k = st.sidebar.slider("FAISS candidates (Top-K)", min_value=10, max_value=100,
                               value=config.TOP_K_RETRIEVE, step=10)
    top_n = st.sidebar.slider("Final re-ranked results (Top-N)", min_value=5, max_value=20,
                               value=config.TOP_N_RERANK, step=1)
    show_kg = st.sidebar.checkbox("Show Knowledge Graph", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Models used**\n"
        f"- Bi-Encoder: `{config.BI_ENCODER_MODEL}`\n"
        f"- Cross-Encoder: `{config.CROSS_ENCODER_MODEL}`\n"
        f"- NER: `{config.SPACY_MODEL}`"
    )
    return top_k, top_n, show_kg


def render_results_table(ranked_df: pd.DataFrame, user_id: str) -> None:
    """
    Display the re-ranked Top-N results as expandable cards, each with a
    bookmark toggle and an optional Gemini "Summarize" action.
    """
    for rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
        with st.expander(f"#{rank}  {row['title']}", expanded=(rank <= 3)):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Abstract:** {truncate_text(row['text'], 600)}")
                st.caption(f"Document ID: `{row['doc_id']}`")

                if gemini_service.is_configured():
                    if st.button("🤖 Summarize with Gemini", key=f"summarize_{row['doc_id']}"):
                        try:
                            with st.spinner("Asking Gemini for a summary ..."):
                                summary = gemini_service.summarize_paper(row["title"], row["text"])
                            st.info(summary)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Gemini summarization failed: {exc}")
            with col2:
                st.metric("Cross-Encoder Score", format_score(row["rerank_score"]))
                st.metric("Dense (FAISS) Score", format_score(row["dense_score"]))
                bookmark.render_bookmark_button(user_id, row["doc_id"], row["title"], key_suffix=row["doc_id"])


def render_knowledge_graph_section(ranked_df: pd.DataFrame) -> None:
    """Extract entities from the Top-N results and render the knowledge graph. (Unchanged logic.)"""
    st.subheader("🕸️ Knowledge Graph of Extracted Entities")

    documents = ranked_df.to_dict("records")
    try:
        with st.spinner("Extracting entities with spaCy ..."):
            doc_entities = extract_entities_from_documents(documents)

        if not any(doc_entities.values()):
            st.info("No recognisable entities were found in the top results.")
            return

        graph = build_knowledge_graph(documents, doc_entities)
        fig = visualize_knowledge_graph(graph, save_path=config.KG_IMAGE_PATH)
        st.pyplot(fig)

        stats = get_graph_statistics(graph)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Nodes", stats["num_nodes"])
        col2.metric("Total Edges", stats["num_edges"])
        col3.metric("Entities Found", stats["num_entities"])

        st.markdown("**Entity type breakdown:**")
        st.json(stats["entity_type_counts"])

        st.markdown("**Most connected entities:**")
        st.table(pd.DataFrame(stats["most_connected_entities"]))

    except ValueError as exc:
        st.info(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Knowledge graph generation failed: {exc}")


def render_search_page(user: dict, corpus_df: pd.DataFrame, retriever: DenseRetriever) -> None:
    """Render the main search page: the original retrieval pipeline, extended with
    history logging and bookmarking."""
    st.title(f"{config.APP_ICON} Search Research Papers")
    st.markdown(
        "Search scientific literature on COVID-19 using **dense retrieval** "
        "(SentenceTransformer + FAISS) and **cross-encoder re-ranking**, "
        "with automatic entity extraction and knowledge-graph visualization."
    )

    top_k, top_n, show_kg = render_sidebar_settings()

    query = st.text_input(
        "🔍 Enter your COVID-19 research query",
        placeholder="e.g. What are the effects of remdesivir on COVID-19 patients?",
    )
    search_clicked = st.button("Search", type="primary")

    if search_clicked and query.strip():
        progress = st.progress(0, text="Starting search ...")
        try:
            progress.progress(20, text="Retrieving candidate documents with FAISS ...")
            candidates_df = retriever.retrieve(query, top_k=top_k)

            if candidates_df.empty:
                progress.empty()
                st.warning("No documents matched your query. Please try a different query.")
                return

            progress.progress(60, text="Re-ranking with CrossEncoder ...")
            ranked_df = rerank(query, candidates_df, top_n=top_n)

            progress.progress(90, text="Preparing results ...")
            history.record_search(user["user_id"], query)
            st.session_state["last_search_query"] = query
            st.session_state["last_search_results"] = ranked_df

            progress.progress(100, text="Done!")
            progress.empty()

            st.success(f"Found {len(ranked_df)} highly relevant papers (from {len(candidates_df)} candidates).")
            render_results_table(ranked_df, user["user_id"])

            if show_kg:
                render_knowledge_graph_section(ranked_df)

        except ValueError as exc:
            progress.empty()
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            progress.empty()
            st.error(f"An unexpected error occurred while processing your query: {exc}")

    elif search_clicked:
        st.warning("Please enter a query before searching.")

    st.markdown("---")
    st.caption(
        f"Corpus size: {len(corpus_df):,} documents | "
        "Dataset: BEIR TREC-COVID | Built with Streamlit"
    )


# --------------------------------------------------------------------------
# AI ASSISTANT PAGE (Gemini RAG — operates only on already-retrieved papers)
# --------------------------------------------------------------------------
def render_ai_assistant_page() -> None:
    """
    Render the AI Assistant page. All actions here operate strictly on the
    papers already retrieved on the Search page (st.session_state
    ['last_search_results']) — Gemini is never used to search the corpus.
    """
    st.title("🤖 AI Assistant")

    if not gemini_service.is_configured():
        st.warning(
            "Gemini is not configured. Add `GEMINI_API_KEY` to your `.env` file to enable "
            "AI-assisted summaries, explanations, comparisons, and Q&A. See the README's "
            "'Gemini API Setup' section."
        )
        return

    ranked_df = st.session_state.get("last_search_results")
    last_query = st.session_state.get("last_search_query", "")

    if ranked_df is None or ranked_df.empty:
        st.info("Run a search on **Search Research Papers** first — the AI Assistant works only on your retrieved results.")
        return

    st.caption(f"Using your {len(ranked_df)} most recent results for query: \u201C{last_query}\u201D")
    papers = ranked_df.to_dict("records")
    titles = [f"{i+1}. {p['title']}" for i, p in enumerate(papers)]

    tab_findings, tab_compare, tab_terms, tab_qa = st.tabs(
        ["📌 Key Findings", "⚖️ Compare Papers", "📖 Explain Terminology", "💬 Ask a Question"]
    )

    with tab_findings:
        if st.button("Generate Key Findings", key="gen_findings"):
            try:
                with st.spinner("Asking Gemini ..."):
                    result = gemini_service.generate_key_findings(papers)
                st.markdown(result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to generate key findings: {exc}")

    with tab_compare:
        col1, col2 = st.columns(2)
        idx_a = col1.selectbox("Paper A", options=range(len(titles)), format_func=lambda i: titles[i], key="cmp_a")
        idx_b = col2.selectbox("Paper B", options=range(len(titles)), format_func=lambda i: titles[i], key="cmp_b")
        if st.button("Compare", key="do_compare"):
            if idx_a == idx_b:
                st.warning("Please select two different papers to compare.")
            else:
                try:
                    with st.spinner("Asking Gemini ..."):
                        result = gemini_service.compare_papers(papers[idx_a], papers[idx_b])
                    st.markdown(result)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Comparison failed: {exc}")

    with tab_terms:
        idx = st.selectbox("Choose a paper", options=range(len(titles)), format_func=lambda i: titles[i], key="terms_idx")
        if st.button("Explain Terminology", key="do_explain"):
            try:
                with st.spinner("Asking Gemini ..."):
                    result = gemini_service.explain_terminology(papers[idx]["text"])
                st.markdown(result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Explanation failed: {exc}")

    with tab_qa:
        question = st.text_input("Ask a question about your retrieved papers", key="rag_question")
        if st.button("Ask", key="do_ask") and question.strip():
            try:
                with st.spinner("Asking Gemini (using only your retrieved papers as context) ..."):
                    answer = gemini_service.answer_question(question, papers)
                st.markdown(answer)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to answer question: {exc}")


# --------------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------------
_DASHBOARD_CARDS = [
    ("search", "🔍", "Search Research Papers"),
    ("history", "📜", "Search History"),
    ("bookmarks", "⭐", "Saved Papers"),
    ("assistant", "🤖", "AI Assistant"),
    ("profile", "👤", "Profile"),
]


def render_dashboard_home(user: dict) -> None:
    """Render the dashboard landing page with navigation cards."""
    st.title(f"{config.APP_ICON} Welcome, {user['name']}")
    st.markdown("Choose where you'd like to go:")

    cols = st.columns(len(_DASHBOARD_CARDS))
    for col, (page_key, icon, label) in zip(cols, _DASHBOARD_CARDS):
        with col:
            st.markdown(f"<div class='dash-card'><div style='font-size:32px;'>{icon}</div>{label}</div>", unsafe_allow_html=True)
            if st.button("Open", key=f"card_{page_key}", use_container_width=True):
                st.session_state[config.SESSION_STATE_PAGE_KEY] = page_key
                st.rerun()


def render_sidebar_nav(user: dict) -> None:
    """Render persistent sidebar navigation available from every page."""
    st.sidebar.markdown(f"## {config.APP_ICON} Menu")
    st.sidebar.markdown(f"**{user['name']}**")
    st.sidebar.markdown("---")

    nav_items = [("dashboard", "🏠 Dashboard")] + [
        (key, f"{icon} {label}") for key, icon, label in _DASHBOARD_CARDS
    ]
    for page_key, label in nav_items:
        if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state[config.SESSION_STATE_PAGE_KEY] = page_key
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth.logout_user()
        st.rerun()


# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------
def main():
    """Streamlit application entry point: auth gate, then dashboard routing."""
    inject_custom_css()

    if not auth.is_logged_in():
        auth.render_auth_page()
        render_footer()
        return

    user = auth.get_current_user()
    render_sidebar_nav(user)

    page = st.session_state.get(config.SESSION_STATE_PAGE_KEY, "dashboard")

    if page == "dashboard":
        render_dashboard_home(user)
    elif page == "search":
        corpus_df, retriever = _init_app_state()
        render_search_page(user, corpus_df, retriever)
    elif page == "history":
        history.render_history_page(user["user_id"])
    elif page == "bookmarks":
        bookmark.render_bookmarks_page(user["user_id"])
    elif page == "assistant":
        render_ai_assistant_page()
    elif page == "profile":
        profile.render_profile_page(user)
    else:
        render_dashboard_home(user)

    render_footer()


if __name__ == "__main__":
    main()
