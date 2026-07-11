"""
reranker.py
===========
Re-ranks the Top-K candidates returned by the dense retriever using a
Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

Why re-rank?
------------
Bi-encoders (SentenceTransformers) encode the query and each document
*independently* into fixed vectors, which is fast but loses fine-grained
query-document interaction. A cross-encoder instead feeds the
(query, document) pair *jointly* through a transformer and outputs a single
relevance score, which is far more accurate but too slow to run over an
entire corpus. The standard IR pattern is therefore:

    FAISS (fast, coarse) -> Top-50 candidates -> Cross-Encoder (slow, precise) -> Top-10
"""

import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model_cache = {}


def load_cross_encoder():
    """
    Load (and cache) the CrossEncoder re-ranking model.

    Returns
    -------
    CrossEncoder

    Raises
    ------
    RuntimeError
        If the model fails to load.
    """
    if "cross_encoder" in _model_cache:
        return _model_cache["cross_encoder"]

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", config.CROSS_ENCODER_MODEL)
        model = CrossEncoder(config.CROSS_ENCODER_MODEL)
        _model_cache["cross_encoder"] = model
        return model

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load cross-encoder model: {exc}") from exc


def rerank(query: str, candidates_df: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    """
    Re-rank retrieved candidate documents against the query using the
    cross-encoder and return the top-n most relevant.

    Parameters
    ----------
    query : str
        The original (raw) user query.
    candidates_df : pd.DataFrame
        Output of DenseRetriever.retrieve(); must contain 'doc_id', 'title',
        'text', and 'dense_score' columns.
    top_n : int, optional
        Number of documents to keep after re-ranking (default:
        config.TOP_N_RERANK).

    Returns
    -------
    pd.DataFrame
        candidates_df with an added 'rerank_score' column, sorted
        descending by 'rerank_score', truncated to top_n rows.

    Raises
    ------
    ValueError
        If candidates_df is empty.
    """
    top_n = top_n or config.TOP_N_RERANK

    if candidates_df.empty:
        raise ValueError("No candidate documents to re-rank (empty DataFrame).")

    try:
        model = load_cross_encoder()

        # Build (query, document) pairs. We re-rank on title + abstract so
        # the cross-encoder sees the same information a human reader would.
        pairs = [
            (query, f"{row['title']}. {row['text']}")
            for _, row in candidates_df.iterrows()
        ]

        logger.info("Re-ranking %d candidates with cross-encoder ...", len(pairs))
        rerank_scores = model.predict(pairs, show_progress_bar=False)

        result_df = candidates_df.copy()
        result_df["rerank_score"] = rerank_scores
        result_df = result_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

        return result_df.head(top_n)

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Re-ranking failed: {exc}") from exc


if __name__ == "__main__":
    demo_candidates = pd.DataFrame(
        {
            "doc_id": ["1", "2", "3"],
            "title": [
                "Incubation period of SARS-CoV-2",
                "Economic impact of COVID-19 lockdowns",
                "Clinical features of COVID-19 patients",
            ],
            "text": [
                "We estimate a median incubation period of 5.1 days.",
                "Lockdowns caused significant GDP contraction worldwide.",
                "Fever and cough were the most common symptoms.",
            ],
            "dense_score": [0.81, 0.55, 0.78],
        }
    )
    demo_result = rerank("What is the incubation period of COVID-19?", demo_candidates, top_n=3)
    print(demo_result[["title", "dense_score", "rerank_score"]])
