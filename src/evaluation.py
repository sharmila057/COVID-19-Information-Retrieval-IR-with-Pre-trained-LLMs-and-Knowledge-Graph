"""
evaluation.py
=============
Evaluates the retrieval system's ranking quality against the official
TREC-COVID relevance judgements (qrels) using standard Information
Retrieval metrics:

    - Precision@K
    - Recall@K
    - Mean Average Precision (MAP)
    - Mean Reciprocal Rank (MRR)
    - Normalised Discounted Cumulative Gain (NDCG@K)

All metrics are computed per-query and then averaged across queries,
following the standard TREC evaluation methodology.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _relevant_doc_ids(qrels_df: pd.DataFrame, query_id: str, min_score: int = 1) -> set:
    """Return the set of doc_ids judged relevant (score >= min_score) for a query."""
    subset = qrels_df[(qrels_df["query_id"] == query_id) & (qrels_df["score"] >= min_score)]
    return set(subset["doc_id"].astype(str))


def precision_at_k(ranked_doc_ids: list, relevant_ids: set, k: int) -> float:
    """
    Fraction of the top-k retrieved documents that are relevant.

    Parameters
    ----------
    ranked_doc_ids : list[str]
        Retrieved doc_ids sorted by predicted relevance (descending).
    relevant_ids : set[str]
        Ground-truth relevant doc_ids for this query.
    k : int

    Returns
    -------
    float
    """
    if k == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked_doc_ids: list, relevant_ids: set, k: int) -> float:
    """
    Fraction of all relevant documents that were retrieved in the top-k.

    Parameters
    ----------
    ranked_doc_ids : list[str]
    relevant_ids : set[str]
    k : int

    Returns
    -------
    float
    """
    if not relevant_ids:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(ranked_doc_ids: list, relevant_ids: set) -> float:
    """
    Average Precision (AP) for a single query: the mean of the precision
    values computed at each rank where a relevant document is retrieved.

    Parameters
    ----------
    ranked_doc_ids : list[str]
    relevant_ids : set[str]

    Returns
    -------
    float
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def reciprocal_rank(ranked_doc_ids: list, relevant_ids: set) -> float:
    """
    Reciprocal of the rank of the first relevant document retrieved.

    Parameters
    ----------
    ranked_doc_ids : list[str]
    relevant_ids : set[str]

    Returns
    -------
    float
        0.0 if no relevant document was retrieved at all.
    """
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: list, relevance_scores: dict, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at rank k.

    Parameters
    ----------
    ranked_doc_ids : list[str]
        Retrieved doc_ids sorted by predicted relevance (descending).
    relevance_scores : dict
        Mapping doc_id -> graded relevance score (0, 1, 2, ...) from qrels.
    k : int

    Returns
    -------
    float
    """
    top_k = ranked_doc_ids[:k]

    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        rel = relevance_scores.get(doc_id, 0)
        dcg += (2 ** rel - 1) / np.log2(rank + 1)

    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query(ranked_doc_ids: list, query_id: str, qrels_df: pd.DataFrame, k: int = None) -> dict:
    """
    Compute all IR metrics for a single query's ranked results.

    Parameters
    ----------
    ranked_doc_ids : list[str]
        System's ranked output for this query (best first).
    query_id : str
    qrels_df : pd.DataFrame
        Full qrels DataFrame with 'query_id', 'doc_id', 'score' columns.
    k : int, optional
        Cut-off rank (default: config.EVAL_K).

    Returns
    -------
    dict
        Keys: precision_at_k, recall_at_k, average_precision,
        reciprocal_rank, ndcg_at_k.
    """
    k = k or config.EVAL_K
    relevant_ids = _relevant_doc_ids(qrels_df, query_id)

    query_qrels = qrels_df[qrels_df["query_id"] == query_id]
    relevance_scores = dict(zip(query_qrels["doc_id"].astype(str), query_qrels["score"]))

    return {
        "precision_at_k": precision_at_k(ranked_doc_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(ranked_doc_ids, relevant_ids, k),
        "average_precision": average_precision(ranked_doc_ids, relevant_ids),
        "reciprocal_rank": reciprocal_rank(ranked_doc_ids, relevant_ids),
        "ndcg_at_k": ndcg_at_k(ranked_doc_ids, relevance_scores, k),
    }


def evaluate_system(query_results: dict, qrels_df: pd.DataFrame, k: int = None) -> dict:
    """
    Aggregate IR metrics across multiple queries to produce system-level
    scores (Precision@K, Recall@K, MAP, MRR, NDCG@K).

    Parameters
    ----------
    query_results : dict
        Mapping query_id -> list[str] of ranked doc_ids (system output).
    qrels_df : pd.DataFrame
    k : int, optional

    Returns
    -------
    dict
        Keys: 'Precision@K', 'Recall@K', 'MAP', 'MRR', 'NDCG@K',
        'num_queries_evaluated', plus 'per_query' breakdown.

    Raises
    ------
    ValueError
        If query_results is empty.
    """
    k = k or config.EVAL_K
    if not query_results:
        raise ValueError("No query results provided for evaluation.")

    per_query_metrics = {}
    for query_id, ranked_doc_ids in query_results.items():
        try:
            per_query_metrics[query_id] = evaluate_query(ranked_doc_ids, query_id, qrels_df, k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation failed for query_id=%s: %s", query_id, exc)

    if not per_query_metrics:
        raise ValueError("Evaluation failed for all queries.")

    metric_names = ["precision_at_k", "recall_at_k", "average_precision", "reciprocal_rank", "ndcg_at_k"]
    averages = {
        name: float(np.mean([m[name] for m in per_query_metrics.values()]))
        for name in metric_names
    }

    summary = {
        f"Precision@{k}": round(averages["precision_at_k"], 4),
        f"Recall@{k}": round(averages["recall_at_k"], 4),
        "MAP": round(averages["average_precision"], 4),
        "MRR": round(averages["reciprocal_rank"], 4),
        f"NDCG@{k}": round(averages["ndcg_at_k"], 4),
        "num_queries_evaluated": len(per_query_metrics),
        "per_query": per_query_metrics,
    }
    return summary


def save_evaluation_results(results: dict, path: str = None) -> None:
    """
    Persist evaluation results to a JSON file.

    Parameters
    ----------
    results : dict
    path : str, optional
        Defaults to config.EVAL_RESULTS_FILE.
    """
    path = path or config.EVAL_RESULTS_FILE
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to %s", path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to save evaluation results: {exc}") from exc


if __name__ == "__main__":
    demo_qrels = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "doc_id": ["d1", "d2", "d3"],
            "score": [2, 1, 0],
        }
    )
    demo_results = {"q1": ["d2", "d1", "d4", "d3"]}
    print(evaluate_system(demo_results, demo_qrels, k=3))
