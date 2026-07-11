#!/usr/bin/env python3
"""
run_evaluation.py
==================
Evaluates the full retrieval + re-ranking pipeline against the official
TREC-COVID relevance judgements (qrels) and reports Precision@10,
Recall@10, MAP, MRR, and NDCG@10.

Run after `python build_index.py`:

    python run_evaluation.py
"""

import logging
import sys

from src import config
from src.data_loader import load_or_download
from src.evaluation import evaluate_system, save_evaluation_results
from src.preprocessing import preprocess_corpus
from src.reranker import rerank
from src.retriever import DenseRetriever
from src.utils import timeit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@timeit
def main(use_reranker: bool = True, num_queries: int = None) -> None:
    """
    Run the end-to-end evaluation over the official TREC-COVID topics.

    Parameters
    ----------
    use_reranker : bool
        If True, evaluate the full pipeline (dense retrieval + cross-encoder
        re-ranking). If False, evaluate dense retrieval alone (useful to
        quantify the re-ranker's contribution).
    num_queries : int, optional
        Evaluate only the first N queries (for a quick smoke test).
    """
    try:
        logger.info("Loading dataset and building retriever ...")
        corpus_df, queries_df, qrels_df = load_or_download()
        corpus_df = preprocess_corpus(corpus_df)
        retriever = DenseRetriever(corpus_df)

        if num_queries:
            queries_df = queries_df.head(num_queries)

        query_results = {}
        for _, row in queries_df.iterrows():
            query_id, query_text = str(row["query_id"]), row["text"]
            try:
                candidates = retriever.retrieve(query_text, top_k=config.TOP_K_RETRIEVE)
                if candidates.empty:
                    query_results[query_id] = []
                    continue

                if use_reranker:
                    ranked = rerank(query_text, candidates, top_n=config.TOP_K_RETRIEVE)
                else:
                    ranked = candidates

                query_results[query_id] = ranked["doc_id"].tolist()
                logger.info("Evaluated query_id=%s ('%s')", query_id, query_text[:60])

            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping query_id=%s due to error: %s", query_id, exc)
                query_results[query_id] = []

        logger.info("Computing IR metrics against qrels ...")
        results = evaluate_system(query_results, qrels_df, k=config.EVAL_K)

        print("\n===== EVALUATION RESULTS =====")
        for metric in (f"Precision@{config.EVAL_K}", f"Recall@{config.EVAL_K}", "MAP", "MRR", f"NDCG@{config.EVAL_K}"):
            print(f"{metric:15s}: {results[metric]:.4f}")
        print(f"Queries evaluated: {results['num_queries_evaluated']}")

        save_evaluation_results(results)

    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
