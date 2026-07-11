"""
retriever.py
============
High-level dense-retrieval interface that ties together the bi-encoder
(embedding_generator.py) and the FAISS index (faiss_indexer.py) to return
the Top-K most relevant documents for a natural-language user query.
"""

import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.embedding_generator import generate_embeddings
from src.faiss_indexer import load_faiss_index, search_index
from src.preprocessing import preprocess_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Wraps the FAISS index + bi-encoder to perform dense (semantic) document
    retrieval for a given corpus.

    Attributes
    ----------
    index : faiss.Index
        The loaded FAISS inner-product index.
    doc_ids : list[str]
        doc_id for every row of the index, in insertion order.
    corpus_df : pd.DataFrame
        Full corpus with 'doc_id', 'title', 'text' columns, used to fetch
        display-ready title/abstract text for retrieved doc_ids.
    """

    def __init__(self, corpus_df: pd.DataFrame):
        """
        Parameters
        ----------
        corpus_df : pd.DataFrame
            The preprocessed corpus (see preprocessing.preprocess_corpus).

        Raises
        ------
        FileNotFoundError
            If the FAISS index has not been built yet.
        """
        self.index, self.doc_ids = load_faiss_index()
        self.corpus_df = corpus_df.set_index("doc_id", drop=False)

    def retrieve(self, query: str, top_k: int = None) -> pd.DataFrame:
        """
        Retrieve the top-k most relevant documents for a query using dense
        (embedding cosine-similarity) search.

        Parameters
        ----------
        query : str
            Raw user query string.
        top_k : int, optional
            Number of documents to retrieve (default: config.TOP_K_RETRIEVE).

        Returns
        -------
        pd.DataFrame
            Columns: ['doc_id', 'title', 'text', 'dense_score'], sorted by
            dense_score descending.

        Raises
        ------
        ValueError
            If the query is empty after cleaning.
        """
        top_k = top_k or config.TOP_K_RETRIEVE

        clean_query = preprocess_query(query)
        if not clean_query:
            raise ValueError("Query is empty after preprocessing. Please enter a valid query.")

        try:
            query_embedding = generate_embeddings([clean_query], show_progress=False)[0]
            scores, indices = search_index(self.index, query_embedding, top_k=top_k)

            results = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue  # FAISS pads with -1 if fewer than top_k results exist
                doc_id = self.doc_ids[idx]
                if doc_id not in self.corpus_df.index:
                    continue
                row = self.corpus_df.loc[doc_id]
                results.append(
                    {
                        "doc_id": doc_id,
                        "title": row["title"],
                        "text": row["text"],
                        "dense_score": float(score),
                    }
                )

            result_df = pd.DataFrame(results)
            logger.info("Retrieved %d candidate documents for query: '%s'", len(result_df), query)
            return result_df

        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Dense retrieval failed for query '{query}': {exc}") from exc


if __name__ == "__main__":
    from src.data_loader import load_or_download
    from src.preprocessing import preprocess_corpus

    corpus, _, _ = load_or_download()
    corpus = preprocess_corpus(corpus)

    retriever = DenseRetriever(corpus)
    demo_results = retriever.retrieve("What is the incubation period of COVID-19?", top_k=10)
    print(demo_results[["doc_id", "title", "dense_score"]])
