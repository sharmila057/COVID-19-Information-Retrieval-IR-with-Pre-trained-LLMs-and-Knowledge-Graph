"""
faiss_indexer.py
================
Builds, saves, loads, and searches a FAISS index over the corpus
embeddings for fast approximate/exact nearest-neighbour dense retrieval.

We use `IndexFlatIP` (exact inner-product search). Because embeddings are
L2-normalised (see embedding_generator.py), inner product is equivalent to
cosine similarity while remaining simple, exact, and fast enough for the
~171k-document TREC-COVID corpus on a single machine.
"""

import logging
import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """
    Build an exact inner-product FAISS index from a matrix of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_docs, embedding_dim), dtype float32, L2-normalised.

    Returns
    -------
    faiss.Index
        A populated FAISS `IndexFlatIP`.

    Raises
    ------
    ValueError
        If the embeddings array is empty or has the wrong dtype/shape.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is not installed. Run `pip install faiss-cpu`."
        ) from exc

    if embeddings.size == 0:
        raise ValueError("Cannot build a FAISS index from an empty embeddings array.")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embeddings array, got shape {embeddings.shape}.")

    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    dim = embeddings.shape[1]

    logger.info("Building FAISS IndexFlatIP with dimension %d for %d vectors ...", dim, len(embeddings))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built. Total vectors indexed: %d", index.ntotal)
    return index


def save_faiss_index(index, doc_ids: list) -> None:
    """
    Persist a FAISS index and its accompanying doc_id row-order mapping.

    Parameters
    ----------
    index : faiss.Index
    doc_ids : list[str]
        doc_id for each row in the index, in insertion order.
    """
    try:
        import faiss

        faiss.write_index(index, config.FAISS_INDEX_PATH)
        with open(config.DOC_ID_MAP_PATH, "wb") as f:
            pickle.dump(doc_ids, f)
        logger.info("FAISS index saved to %s", config.FAISS_INDEX_PATH)
        logger.info("doc_id map saved to %s", config.DOC_ID_MAP_PATH)

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to save FAISS index: {exc}") from exc


def load_faiss_index():
    """
    Load a previously saved FAISS index and its doc_id mapping from disk.

    Returns
    -------
    tuple(faiss.Index, list[str])

    Raises
    ------
    FileNotFoundError
        If the index files do not exist yet.
    """
    try:
        import faiss

        if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.DOC_ID_MAP_PATH):
            raise FileNotFoundError(
                "FAISS index not found. Run `python build_index.py` first."
            )

        index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.DOC_ID_MAP_PATH, "rb") as f:
            doc_ids = pickle.load(f)

        logger.info("Loaded FAISS index with %d vectors.", index.ntotal)
        return index, doc_ids

    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load FAISS index: {exc}") from exc


def search_index(index, query_embedding: np.ndarray, top_k: int = None) -> tuple:
    """
    Run a top-k nearest-neighbour search against the FAISS index.

    Parameters
    ----------
    index : faiss.Index
    query_embedding : np.ndarray
        Shape (embedding_dim,) or (1, embedding_dim), L2-normalised.
    top_k : int, optional
        Number of neighbours to retrieve. Defaults to config.TOP_K_RETRIEVE.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        (scores, indices) each of shape (top_k,).
    """
    top_k = top_k or config.TOP_K_RETRIEVE

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = np.ascontiguousarray(query_embedding.astype("float32"))

    try:
        scores, indices = index.search(query_embedding, top_k)
        return scores[0], indices[0]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"FAISS search failed: {exc}") from exc


if __name__ == "__main__":
    # Simple smoke test with random vectors
    rng = np.random.default_rng(42)
    demo_embeddings = rng.random((100, config.EMBEDDING_DIM)).astype("float32")
    demo_embeddings /= np.linalg.norm(demo_embeddings, axis=1, keepdims=True)

    demo_index = build_faiss_index(demo_embeddings)
    demo_scores, demo_idx = search_index(demo_index, demo_embeddings[0], top_k=5)
    print("Top-5 indices:", demo_idx)
    print("Top-5 scores :", demo_scores)
