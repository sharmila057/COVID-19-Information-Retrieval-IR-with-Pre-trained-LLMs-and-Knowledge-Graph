"""
embedding_generator.py
======================
Generates dense vector embeddings for the (preprocessed) TREC-COVID
corpus using the `all-MiniLM-L6-v2` Sentence-Transformer bi-encoder.

The embeddings are L2-normalised so that a FAISS `IndexFlatIP`
(inner-product index) is mathematically equivalent to cosine-similarity
search.
"""

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model_cache = {}


def load_bi_encoder():
    """
    Load (and cache) the SentenceTransformer bi-encoder model.

    Returns
    -------
    SentenceTransformer
        The loaded `all-MiniLM-L6-v2` model.

    Raises
    ------
    RuntimeError
        If the model fails to load (e.g. missing dependency or no internet
        access on first download).
    """
    if "bi_encoder" in _model_cache:
        return _model_cache["bi_encoder"]

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading bi-encoder model: %s", config.BI_ENCODER_MODEL)
        model = SentenceTransformer(config.BI_ENCODER_MODEL)
        _model_cache["bi_encoder"] = model
        return model

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load bi-encoder model: {exc}") from exc


def generate_embeddings(texts: list, show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised dense embeddings.

    Parameters
    ----------
    texts : list[str]
        Documents (or queries) to embed.
    show_progress : bool
        Whether to display a tqdm progress bar during encoding.

    Returns
    -------
    np.ndarray
        Array of shape (len(texts), config.EMBEDDING_DIM), dtype float32.

    Raises
    ------
    ValueError
        If `texts` is empty.
    """
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty list of texts.")

    model = load_bi_encoder()
    try:
        embeddings = model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-norm -> cosine == dot product
        )
        return embeddings.astype("float32")

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc


def generate_corpus_embeddings(corpus_df: pd.DataFrame, save: bool = True) -> np.ndarray:
    """
    Generate and optionally persist embeddings for an entire (preprocessed)
    corpus DataFrame.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Must contain a 'searchable_text' column (see preprocessing.py) and
        a 'doc_id' column.
    save : bool
        If True, saves embeddings (.npy) and the doc_id order (.pkl) to disk.

    Returns
    -------
    np.ndarray
        Corpus embedding matrix of shape (n_docs, embedding_dim).
    """
    if "searchable_text" not in corpus_df.columns:
        raise ValueError("corpus_df must contain a 'searchable_text' column.")

    logger.info("Generating embeddings for %d documents ...", len(corpus_df))
    texts = corpus_df["searchable_text"].tolist()
    embeddings = generate_embeddings(texts)

    if save:
        emb_path = os.path.join(config.EMBEDDINGS_DIR, "corpus_embeddings.npy")
        ids_path = os.path.join(config.EMBEDDINGS_DIR, "corpus_doc_ids.pkl")

        np.save(emb_path, embeddings)
        with open(ids_path, "wb") as f:
            pickle.dump(corpus_df["doc_id"].tolist(), f)

        logger.info("Saved embeddings to %s", emb_path)
        logger.info("Saved doc_id order to %s", ids_path)

    return embeddings


def load_corpus_embeddings() -> tuple:
    """
    Load previously saved corpus embeddings and their doc_id order from disk.

    Returns
    -------
    tuple(np.ndarray, list[str])
        (embeddings matrix, doc_id list) in matching row order.

    Raises
    ------
    FileNotFoundError
        If the embeddings have not been generated yet.
    """
    emb_path = os.path.join(config.EMBEDDINGS_DIR, "corpus_embeddings.npy")
    ids_path = os.path.join(config.EMBEDDINGS_DIR, "corpus_doc_ids.pkl")

    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        raise FileNotFoundError(
            "Corpus embeddings not found. Run `python build_index.py` first."
        )

    embeddings = np.load(emb_path)
    with open(ids_path, "rb") as f:
        doc_ids = pickle.load(f)

    return embeddings, doc_ids


if __name__ == "__main__":
    sample_texts = [
        "Effects of SARS-CoV-2 on the human respiratory system.",
        "Efficacy of mRNA vaccines against emerging COVID-19 variants.",
    ]
    vecs = generate_embeddings(sample_texts)
    print("Embedding shape:", vecs.shape)
