#!/usr/bin/env python3
"""
build_index.py
===============
One-time (or on-demand) offline pipeline that:
    1. Downloads / loads the BEIR TREC-COVID dataset.
    2. Preprocesses the corpus text.
    3. Generates dense embeddings with the SentenceTransformer bi-encoder.
    4. Builds and saves a FAISS index for fast retrieval.

Run this script once before launching the Streamlit app:

    python build_index.py

Optional flags:
    --force-download   Re-download the dataset even if a local cache exists.
    --max-docs N        Limit indexing to the first N corpus documents
                         (useful for a quick local smoke-test).
"""

import argparse
import logging
import sys

from src import config
from src.data_loader import load_or_download
from src.embedding_generator import generate_corpus_embeddings
from src.faiss_indexer import build_faiss_index, save_faiss_index
from src.preprocessing import preprocess_corpus
from src.utils import timeit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@timeit
def main(force_download: bool = False, max_docs: int = None) -> None:
    """
    Execute the full offline indexing pipeline.

    Parameters
    ----------
    force_download : bool
        Re-download the dataset even if cached Parquet files exist.
    max_docs : int, optional
        Restrict the corpus to the first N documents (for quick testing).
    """
    try:
        if max_docs:
            config.MAX_CORPUS_DOCS = max_docs

        logger.info("STEP 1/4: Loading dataset ...")
        corpus_df, queries_df, qrels_df = load_or_download(force_download=force_download)
        logger.info(
            "Loaded %d documents, %d queries, %d qrels.",
            len(corpus_df), len(queries_df), len(qrels_df),
        )

        logger.info("STEP 2/4: Preprocessing corpus text ...")
        corpus_df = preprocess_corpus(corpus_df)

        logger.info("STEP 3/4: Generating dense embeddings (this may take a while) ...")
        embeddings = generate_corpus_embeddings(corpus_df, save=True)

        logger.info("STEP 4/4: Building and saving FAISS index ...")
        index = build_faiss_index(embeddings)
        save_faiss_index(index, corpus_df["doc_id"].tolist())

        logger.info("Pipeline complete. The system is ready to serve queries.")
        logger.info("Launch the app with: streamlit run app.py")

    except Exception as exc:  # noqa: BLE001
        logger.error("Indexing pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the COVID-19 IR dense retrieval index.")
    parser.add_argument("--force-download", action="store_true", help="Re-download the dataset.")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit corpus size for a quick run.")
    args = parser.parse_args()

    main(force_download=args.force_download, max_docs=args.max_docs)
