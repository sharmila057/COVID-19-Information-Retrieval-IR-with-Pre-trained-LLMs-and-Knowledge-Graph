"""
data_loader.py
==============
Downloads and loads the official BEIR TREC-COVID dataset using the
official `beir` Python library (https://github.com/beir-cellar/beir),
and caches the parsed corpus, queries, and qrels locally as Parquet
files for fast re-use in later pipeline stages.

The BEIR TREC-COVID dataset is distributed as a single zip archive that
unpacks into a folder containing:
    1. corpus.jsonl   : ~171,000 scientific papers, one JSON object per
                         line with keys "_id", "title", "text".
    2. queries.jsonl  : 50 official TREC-COVID topics, one JSON object
                         per line with keys "_id", "text".
    3. qrels/test.tsv : relevance judgements as a TSV file with columns
                         (query-id, corpus-id, score).

Unlike the generic Hugging Face `datasets` loader, the official `beir`
library's `GenericDataLoader` knows how to parse this exact on-disk
layout (including the `qrels/test.tsv` relevance file, which is NOT
exposed as a separate Hugging Face "qrels" config for this dataset).
This module therefore uses `beir.util.download_and_unzip` +
`GenericDataLoader` instead of `datasets.load_dataset`.

Reference: https://github.com/beir-cellar/beir
Dataset archive: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
"""

import logging
import os
import sys

import pandas as pd

# Allow running this file both as a module (src.data_loader) and as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_beir_dataset() -> str:
    """
    Download (if not already cached on disk) and unzip the official BEIR
    TREC-COVID dataset archive using the `beir` library's utility function.

    Returns
    -------
    str
        Path to the local folder containing `corpus.jsonl`, `queries.jsonl`,
        and the `qrels/` subfolder (e.g. `data/raw/trec-covid`).

    Raises
    ------
    RuntimeError
        If the download or unzip step fails (e.g. no internet connection,
        or the remote archive is temporarily unavailable).
    """
    try:
        from beir import util

        url = config.BEIR_DATASET_URL
        out_dir = config.RAW_DATA_DIR

        logger.info("Downloading official BEIR TREC-COVID dataset from %s ...", url)
        # download_and_unzip is idempotent: if the zip/folder already exists
        # under out_dir it will skip re-downloading.
        data_path = util.download_and_unzip(url, out_dir)
        logger.info("Dataset ready at %s", data_path)
        return data_path

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download the official BEIR TREC-COVID dataset: {exc}"
        ) from exc


def load_beir_dataset(data_path: str, split: str = None) -> tuple:
    """
    Parse a downloaded BEIR dataset folder using the official
    `GenericDataLoader`.

    Parameters
    ----------
    data_path : str
        Folder returned by `download_beir_dataset()`, containing
        `corpus.jsonl`, `queries.jsonl`, and `qrels/`.
    split : str, optional
        Which qrels split to load. TREC-COVID only ships a "test" split
        (defaults to `config.BEIR_QRELS_SPLIT`).

    Returns
    -------
    tuple(dict, dict, dict)
        (corpus, queries, qrels) in BEIR's native in-memory format:
            corpus  : {doc_id: {"title": str, "text": str}}
            queries : {query_id: query_text}
            qrels   : {query_id: {doc_id: relevance_score}}

    Raises
    ------
    RuntimeError
        If the folder is missing expected files or parsing otherwise fails.
    """
    split = split or config.BEIR_QRELS_SPLIT
    try:
        from beir.datasets.data_loader import GenericDataLoader

        logger.info("Parsing BEIR dataset at %s (qrels split='%s') ...", data_path, split)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        logger.info(
            "Parsed %d corpus documents, %d queries, %d qrels query groups.",
            len(corpus), len(queries), len(qrels),
        )
        return corpus, queries, qrels

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to parse the BEIR TREC-COVID dataset at '{data_path}': {exc}"
        ) from exc


def _corpus_to_dataframe(corpus: dict) -> pd.DataFrame:
    """
    Convert BEIR's native corpus dict into this project's standard
    DataFrame schema: columns ['doc_id', 'title', 'text'].

    Parameters
    ----------
    corpus : dict
        {doc_id: {"title": str, "text": str}}, as returned by GenericDataLoader.

    Returns
    -------
    pd.DataFrame
    """
    records = [
        {
            "doc_id": str(doc_id),
            "title": doc.get("title", "") or "",
            "text": doc.get("text", "") or "",
        }
        for doc_id, doc in corpus.items()
    ]
    df = pd.DataFrame(records, columns=["doc_id", "title", "text"])

    if config.MAX_CORPUS_DOCS:
        df = df.head(config.MAX_CORPUS_DOCS)

    return df


def _queries_to_dataframe(queries: dict) -> pd.DataFrame:
    """
    Convert BEIR's native queries dict into this project's standard
    DataFrame schema: columns ['query_id', 'text'].

    Parameters
    ----------
    queries : dict
        {query_id: query_text}, as returned by GenericDataLoader.

    Returns
    -------
    pd.DataFrame
    """
    records = [{"query_id": str(qid), "text": text} for qid, text in queries.items()]
    return pd.DataFrame(records, columns=["query_id", "text"])


def _qrels_to_dataframe(qrels: dict) -> pd.DataFrame:
    """
    Convert BEIR's nested qrels dict into this project's standard flat
    DataFrame schema: columns ['query_id', 'doc_id', 'score'].

    Parameters
    ----------
    qrels : dict
        {query_id: {doc_id: relevance_score}}, as returned by GenericDataLoader.

    Returns
    -------
    pd.DataFrame
    """
    records = [
        {"query_id": str(qid), "doc_id": str(doc_id), "score": int(score)}
        for qid, doc_scores in qrels.items()
        for doc_id, score in doc_scores.items()
    ]
    return pd.DataFrame(records, columns=["query_id", "doc_id", "score"])


def load_or_download(force_download: bool = False) -> tuple:
    """
    Load corpus, queries, and qrels from local Parquet cache if present,
    otherwise download the official BEIR TREC-COVID archive, parse it with
    `GenericDataLoader`, and cache the result as Parquet for future runs.

    Parameters
    ----------
    force_download : bool
        If True, re-download and re-parse even if local Parquet cache
        files already exist.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (corpus_df, queries_df, qrels_df) — the same schema used
        throughout the rest of the pipeline (preprocessing, embedding,
        retrieval, evaluation), regardless of the underlying data source.
    """
    try:
        if (
            not force_download
            and os.path.exists(config.CORPUS_FILE)
            and os.path.exists(config.QUERIES_FILE)
            and os.path.exists(config.QRELS_FILE)
        ):
            logger.info("Loading cached dataset from %s ...", config.PROCESSED_DATA_DIR)
            corpus_df = pd.read_parquet(config.CORPUS_FILE)
            queries_df = pd.read_parquet(config.QUERIES_FILE)
            qrels_df = pd.read_parquet(config.QRELS_FILE)
        else:
            data_path = download_beir_dataset()
            corpus, queries, qrels = load_beir_dataset(data_path, split=config.BEIR_QRELS_SPLIT)

            corpus_df = _corpus_to_dataframe(corpus)
            queries_df = _queries_to_dataframe(queries)
            qrels_df = _qrels_to_dataframe(qrels)

            corpus_df.to_parquet(config.CORPUS_FILE, index=False)
            queries_df.to_parquet(config.QUERIES_FILE, index=False)
            qrels_df.to_parquet(config.QRELS_FILE, index=False)
            logger.info("Dataset cached to %s", config.PROCESSED_DATA_DIR)

        return corpus_df, queries_df, qrels_df

    except Exception as exc:  # noqa: BLE001
        logger.error("Dataset loading failed: %s", exc)
        raise


if __name__ == "__main__":
    corpus, queries, qrels = load_or_download()
    print(f"Corpus : {len(corpus)} documents")
    print(f"Queries: {len(queries)} topics")
    print(f"Qrels  : {len(qrels)} relevance judgements")
    print(corpus.head(3))
