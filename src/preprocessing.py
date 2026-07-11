"""
preprocessing.py
================
Cleans and normalizes the raw TREC-COVID research-paper text before it is
fed to the sentence-embedding model.

The cleaning is intentionally light-weight: transformer-based sentence
encoders are trained on natural language, so aggressive steps such as
stop-word removal or stemming are avoided as they would hurt retrieval
quality. We only strip noise (HTML tags, URLs, extra whitespace, control
characters) and standardise casing artefacts.
"""

import logging
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def clean_text(text: str) -> str:
    """
    Remove HTML tags, URLs, control characters, and redundant whitespace
    from a single piece of text.

    Parameters
    ----------
    text : str
        Raw input text (title or abstract).

    Returns
    -------
    str
        Cleaned text. Returns an empty string for null / non-string input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_searchable_document(title: str, abstract: str) -> str:
    """
    Combine a cleaned title and abstract into a single searchable string.

    The title is repeated once so that title terms receive slightly more
    weight in the resulting sentence embedding (a common trick in IR).

    Parameters
    ----------
    title : str
    abstract : str

    Returns
    -------
    str
        Combined "title. title. abstract" string ready for embedding.
    """
    title_clean = clean_text(title)
    abstract_clean = clean_text(abstract)

    if not title_clean and not abstract_clean:
        return ""
    return f"{title_clean}. {title_clean}. {abstract_clean}".strip()


def preprocess_corpus(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning to an entire corpus DataFrame and add a `searchable_text`
    column used downstream by the embedding generator.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Must contain 'doc_id', 'title', 'text' columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added/overwritten 'title', 'text',
        and 'searchable_text' columns. Empty documents are dropped.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_cols = {"doc_id", "title", "text"}
    missing = required_cols - set(corpus_df.columns)
    if missing:
        raise ValueError(f"corpus_df is missing required columns: {missing}")

    logger.info("Preprocessing %d corpus documents ...", len(corpus_df))

    df = corpus_df.copy()
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    df["searchable_text"] = df.apply(
        lambda row: build_searchable_document(row["title"], row["text"]), axis=1
    )

    before = len(df)
    df = df[df["searchable_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d documents with empty title/abstract.", dropped)

    logger.info("Preprocessing complete. %d documents remain.", len(df))
    return df


def preprocess_query(query_text: str) -> str:
    """
    Clean a single user query string before encoding.

    Parameters
    ----------
    query_text : str

    Returns
    -------
    str
    """
    return clean_text(query_text)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "doc_id": ["1", "2"],
            "title": ["<b>COVID-19</b> and Lung Damage", None],
            "text": [
                "This study investigates   the effects of SARS-CoV-2 on lung tissue. See http://example.com",
                "",
            ],
        }
    )
    print(preprocess_corpus(sample))
