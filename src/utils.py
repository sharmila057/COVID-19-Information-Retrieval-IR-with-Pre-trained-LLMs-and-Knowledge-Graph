"""
utils.py
========
Small shared helper functions used across multiple modules (text
truncation for display, timing decorator, safe formatting, etc.).
"""

import functools
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int = 400) -> str:
    """
    Truncate a string to a maximum number of characters, appending an
    ellipsis if it was shortened.

    Parameters
    ----------
    text : str
    max_chars : int

    Returns
    -------
    str
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def timeit(func):
    """
    Decorator that logs the execution time of the wrapped function.
    Useful for profiling embedding generation, index building, etc.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f seconds.", func.__name__, elapsed)
        return result

    return wrapper


def format_score(score: float) -> str:
    """
    Format a similarity/relevance score for display in the Streamlit UI.

    Parameters
    ----------
    score : float

    Returns
    -------
    str
    """
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    return numerator / denominator if denominator else 0.0
