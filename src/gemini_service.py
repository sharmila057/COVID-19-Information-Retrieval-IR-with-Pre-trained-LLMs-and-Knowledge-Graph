"""
gemini_service.py
==================
Google Gemini integration used strictly as a *post-retrieval* assistant.

Design principle (Retrieval-Augmented Generation only)
-------------------------------------------------------
Gemini is NEVER used to search for or rank documents — that job belongs
entirely to the existing dense-retrieval + cross-encoder re-ranking
pipeline (retriever.py / reranker.py), which is left untouched. Gemini is
only invoked *after* retrieval has already produced a Top-N result set,
and every prompt in this module explicitly instructs the model to answer
using ONLY the supplied paper titles/abstracts — never its own background
knowledge — and to say so plainly if the provided context is insufficient.

Features
--------
    - summarize_paper(title, abstract)
    - explain_terminology(text)
    - compare_papers(paper_a, paper_b)
    - generate_key_findings(papers)
    - answer_question(question, papers)   (RAG Q&A over retrieved abstracts)
"""

import logging

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model_cache = {}

_RAG_SYSTEM_INSTRUCTION = (
    "You are a scientific literature assistant embedded in a COVID-19 research search engine. "
    "You MUST answer using ONLY the paper title(s) and abstract(s) provided in the context below. "
    "Do NOT use any outside knowledge, and do NOT make up facts that are not present in the context. "
    "If the provided context does not contain enough information to answer, say so explicitly instead "
    "of guessing. Keep answers clear, concise, and grounded strictly in the given text."
)


def is_configured() -> bool:
    """Return True if a Gemini API key has been configured via .env."""
    return bool(config.GEMINI_API_KEY)


def _get_model():
    """
    Configure (once) and return the Gemini generative model client.

    Returns
    -------
    google.generativeai.GenerativeModel

    Raises
    ------
    RuntimeError
        If GEMINI_API_KEY is not configured or the client fails to initialise.
    """
    if "model" in _model_cache:
        return _model_cache["model"]

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file — see .env.example "
            "and the README's 'Gemini API Setup' section."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            system_instruction=_RAG_SYSTEM_INSTRUCTION,
        )
        _model_cache["model"] = model
        return model

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to initialise the Gemini client: {exc}") from exc


def _truncate(text: str, max_chars: int = None) -> str:
    """Truncate abstract text so prompts stay within a reasonable size."""
    max_chars = max_chars or config.GEMINI_MAX_CONTEXT_CHARS_PER_DOC
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _format_paper_context(papers: list) -> str:
    """
    Format a list of {'title', 'text'} paper dicts into a numbered context
    block suitable for insertion into a Gemini prompt.

    Parameters
    ----------
    papers : list[dict]

    Returns
    -------
    str
    """
    blocks = []
    for i, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        abstract = _truncate(paper.get("text", ""))
        blocks.append(f"[Paper {i}] Title: {title}\nAbstract: {abstract}")
    return "\n\n".join(blocks)


def _generate(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the response text, with consistent
    error handling across all public functions in this module.

    Parameters
    ----------
    prompt : str

    Returns
    -------
    str

    Raises
    ------
    RuntimeError
        If the Gemini API call fails for any reason.
    """
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        return (response.text or "").strip()
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# PUBLIC RAG FEATURES
# ---------------------------------------------------------------------------
def summarize_paper(title: str, abstract: str) -> str:
    """
    Summarise a single retrieved paper in plain language.

    Parameters
    ----------
    title : str
    abstract : str

    Returns
    -------
    str
    """
    context = _format_paper_context([{"title": title, "text": abstract}])
    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Write a concise 3-4 sentence plain-language summary of this paper's purpose, "
        "method, and key result, using only the context above."
    )
    return _generate(prompt)


def explain_terminology(text: str) -> str:
    """
    Explain difficult medical/scientific terminology found in a piece of
    retrieved text (e.g. an abstract), in plain language.

    Parameters
    ----------
    text : str
        The abstract or passage whose terminology should be explained.

    Returns
    -------
    str
    """
    context = _truncate(text, config.GEMINI_MAX_CONTEXT_CHARS_PER_DOC)
    prompt = (
        f"Context (excerpt from a retrieved research paper):\n{context}\n\n"
        "Task: Identify the 3-6 most difficult medical/scientific terms in this excerpt and "
        "explain each one in one simple sentence a non-expert could understand. Base your "
        "explanations only on how the term is used in this context; do not introduce facts not "
        "supported by the text. Format as a bulleted list: **Term** — explanation."
    )
    return _generate(prompt)


def compare_papers(paper_a: dict, paper_b: dict) -> str:
    """
    Compare two retrieved papers' focus, methods, and findings.

    Parameters
    ----------
    paper_a : dict
        {'title': str, 'text': str}
    paper_b : dict
        {'title': str, 'text': str}

    Returns
    -------
    str
    """
    context = _format_paper_context([paper_a, paper_b])
    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Compare Paper 1 and Paper 2 using only the context above. Cover: "
        "(1) what each paper studies, (2) how their methods or findings differ or agree, and "
        "(3) one key takeaway from comparing them. If the context is too limited for a "
        "meaningful comparison, say so explicitly."
    )
    return _generate(prompt)


def generate_key_findings(papers: list) -> str:
    """
    Generate a consolidated bulleted list of key findings across several
    retrieved papers.

    Parameters
    ----------
    papers : list[dict]
        Each dict: {'title': str, 'text': str}.

    Returns
    -------
    str
    """
    context = _format_paper_context(papers)
    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Using only the context above, produce a bulleted list of the key findings "
        "across these papers. Attribute each finding to its paper number (e.g. '[Paper 2] ...'). "
        "Do not add findings that aren't supported by the given abstracts."
    )
    return _generate(prompt)


def answer_question(question: str, papers: list) -> str:
    """
    Answer a user's question strictly using the retrieved papers as
    context (Retrieval-Augmented Generation). Gemini must not answer from
    its own background knowledge.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    papers : list[dict]
        The retrieved papers to use as grounding context:
        each dict is {'title': str, 'text': str}.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If no papers are supplied as context.
    """
    if not papers:
        raise ValueError("No retrieved papers were supplied as context for this question.")

    context = _format_paper_context(papers)
    prompt = (
        f"Context (retrieved research papers):\n{context}\n\n"
        f"Question: {question}\n\n"
        "Task: Answer the question using ONLY the context above. Cite which paper number(s) "
        "support your answer (e.g. '[Paper 1]'). If the context does not contain enough "
        "information to answer confidently, say so explicitly instead of guessing."
    )
    return _generate(prompt)
