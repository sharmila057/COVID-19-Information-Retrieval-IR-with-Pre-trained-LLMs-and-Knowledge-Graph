"""
gemini_service.py
==================
Google Gemini integration used strictly as a post-retrieval assistant.

Gemini is invoked only after the retrieval pipeline has produced
relevant papers. It uses the supplied paper titles/abstracts as context.
"""

import logging

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

_model_cache = {}

_RAG_SYSTEM_INSTRUCTION = (
    "You are a scientific literature assistant embedded in a COVID-19 research "
    "search engine. You MUST answer using ONLY the paper title(s) and abstract(s) "
    "provided in the context below. Do NOT use any outside knowledge, and do NOT "
    "make up facts that are not present in the context. If the provided context "
    "does not contain enough information to answer, say so explicitly instead "
    "of guessing. Keep answers clear, concise, and grounded strictly in the given text."
)


def is_configured() -> bool:
    """Return True if a Gemini API key has been configured."""
    return bool(config.GEMINI_API_KEY)


def _get_model():
    """
    Configure and return the Gemini client.

    Uses the current google-genai SDK.
    """
    if "model" in _model_cache:
        return _model_cache["model"]

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to Streamlit Secrets "
            "or your local .env file."
        )

    try:
        from google import genai

        client = genai.Client(
            api_key=config.GEMINI_API_KEY
        )

        _model_cache["model"] = client
        return client

    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise the Gemini client: {exc}"
        ) from exc


def _truncate(text: str, max_chars: int = None) -> str:
    """Truncate abstract text so prompts stay within a reasonable size."""
    max_chars = max_chars or config.GEMINI_MAX_CONTEXT_CHARS_PER_DOC

    text = text or ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _format_paper_context(papers: list) -> str:
    """
    Format retrieved papers into a numbered context block.
    """
    blocks = []

    for i, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        abstract = _truncate(paper.get("text", ""))

        blocks.append(
            f"[Paper {i}] Title: {title}\n"
            f"Abstract: {abstract}"
        )

    return "\n\n".join(blocks)


def _generate(prompt: str) -> str:
    """
    Send a grounded prompt to Gemini and return the response text.
    """
    try:
        client = _get_model()

        full_prompt = (
            f"{_RAG_SYSTEM_INSTRUCTION}\n\n"
            f"{prompt}"
        )

        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=full_prompt,
        )

        return (response.text or "").strip()

    except RuntimeError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Gemini request failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# PUBLIC RAG FEATURES
# ---------------------------------------------------------------------------

def summarize_paper(title: str, abstract: str) -> str:
    """
    Summarise a single retrieved paper in plain language.
    """
    context = _format_paper_context(
        [{"title": title, "text": abstract}]
    )

    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Write a concise 3-4 sentence plain-language summary "
        "of this paper's purpose, method, and key result, using only "
        "the context above."
    )

    return _generate(prompt)


def explain_terminology(text: str) -> str:
    """
    Explain difficult medical/scientific terminology from retrieved text.
    """
    context = _truncate(
        text,
        config.GEMINI_MAX_CONTEXT_CHARS_PER_DOC
    )

    prompt = (
        f"Context (excerpt from a retrieved research paper):\n"
        f"{context}\n\n"
        "Task: Identify the 3-6 most difficult medical/scientific terms "
        "in this excerpt and explain each one in one simple sentence "
        "a non-expert could understand. Base your explanations only "
        "on how the term is used in this context. Do not introduce "
        "facts not supported by the text. Format as a bulleted list: "
        "**Term** — explanation."
    )

    return _generate(prompt)


def compare_papers(paper_a: dict, paper_b: dict) -> str:
    """
    Compare two retrieved papers.
    """
    context = _format_paper_context(
        [paper_a, paper_b]
    )

    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Compare Paper 1 and Paper 2 using only the context above. "
        "Cover: (1) what each paper studies, (2) how their methods or "
        "findings differ or agree, and (3) one key takeaway from comparing "
        "them. If the context is too limited for a meaningful comparison, "
        "say so explicitly."
    )

    return _generate(prompt)


def generate_key_findings(papers: list) -> str:
    """
    Generate consolidated key findings across retrieved papers.
    """
    context = _format_paper_context(papers)

    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Using only the context above, produce a bulleted list "
        "of the key findings across these papers. Attribute each finding "
        "to its paper number (e.g. '[Paper 2] ...'). Do not add findings "
        "that aren't supported by the given abstracts."
    )

    return _generate(prompt)


def answer_question(question: str, papers: list) -> str:
    """
    Answer a question using retrieved papers as RAG context.
    """
    if not papers:
        raise ValueError(
            "No retrieved papers were supplied as context for this question."
        )

    context = _format_paper_context(papers)

    prompt = (
        f"Context (retrieved research papers):\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Task: Answer the question using ONLY the context above. "
        "Cite which paper number(s) support your answer "
        "(e.g. '[Paper 1]'). If the context does not contain enough "
        "information to answer confidently, say so explicitly instead "
        "of guessing."
    )

    return _generate(prompt)