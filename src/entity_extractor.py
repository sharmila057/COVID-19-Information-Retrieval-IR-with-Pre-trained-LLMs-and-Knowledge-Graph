"""
entity_extractor.py
====================
Extracts named entities relevant to COVID-19 research papers using spaCy,
combining spaCy's generic NER labels (ORG, GPE/LOC, PERSON, etc.) with
domain-specific keyword matching for categories spaCy's default English
model does not natively recognise: DISEASE, VIRUS, DRUG, VACCINE, PROTEIN.

These extracted entities are the nodes used to build the knowledge graph
in knowledge_graph.py.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

_nlp_cache = {}

# Mapping of spaCy's default entity labels to the categories we display
_SPACY_LABEL_MAP = {
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PERSON": "PERSON",
    "NORP": "GROUP",
    "FAC": "FACILITY",
}


def load_nlp_model():
    """
    Load (and cache) the spaCy English NER pipeline, adding an
    EntityRuler that recognises COVID-specific terms (viruses, drugs,
    vaccines, proteins) via keyword/phrase matching.

    Returns
    -------
    spacy.Language

    Raises
    ------
    RuntimeError
        If the spaCy model is not installed.
    """
    if "nlp" in _nlp_cache:
        return _nlp_cache["nlp"]

    try:
        import spacy

        try:
            nlp = spacy.load(config.SPACY_MODEL)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{config.SPACY_MODEL}' not found. "
                f"Install it with: python -m spacy download {config.SPACY_MODEL}"
            ) from exc

        # Add an EntityRuler *before* the statistical NER component so
        # domain-specific phrases (e.g. "spike protein") are matched even
        # when they span multiple tokens.
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            patterns = []
            for term in config.VIRUS_KEYWORDS:
                patterns.append({"label": "VIRUS", "pattern": term})
            for term in config.DRUG_KEYWORDS:
                patterns.append({"label": "DRUG", "pattern": term})
            for term in config.VACCINE_KEYWORDS:
                patterns.append({"label": "VACCINE", "pattern": term})
            for term in config.PROTEIN_KEYWORDS:
                patterns.append({"label": "PROTEIN", "pattern": term})
            ruler.add_patterns(patterns)

        _nlp_cache["nlp"] = nlp
        return nlp

    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load spaCy NLP pipeline: {exc}") from exc


def extract_entities(text: str) -> list:
    """
    Extract named entities from a single piece of text.

    Parameters
    ----------
    text : str
        Title + abstract (or any free text) to analyse.

    Returns
    -------
    list[dict]
        Each dict has keys: 'text' (surface form), 'label' (category).
        Duplicate (text, label) pairs are removed while preserving order.

    Raises
    ------
    ValueError
        If text is empty.
    """
    if not text or not text.strip():
        return []

    try:
        nlp = load_nlp_model()
        doc = nlp(text)

        entities = []
        seen = set()
        for ent in doc.ents:
            label = _SPACY_LABEL_MAP.get(ent.label_, ent.label_)

            # Keep only categories relevant to the COVID-19 IR domain
            relevant_labels = {
                "ORGANIZATION", "LOCATION", "PERSON", "GROUP",
                "VIRUS", "DRUG", "VACCINE", "PROTEIN",
            }
            if label not in relevant_labels:
                continue

            key = (ent.text.strip().lower(), label)
            if key in seen:
                continue
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": label})

        return entities

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Entity extraction failed: {exc}") from exc


def extract_entities_from_documents(documents: list) -> dict:
    """
    Extract entities from multiple documents at once.

    Parameters
    ----------
    documents : list[dict]
        Each dict must have 'doc_id' and combined 'title'/'text' fields,
        e.g. {'doc_id': ..., 'title': ..., 'text': ...}.

    Returns
    -------
    dict
        Mapping doc_id -> list of entity dicts (see extract_entities).
    """
    results = {}
    for doc in documents:
        full_text = f"{doc.get('title', '')}. {doc.get('text', '')}"
        try:
            results[doc["doc_id"]] = extract_entities(full_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity extraction failed for doc_id=%s: %s", doc.get("doc_id"), exc)
            results[doc["doc_id"]] = []
    return results


if __name__ == "__main__":
    sample_text = (
        "Remdesivir and dexamethasone have shown efficacy against SARS-CoV-2 "
        "by targeting the spike protein. The World Health Organization and "
        "researchers in Wuhan, China studied Pfizer's mRNA vaccine."
    )
    for entity in extract_entities(sample_text):
        print(entity)
