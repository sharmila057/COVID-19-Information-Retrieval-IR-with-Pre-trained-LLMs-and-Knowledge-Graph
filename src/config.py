"""
config.py
=========
Central configuration file for the COVID-19 Information Retrieval system.

All file paths, model names, and hyperparameters used across the project
are defined here so that every module stays in sync and nothing is
hard-coded in multiple places.
"""

import os

# --------------------------------------------------------------------------
# BASE DIRECTORIES
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure required directories exist at import time
for directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    FAISS_INDEX_DIR,
    RESULTS_DIR,
    ASSETS_DIR,
):
    os.makedirs(directory, exist_ok=True)

# --------------------------------------------------------------------------
# DATASET CONFIGURATION (official BEIR TREC-COVID dataset)
# --------------------------------------------------------------------------
# The BEIR benchmark's TREC-COVID dataset is distributed as a zip archive
# via the official `beir` Python library (https://github.com/beir-cellar/beir),
# NOT via the `datasets`/Hugging Face loader. The archive unpacks into a
# folder containing:
#   - corpus.jsonl        : research paper collection ({"_id","title","text"})
#   - queries.jsonl        : the 50 official TREC-COVID topics ({"_id","text"})
#   - qrels/test.tsv       : relevance judgements (query-id, corpus-id, score)
BEIR_DATASET_NAME = "trec-covid"
BEIR_DATASET_URL = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{BEIR_DATASET_NAME}.zip"
)
BEIR_QRELS_SPLIT = "test"  # TREC-COVID only ships a "test" qrels split

CORPUS_FILE = os.path.join(PROCESSED_DATA_DIR, "corpus.parquet")
QUERIES_FILE = os.path.join(PROCESSED_DATA_DIR, "queries.parquet")
QRELS_FILE = os.path.join(PROCESSED_DATA_DIR, "qrels.parquet")

# Limit corpus size while prototyping locally (set to None for the full ~171k docs)
MAX_CORPUS_DOCS = None  # e.g. 20000 for a quick local run on a laptop

# --------------------------------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------------------------------
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SPACY_MODEL = "en_core_web_sm"

EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE = 64

# --------------------------------------------------------------------------
# RETRIEVAL / RE-RANKING CONFIGURATION
# --------------------------------------------------------------------------
TOP_K_RETRIEVE = 50   # number of candidates fetched from FAISS
TOP_N_RERANK = 10     # number of documents shown to the user after re-ranking

FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "trec_covid.index")
DOC_ID_MAP_PATH = os.path.join(FAISS_INDEX_DIR, "doc_id_map.pkl")

# --------------------------------------------------------------------------
# EVALUATION CONFIGURATION
# --------------------------------------------------------------------------
EVAL_K = 10  # cut-off rank used for Precision@K / Recall@K / NDCG@K
EVAL_RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")
NUM_EVAL_QUERIES = 50  # TREC-COVID ships 50 official topics

# --------------------------------------------------------------------------
# ENTITY EXTRACTION / KNOWLEDGE GRAPH CONFIGURATION
# --------------------------------------------------------------------------
# Domain keyword lists used to enrich spaCy's generic NER with COVID-specific
# categories that spaCy's default English model does not know about.
VIRUS_KEYWORDS = [
    "sars-cov-2", "covid-19", "coronavirus", "influenza", "mers-cov",
    "sars-cov", "h1n1", "ebola", "hiv", "rhinovirus",
]
DRUG_KEYWORDS = [
    "remdesivir", "hydroxychloroquine", "chloroquine", "dexamethasone",
    "favipiravir", "lopinavir", "ritonavir", "ivermectin", "tocilizumab",
    "azithromycin", "molnupiravir", "paxlovid",
]
VACCINE_KEYWORDS = [
    "pfizer", "moderna", "astrazeneca", "sputnik", "covaxin", "novavax",
    "janssen", "sinopharm", "sinovac", "mrna vaccine", "vaccine",
]
PROTEIN_KEYWORDS = [
    "spike protein", "ace2", "furin", "protease", "rna polymerase",
    "nucleocapsid", "hemagglutinin", "cytokine", "antibody", "receptor",
]

KG_MAX_DOCS = 10          # build the KG from top-N re-ranked documents
KG_IMAGE_PATH = os.path.join(RESULTS_DIR, "knowledge_graph.png")

# --------------------------------------------------------------------------
# STREAMLIT UI CONFIGURATION
# --------------------------------------------------------------------------
APP_TITLE = "COVID-19 Research Paper Information Retrieval System"
APP_ICON = "🦠"

# --------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (.env)
# --------------------------------------------------------------------------
# Secrets (MongoDB URI, Gemini API key) are never hard-coded. They are
# loaded from a local `.env` file (see `.env.example` for the template)
# using python-dotenv. This call is safe to repeat across modules —
# load_dotenv() is idempotent and a no-op if `.env` is absent.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# --------------------------------------------------------------------------
# AUTHENTICATION CONFIGURATION
# --------------------------------------------------------------------------
BCRYPT_ROUNDS = 12                 # cost factor for bcrypt password hashing
MIN_PASSWORD_LENGTH = 8
SESSION_STATE_USER_KEY = "auth_user"      # st.session_state key holding the logged-in user dict
SESSION_STATE_PAGE_KEY = "current_page"   # st.session_state key holding the active dashboard page


# --------------------------------------------------------------------------
# MONGODB ATLAS CONFIGURATION
# --------------------------------------------------------------------------
try:
    import streamlit as st

    MONGO_URI = os.environ.get("MONGO_URI") or st.secrets.get("MONGO_URI", "")
    MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME") or st.secrets.get(
        "MONGO_DB_NAME", "covid_ir_system"
    )

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get(
        "GEMINI_API_KEY", ""
    )
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME") or st.secrets.get(
        "GEMINI_MODEL_NAME", "gemini-3.6-flash"
    )

except Exception:
    MONGO_URI = os.environ.get("MONGO_URI", "")
    MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "covid_ir_system")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME = os.environ.get(
        "GEMINI_MODEL_NAME", "gemini-3.6-flash"
    )

USERS_COLLECTION = "users"
SEARCH_HISTORY_COLLECTION = "search_history"
BOOKMARKS_COLLECTION = "bookmarks"

# Hard cap on retrieved context sent to Gemini
GEMINI_MAX_CONTEXT_CHARS_PER_DOC = 1200

