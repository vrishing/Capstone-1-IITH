"""
Single source of truth for settings. Secrets come from .env (never write keys here).
Paths are anchored to this file's folder, so scripts behave the same from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


def _p(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


# --- Corpus -------------------------------------------------------------
# Playlists live in playlists.json, one entry per playlist with its own language. The active one is
# chosen by `--playlist <name|url>` on run_pipeline.py, or the PLAYLIST environment variable (for
# glossary_builder.py and other scripts), or the registry's "default". select_playlist() below sets
# PLAYLIST_ID, TARGET_LANGUAGE, TRANSCRIPT_LANGS and the glossary paths from that entry.
_FALLBACK = {"id": "PLD0OXwSZSSA8", "language": "te", "name": "mb-test"}  # used only if playlists.json is missing

# --- ASR (IndicConformer only) --------------------------------------------
ASR_DECODER = "ctc"        # "ctc" or "rnnt"; changing it reruns the ASR stage for every video
ASR_MAX_CHUNK_S = 8.0      # longest audio piece sent to the model (hard ceiling is about 30 s)
ASR_TOP_DB = 35.0          # silence threshold for cutting
ASR_PAD_S = 0.2            # silence kept around each chunk
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Pipeline state and data ------------------------------------------------
ASR_OUTPUT_DIR = _p("data", "asr_output")   # wav audio and *_asr.json
CAPTIONS_DIR = _p("data", "transcripts")    # baseline captions
MANIFEST_PATH = _p("data", "manifest.db")
MAX_ATTEMPTS = 3

# --- Correction layer (correction.py) ---------------------------------------------
CORRECTED_DIR = _p("data", "corrected")   # *_corrected.json: corrected text + raw_segments + change log
CORRECTION_AUTO_SCORE = 92.0      # apply a fix at or above this similarity (0-100) ...
CORRECTION_MARGIN = 5.0           # ... when the best entity leads the runner-up entity by this much
CORRECTION_REVIEW_SCORE = 86.0    # from here up to the auto score: listed for review, text unchanged
# LLM review of correction proposals: one verdict per distinct (heard word, fix) pair, batched and cached
# in data/corrected/_llm_cache.json, so a playlist costs a few calls. It never writes text. Off unless
# CORRECTION_USE_LLM=1 in .env. Inspect the verdicts first with: python correction.py llm-review
CORRECTION_USE_LLM = os.environ.get("CORRECTION_USE_LLM", "0").strip().lower() in ("1", "true", "yes")
CORRECTION_LLM_AUDIT_AUTO = True  # also ask about high-score fixes (catches real words that sound like names)
CORRECTION_LLM_BATCH = 50         # pairs per LLM call (Gemini free tier: 20 calls per day per model)
CORRECTION_DROP_LEADING_CHARS = 2  # drop leading segments of at most this many characters (jingle/noise); 0 = keep

# --- Evidence spans (spans.py) ---------------------------------------------------
SPAN_MIN_S = 45.0     # a span closes at the first pause of at least SPAN_PAUSE_S after this length
SPAN_MAX_S = 90.0     # hard ceiling; spans never overlap
SPAN_PAUSE_S = 1.0    # gap between ASR segments that counts as a long pause
SPANS_DIR = _p("data", "spans")
# Old fixed windows, read only by the retired 06_asr_to_chunks.py.
CHUNK_SECONDS = 60
CHUNK_OVERLAP_SECONDS = 10

# --- Embedding and vector store ---------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # BGE-M3 dense size; change if you swap models
QDRANT_PATH = _p("qdrant_data")
COLLECTION_NAME = "transcript_chunks"   # old fixed-window index (retired scripts 02/07/03)
SPANS_COLLECTION = "spans_v1"           # span index: dense + sparse, one collection for all playlists
INDEX_DIR = _p("data", "index")         # per-video "indexed" markers and the cluster map
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "auto")   # auto | cpu | cuda | cuda:N
EMBED_MAX_LEN = 1024                    # tokens per span (BGE-M3 reads up to 8192)
EMBED_BATCH = 16

# --- Generation and verification (Gemini through llm_client.py) -----------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-3.6-flash"
TOP_K = 10
ANSWER_MAX_ATTEMPTS = 3      # generation attempts per question: 1 + 2 regenerations
ANSWER_STRICT = True        # any unsupported sentence after the retries -> abstain
ANSWER_CACHE_DIR = _p("data", "answers")
TRACE_DIR = _p("data", "traces")
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 30   # fused candidates the reranker reads per question
RETRIEVE_PER_LIST = 40   # candidates taken from each of the dense and sparse lists

# --- Which LLM answers and verifies (llm_client.py) --------------------------------------------
# "gemini" = Google's API. "openai" = a self-hosted model behind any OpenAI-compatible server
# (vLLM, Ollama, llama.cpp). Put these in .env to switch without editing code:
#   LLM_BACKEND=openai  LLM_BASE_URL=http://localhost:8000/v1  LLM_MODEL=<name the server serves>
# Check the model first with:  python llm_client.py --check --role verifier
LLM_BACKEND = os.environ.get("LLM_BACKEND", "gemini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_VERIFIER_MODEL = os.environ.get("LLM_VERIFIER_MODEL", "")   # empty = same as LLM_MODEL
LLM_API_KEY = os.environ.get("LLM_API_KEY") or None             # most local servers need none
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "180"))    # shared GPUs can be slow

# Read by the verifier (not built yet).
VERIFIER_MODEL = "gemini-3.6-flash"
VERIFIER_ENABLED = True
ABSTAIN_ON_UNSUPPORTED = True
VERIFIER_ON_PARTIAL = "warn"     # partly supported answers are shown with a caution warning
VERIFIER_ON_ERROR = "abstain"    # fail closed if the verifier itself errors
VERIFIER_MAX_RETRIES = 2
ABSTAIN_MESSAGE = "I could not find enough support in the specified playlist."


# --- Glossary (built per playlist from Wikipedia and Wikidata by glossary_builder.py) -----------
# One folder per playlist; the paths are set by select_playlist(). Wikimedia asks every script to
# identify itself: put EMAIL_WIKIMEDIA=you@example.com in .env (the builder refuses the placeholder).
GLOSSARY_USER_AGENT = "PravachanamRAG/0.1 (" + os.environ.get("EMAIL_WIKIMEDIA", "CHANGE_ME@example.com") + ")"


# --- Active playlist -------------------------------------------------------------------------
def select_playlist(key_or_url: str | None = None) -> dict:
    """Point every playlist-dependent setting at one registry entry. Returns the entry."""
    global PLAYLIST_KEY, PLAYLIST_ID, PLAYLIST_NAME, YOUTUBE_PLAYLIST_ID, TARGET_LANGUAGE
    global TRANSCRIPT_LANGS, PLAYLIST_META_PATH, SAMPLE_IDS_PATH
    global GLOSSARY_DIR, DOMAIN_PACK_PATH, ALIAS_TABLE_PATH, PROTECTED_WORDS_PATH
    import playlists
    reg = playlists.load()
    if not reg["playlists"] and not key_or_url:
        key, entry = "default", dict(_FALLBACK)
    else:
        key, entry = playlists.resolve(key_or_url)
    PLAYLIST_KEY = key
    PLAYLIST_ID = entry["id"]
    PLAYLIST_NAME = entry.get("name", key)
    YOUTUBE_PLAYLIST_ID = PLAYLIST_ID
    TARGET_LANGUAGE = playlists.check_language(entry["language"])
    # YouTube captions are a baseline for comparison only: target-language tracks, no English fallback.
    TRANSCRIPT_LANGS = [TARGET_LANGUAGE, f"{TARGET_LANGUAGE}-IN"]
    PLAYLIST_META_PATH = _p("data", "playlists", f"{PLAYLIST_ID}.json")        # titles, durations
    SAMPLE_IDS_PATH = _p("data", "playlists", f"{PLAYLIST_ID}_sample.txt")     # glossary sample; keep out of eval
    GLOSSARY_DIR = _p("glossary", PLAYLIST_ID)
    DOMAIN_PACK_PATH = _p("glossary", PLAYLIST_ID, "domain_pack.json")
    ALIAS_TABLE_PATH = _p("glossary", PLAYLIST_ID, "aliases.csv")
    PROTECTED_WORDS_PATH = _p("glossary", PLAYLIST_ID, "protected_words.txt")
    return entry


select_playlist(os.environ.get("PLAYLIST") or None)