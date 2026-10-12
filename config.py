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
# PLAYLIST_ID is the real YouTube playlist ID (the full list= value in the URL) and is the
# closed-corpus metadata filter. PLAYLIST_NAME is a human-readable label only.
PLAYLIST_ID = "PLD0OXwSZSSA8" 
PLAYLIST_NAME = "mb-test"
YOUTUBE_PLAYLIST_ID = PLAYLIST_ID  # alias read by 01/04/run_pipeline; remove once 01/04 are retired

TARGET_LANGUAGE = "te"
# YouTube captions are a baseline for comparison only, so Telugu tracks only (no English fallback).
TRANSCRIPT_LANGS = ["te", "te-IN"]

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

# --- Evidence spans (transitional: read by 06) ----------------------------------
# Replaced by SPAN_MIN_S = 45 / SPAN_MAX_S = 90 with no overlap when the span builder is written.
CHUNK_SECONDS = 60
CHUNK_OVERLAP_SECONDS = 10

# --- Embedding and vector store ---------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # BGE-M3 dense size; change if you swap models
QDRANT_PATH = _p("qdrant_data")
COLLECTION_NAME = "transcript_chunks"

# --- Generation and verification (Gemini through llm_client.py) -----------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-3.6-flash"
TOP_K = 10

# Read by the verifier (not built yet).
VERIFIER_MODEL = "gemini-3.6-flash"
VERIFIER_ENABLED = True
ABSTAIN_ON_UNSUPPORTED = True
VERIFIER_ON_PARTIAL = "warn"     # partly supported answers are shown with a caution warning
VERIFIER_ON_ERROR = "abstain"    # fail closed if the verifier itself errors
VERIFIER_MAX_RETRIES = 2
ABSTAIN_MESSAGE = "I could not find enough support in the specified playlist."
