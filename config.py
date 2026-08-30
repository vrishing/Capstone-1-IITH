"""
MVP-1 config — edit the values in this file before running anything.
Everything downstream (fetch, chunk, index, ask) reads from here so you
only have to change your corpus definition in one place.
"""

import os
# --- Step 0: pick your pilot corpus -----------------------------------
# 3 YouTube video IDs from ONE playlist. The video ID is the part after
# "v=" in a YouTube URL, e.g. https://www.youtube.com/watch?v=XXXXXXXXXXX
# VIDEO_IDS = [
#     "j9WZyLZCBzs",
#     "TluTv5V0RmE",
#     "19Ql_Q3l0GA",
# ]

# PLAYLIST_ID = ""
# A label for this corpus — this becomes the metadata filter that makes
# retrieval "closed-corpus" instead of open. Pick something stable.
PLAYLIST_ID = "mit-prob"
# PLAYLIST_ID = "mit-prob"

# Preferred transcript language codes to try, in order. YouTube exposes
# both manually-uploaded and auto-generated caption tracks per language;
# the fetch script tries each of these in turn.
TRANSCRIPT_LANGS = ["te", "te-IN", "en"]  # Telugu first, English fallback

# --- Chunking -----------------------------------------------------------
CHUNK_SECONDS = 60          # target chunk length
CHUNK_OVERLAP_SECONDS = 10  # overlap so we don't cut a sentence in half

# --- Embedding ------------------------------------------------------------
# BGE-M3 matches the model chosen in the full architecture. It's a ~2GB
# download the first time you run it. If you want something that downloads
# faster to get moving today, swap in:
#   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# and upgrade to BGE-M3 later — the rest of the code doesn't change.
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # BGE-M3's dense output size; change if you swap models

# --- Vector store ---------------------------------------------------------
# Local, on-disk Qdrant — no server to run for MVP-1. Same client API
# works against a real Qdrant server later with one line changed.
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "transcript_chunks"

# --- Generation -------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-5"
GEMINI_MODEL_NAME = "gemini-3.6-flash"
TOP_K = 5  # how many evidence chunks to retrieve per question


# ASR Ensemble (completely free, self-hosted)
ASR_ENGINES = ["indiconformer", "whisper_indic"]  # both free, open source
INDICONFORMER_ENABLED = True  # AI4Bharat, covers all 22 Indian languages, MIT license
WHISPER_INDIC_ENABLED = True  # OpenAI Whisper + AI4Bharat fine-tune, free fallback
ASR_ENSEMBLE_STRATEGY = "highest_confidence"  # pick IndicConformer or Whisper output with best score
TARGET_LANGUAGE = "te"  # Telugu (change to "sa" for Sanskrit if needed, or "hi" for Hindi)


VERIFIER_MODEL = "gemini-3.6-flash"  # ~$0.00001 per request (cheapest option)
# Alternative: "gemini-1.5-pro" if you need stronger reasoning (still <$0.05 per request)
VERIFIER_ENABLED = True
ABSTAIN_ON_UNSUPPORTED = True  # if True, refuse any answer with unsupported sentences
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ASR_OUTPUT_DIR = "./data/asr_output"