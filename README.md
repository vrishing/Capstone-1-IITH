# Pravachanam RAG

Closed-corpus question answering over 150+ hours of Telugu pravachanam videos from one YouTube playlist. Questions may involve shlokas. Answers come from retrieved transcript evidence, are verified sentence by sentence, and return timestamp links and transcript excerpts (evidence reel later). The code is domain-agnostic: topic, language and trusted knowledge sources live in a domain pack; Mahabharata is the first pack.

Scope, decisions, data contracts and the roadmap are in [SPEC.md](SPEC.md).

## Pipeline

YouTube playlist / channel\
        ↓\
metadata ingestion\
        ↓\
audio extraction / caption extraction\
        ↓\
ASR transcription + timestamp alignment\
        ↓\
Telugu / Sanskrit / Mahabharata term correction\
        ↓\
chunking into timestamped evidence spans\
        ↓\
hybrid index: dense + lexical + entity + timestamp metadata\
        ↓\
closed-corpus retriever\
        ↓\
evidence selector\
        ↓\
LLM answer generator with strict citation requirement\
        ↓\
verifier: every sentence must be supported by retrieved evidence\
        ↓\
answer + timestamp links + transcript excerpts + evidence video reel\

Playlist only for now; channel mode is later.

**Query path (target):** expand question, hybrid search (BGE-M3 dense + sparse), rerank, evidence selector, generate, verify. First cut: expand via the alias table, hybrid search, rerank, generate, verify.

## How it behaves

- **Closed corpus:** applies to retrieval. The knowledge graph only guides retrieval and name normalization; it never supplies statements.
- **Text:** raw and corrected text are both stored, outside the vector DB. The vector DB holds corrected text only.
- **Verifier:** supported answers pass; partially supported answers carry a caution warning; unsupported or verifier-error cases abstain ("I could not find enough support in the specified playlist.").
- **Citations:** `[Video <title or index>, hh:mm:ss–hh:mm:ss]` plus an evidence table.
- **Language:** question and answer in the same language (Telugu or English); answer-language handling is deferred.
- **Interface:** CLI first, then FastAPI, React, Docker.
- **Shlokas:** spans are labeled `shloka_chant`, `quoted_fragment` or `explanation`.

## Status

"Applied" items were changed by the owner after review and not re-run by the reviewer.

**Built and verified**
- [x] ASR: IndicConformer with VAD silence cuts, no overlap, fp32 decode, soundfile loading, per-chunk `segments` (`ASR_TEST.py`). Verified with RNNT; decoder is now CTC (single setting), so re-verify.

**Applied**
- [x] Real per-chunk timestamps used by `06_asr_to_chunks.py` instead of uniform-rate estimates
- [x] `06` fixes: `str(asr_path).replace(...)`, chunk settings read from `config`
- [x] `llm_client.py`: Gemini wrapper (`google-genai`, falls back to `google-generativeai` >= 0.5)

**Scripts present (not verified in this review)**
- [ ] `01_fetch_transcripts.py` (playlist IDs, captions as baseline only), `02_chunk_and_index.py` (dense only), `03_ask.py`, `04_download_audio.py`, `07_reindex_full_corpus.py`

**To retire**
- [ ] `05_asr_ensemble.py` and all Whisper, Sarvam, ensemble and caption-engine settings

**Unbuilt**
- [ ] ASR runner (`--ids`, writes `segments`), `ASR_DECODER` and `ASR_MAX_CHUNK_S` config, remove dead overlap code in `ASR_TEST.py`
- [ ] One downloader (16 kHz mono wav) replacing `01`/`04`/`ASR_TEST.py` download code
- [ ] Manifest (proposed: one row per video per stage with a settings fingerprint; confirm)
- [ ] Alias table, phonetic matching, term correction (pass-through until built)
- [ ] Span builder (45 to 90 s, segment-aligned, no overlap) and shloka labeling
- [ ] BGE-M3 dense + sparse index over corrected text, deterministic UUID point IDs, reranker
- [ ] Query path: alias-table expansion, rerank, verifier (fail closed), citations
- [ ] Evaluation: manual reference (about 30 min), 50 to 100 gold questions including shlokas
- [ ] Domain pack, KG seed (language-independent IDs), cross-video dedup (`cluster_id`)
- [ ] Evidence reel (after rights are cleared)
- [ ] One config name each for embedding and Gemini model settings; remove `getattr` fallbacks

## Dev subset

Development runs on a frozen ~10 hour subset: `data/dev_subset.txt`, one video ID per line, 20 to 30 videos of 15 to 30 min, mixed speakers and audio quality, a few with Telugu captions. `01` and `04` need an `--ids` filter.

## Environments

Primary runs on the GPU server. The laptop (Windows) runs small tests on the dev subset first. Each machine has its own data and manifest; they are not synced.

## Repo layout

Current (kept until the pipeline works):

```
config.py            llm_client.py
01_fetch_transcripts.py   02_chunk_and_index.py   03_ask.py
04_download_audio.py      05_asr_ensemble.py (retire)
06_asr_to_chunks.py       07_reindex_full_corpus.py
ASR_TEST.py          # working ASR; promote into asr/
data/{transcripts,asr_output}/   qdrant_data/
```

Planned after the pipeline works: `raw_audio/`, `transcripts_raw/`, `transcripts_corrected/`, `spans/`, `captions/`, `glossary/`, `clips/`, `manifest.db`.

## Setup

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt                     # google-genai pinned
pip install -r requirements-asr.txt                 # CUDA torch on server, CPU torch on laptop
cp .env.example .env                                # GEMINI_API_KEY, HF_TOKEN; never commit
python llm_client.py                                # prints active Gemini backend
```

ffmpeg must be on PATH. On the server, `google-generativeai` >= 0.5 is enough if `google-genai` cannot be installed. Scripts call `llm_client.generate`, never a Gemini SDK directly. BGE-M3 sparse vectors likely need the `FlagEmbedding` library; verify.

## Design decisions

- IndicConformer only; auto-captions are a baseline, never ground truth.
- Silence-cut, no-overlap ASR chunking, fp32; chunk offsets kept as real timestamps.
- One alias table feeds KG linking and span entity tags; phonetic matching handles variants (Bhishma, Bhisma, Bhishmudu).
- Gemini for the MVP through one seam (`llm_client.py`); self-hosted open-source Indic LLM later.
- Query stages are plain functions with a retry loop; LangGraph optional later.