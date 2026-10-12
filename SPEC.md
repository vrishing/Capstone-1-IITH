# SPEC: Pravachanam RAG (scope frozen, decisions recorded)

**Goal:** Closed-corpus Q&A over 150+ hours of Telugu pravachanam videos from one YouTube playlist. Questions may involve shlokas. Every answer must be supported by retrieved evidence and ship with timestamp links, transcript excerpts, and (later) an evidence reel.
**Hardware / environments:** primary run on a 48 GB GPU server (not under owner control). The laptop runs small local tests on the dev subset before server runs. Manifest and data live on the machine that runs the stages; they are not synced between machines.
**Domain-agnostic by design:** topic, language and trusted knowledge sources live in a *domain pack* file, not in code. Mahabharata is the first pack.

## Pipeline (confirmed)

YouTube playlist / channel → metadata ingestion → audio extraction / caption extraction → ASR transcription + timestamp alignment → Telugu / Sanskrit / Mahabharata term correction → chunking into timestamped evidence spans → hybrid index (dense + lexical + entity + timestamp metadata) → closed-corpus retriever → evidence selector → LLM answer generator with strict citation requirement → verifier (every sentence supported by retrieved evidence) → answer + timestamp links + transcript excerpts + evidence video reel

Playlist only for now; channel mode is later.

## Decisions

| Topic | Decision |
|---|---|
| Closed corpus | Applies to retrieval. The knowledge graph (KG) is a guide for retrieval and name normalization only; it never supplies statements. Outside knowledge never appears in answers or counts as support. |
| Deliverable | CLI first; FastAPI, React, Docker afterwards. |
| Rights | Acquired through official communication. Reel rendering waits on it. |
| ASR | IndicConformer only (no Whisper, Sarvam, ensemble, or captions as candidates). Decoder CTC for now (single `ASR_DECODER` setting); the built and verified runs used RNNT, so re-verify on CTC. Auto-captions are a baseline only, never ground truth. |
| Text stored | Raw and corrected text are both kept, outside the vector DB. The vector DB holds corrected text only; retrieval runs over corrected text only. |
| Language | Question and answer in the same language (Telugu or English). Answer-language handling is deferred. |
| Verifier outcomes | Supported: answer. Partially supported: answer with a caution warning. Unsupported: abstain ("I could not find enough support in the specified playlist."). Verifier error: abstain (fail closed). Retry cap and numeric targets set after the first eval baseline. |
| Citation format | As in the proposal: `[Video <title or index>, hh:mm:ss–hh:mm:ss]` plus an evidence table; underlying stable `span_id`. |
| LLM | Gemini via `llm_client.py` for the MVP. Move to a self-hosted open-source Indic LLM once the pipeline is reliable; `llm_client.py` is the seam. |
| Orchestration | Each query stage is a plain function with explicit inputs/outputs plus a simple retry loop. LangGraph is optional at the FastAPI stage. |
| Labels | `playlist_id` (real YouTube ID) and `playlist_name` are separate fields. |
| Noise benchmark | Dropped. Not required. |
| Corpus size | 150+ hours. Plan-doc figures are ignored. The repo is the source of truth; plan-doc code is ignored. |

## Query path

Target (as in the proposal): expand question → hybrid search (BGE-M3 dense + sparse) → rerank → evidence selector → generate → verify.
First cut: expand using the alias table (no LLM), hybrid search, rerank, generate, verify. Add the LLM selector only if topical-only spans cause unsupported claims.
Retrieval is guided by the KG: query terms resolve to entity IDs and aliases to widen search. BM25 and a separate entity index are deferred; spans carry `entities[]` tags now.

## Evidence spans

- Built from whole ASR segments, cut at long pauses, one rule: 45 to 90 s. No overlap within a video.
- Each span stores its segment timestamps, so a hit can be narrowed to the best-matching segments and neighbors fetched for context.
- Each span is labeled `shloka_chant`, `quoted_fragment` or `explanation`.
- Cross-video dedup: store a `cluster_id` for near-duplicate spans (threshold tuned on the dev subset), keep all members, show one representative per cluster with "supported in N discourses", and cap results at 2 to 3 per video.
- Timestamps are chunk-level (seconds, includes ~0.2 s padding) with a stated tolerance; tolerance value set from the eval baseline.

## Shlokas (in scope)

Chant vs quoted-in-explanation is decided from combined signals: share of Sanskrit-looking tokens with few Telugu suffixes, verse shape (about four pādas of eight syllables), audio (slower, sustained, steady pitch, fewer pauses), and context (a run of chant followed by explanation). A contiguous run of at least two pādas counts as a chant. User-typed verses (Devanagari, Telugu or Roman) are normalized to the same phonetic key for matching. Validate on hand-labeled dev-subset examples early.

## Aliases and knowledge graph

- One alias table (glossary CSV) feeds KG entity linking and span `entities[]`. No parallel alias lists.
- Matching is phonetic: transliterate aliases and transcript tokens to a Latin phonetic key, strip Telugu suffixes (-ుడు, -ుని, -ుడి…), mine near-miss variants from dev-subset ASR output, review the most frequent few hundred by hand, mark short or ambiguous variants "needs context". Replacement is whole-word, longest-first.
- Build: seed canonical names from KG sources, mine variants, hand-review, version the CSV and hash it.
- KG: entity IDs and relations are language-independent (e.g. Wikidata Q-IDs); only the alias layer is language-specific. For a new playlist an LLM proposes the domain pack (topic, language, root items, trusted sources, entity types) from the playlist title and sample transcripts; the owner approves it. Entities mined from transcripts are marked "corpus-only".

## Status

"Applied" fixes were made by the owner after review and not re-run by the reviewer. Only ASR was verified by running it.

| Stage | Status | Where |
|---|---|---|
| Metadata / caption ingestion | Partial | `01_fetch_transcripts.py`: playlist IDs + captions (baseline only). yt-dlp listing usually gives title and duration |
| Audio extraction | Script present | `04_download_audio.py` (mp3). Target: one downloader, 16 kHz mono wav |
| ASR (IndicConformer) | **Built, verified (RNNT)** | `ASR_TEST.py`: silence cuts, no overlap, fp32, soundfile loading, returns `segments` |
| Timestamp alignment | Applied | chunker emits real offsets; `06` prefers `segments` |
| Term correction | Unbuilt | pass-through until built; runs before chunking |
| Evidence spans | Script present, fixed | `06`/`02` use time windows; to be replaced by the span builder above |
| Index | Dense only | BGE-M3 dense + Qdrant in `02`/`07`; sparse, reranker unbuilt |
| Gemini access | Applied | `llm_client.py` (`google-genai`, falls back to `google-generativeai` >= 0.5) |
| Retriever + generator | Script present | `03_ask.py`: no expansion, rerank, or verifier |
| Verifier | Unbuilt | must use `llm_client.generate` and fail closed |
| Manifest, dedup, KG, reel, shloka labeling | Unbuilt | |

## Data contracts

- **Manifest (proposed, confirm):** one row per video per stage: `video_id, stage, status, fingerprint, updated_at, error`. The fingerprint holds the settings used (ASR decoder and chunk cap, glossary hash, embedding model). If it differs from current settings, that video reruns from that stage onward.
- **ASR segment:** `video_id, start_s, end_s, text_raw` (stored as `start`, `duration`, `text` in `segments`).
- **Evidence span:** `span_id, playlist_id, video_id, start_s, end_s, segment_times[], text_raw, text_corrected, entities[], label, cluster_id`. The vector DB payload excludes `text_raw`.
- **Qdrant point ID:** UUID derived from `span_id` (deterministic).

## Roadmap (one sequence)

1. **Foundations:** ASR runner (`--ids`, writes `segments`), `ASR_MAX_CHUNK_S` and `ASR_DECODER` in config, remove dead overlap code in `ASR_TEST.py`, one downloader, deterministic IDs, manifest.
2. **Text pipeline:** alias table + phonetic correction, span builder with shloka labels, BGE-M3 dense + sparse index over corrected text.
3. **Query path (CLI):** expand, hybrid search, rerank, generate (Gemini), verify, citations.
4. **Eval:** manual reference (about 30 min across 3 to 4 videos and speakers) for WER; 50 to 100 gold questions with answer timestamps, including shloka questions; thresholds from the first baseline.
5. **Guidance and dedup:** domain pack, KG seed, `cluster_id` dedup.
6. **Later:** evidence reel (after rights), FastAPI/React/Docker, move to the proposal's folder split, channel mode, incremental updates, self-hosted Indic LLM, answer-language handling, LLM selector if needed.

## Reconcile in code

1. Config: one name each for the embedding and Gemini model settings (`EMBED_MODEL_NAME`, `GEMINI_MODEL`); remove `getattr` fallbacks in `03_ask.py`.
2. Remove Whisper/ensemble/caption-engine settings and `05_asr_ensemble.py`.
3. `01` and `04` honor `--ids data/dev_subset.txt`.
4. No script imports a Gemini SDK directly (`grep -rn "generativeai\|google import genai" .`).
5. Pin `google-genai`; ASR packages in a separate `requirements-asr.txt`.

## Dev subset (frozen)

About 10 hours in `data/dev_subset.txt`, one video ID per line, never edited mid-build; 20 to 30 videos of 15 to 30 min, multiple speakers and audio quality, a few with Telugu captions (baseline only). Each stage must run on it in minutes.

## Out of scope for now

Other ASR engines, hosted GPU rental, graph DB server, Tier 2 KG triple extraction, channel mode, answer-language handling, noise benchmarking, folder-layout migration.

## Open

Timestamp tolerance value; confirm the per-stage manifest design; CTC re-verification; verifier retry cap and numeric targets.