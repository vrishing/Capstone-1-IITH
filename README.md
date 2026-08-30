# MVP-1: Transcript-only QA on 3 pilot videos

This is the smallest possible version of the full architecture that still
proves the core loop works: retrieve evidence from a closed corpus, force
the LLM to cite it, get a timestamped answer back. Everything else in the
thesis architecture (ASR ensemble, glossary correction, hybrid retrieval,
sentence-level verifier, evidence reels) is a deliberate upgrade on top of
this skeleton — build this first so you have something real to upgrade.

**What's deliberately NOT in MVP-1** (these are later stages, don't try to
add them today): the multi-engine ASR ensemble (using YouTube's existing
captions instead), the Sanskrit/Telugu glossary correction layer, sparse
(BM25) and entity indexing (dense-only for now), the sentence-level
verifier (the system prompt is your only faithfulness mechanism today —
that's intentional, and you'll feel exactly why the verifier is needed).

## Before you start

- Pick 3 videos from ONE playlist. **Check they have captions first** —
  open the video on YouTube, click the "..." menu, look for "Show
  transcript." If that option isn't there, pick a different video; you
  don't want to debug ASR today.
- Get an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).
- This code was written and logic-tested (chunking + local vector search)
  in a sandboxed environment without live internet access to YouTube or
  Hugging Face — run the actual fetch/embed/ask steps below on your own
  machine where you have normal internet access.

## Steps to run today

**1. Set up the environment (5 min)**
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your real API key into .env
```

**2. Fill in `config.py` (2 min)**
Open `config.py` and replace `VIDEO_ID_1/2/3` with your 3 real YouTube
video IDs (the string after `watch?v=` in the URL). Leave everything else
at its default for your first run.

**3. Fetch transcripts (2 min)**
```bash
python 01_fetch_transcripts.py
```
You should see one `[ok]` line per video with a caption-line count. If you
get `[FAIL]`, that video has no captions — swap in a different one from
the same playlist and re-run.

**4. Chunk and index (5-15 min, mostly model download time)**
```bash
python 02_chunk_and_index.py
```
First run downloads BGE-M3 (~2GB) — this is the slow part, only happens
once. You should end with `[ok] indexed N chunks into Qdrant`.

**5. Ask it questions (immediate)**
```bash
python 03_ask.py
```
Try, in order:
- A **direct factual** question you know the answer to from those 3
  specific videos — confirm the citation actually points at the right
  video/timestamp.
- A question about something **not in those 3 videos** (even if it's
  real Mahabharata content from elsewhere in the playlist) — confirm it
  says it doesn't have enough information, rather than answering from
  the model's general knowledge. **If it answers anyway, you've just
  reproduced the exact failure mode your verifier is meant to catch** —
  write that example down, it's your best MVP-2 test case.
- A totally unrelated question ("what's the capital of France") — same
  check.

## What "done" looks like today

You should be able to ask a question, get an answer with `[chunk N]`
citations, and click through a printed `youtu.be/...?t=123` link that
actually lands on the right moment. That's the whole MVP-1 bar. Don't
polish beyond this — get it working end to end first, then move to MVP-2.

## Your very next task after this works

Run the "not in those 3 videos" test above a dozen times with different
questions and count how often it fails (answers anyway instead of
abstaining). That failure rate is your baseline "why I need a verifier"
number for the thesis — write it down before you build MVP-2, so you have
a before/after comparison once the verifier exists.

## File map

| File | What it does |
|---|---|
| `config.py` | Every setting lives here — corpus definition, chunk size, model names |
| `01_fetch_transcripts.py` | Pulls YouTube's own captions for your 3 videos |
| `02_chunk_and_index.py` | Merges captions into 60s timestamped chunks, embeds with BGE-M3, stores in local Qdrant |
| `03_ask.py` | Retrieval + citation-constrained generation loop |
