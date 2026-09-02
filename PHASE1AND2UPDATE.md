# Closed-Corpus Video QA — Build Plan (Phase 1 → Phase 2)

This is the combined build plan for the project: retrieve evidence from a
closed video-transcript corpus, force the LLM to cite it, get a
timestamped answer back — starting from a 3-video pilot (Phase 1 / MVP-1)
and scaling to a full 50+ hour playlist with free self-hosted ASR and a
Gemini verifier (Phase 2).

Build Phase 1 first. It's the smallest version of the architecture that
still proves the core loop works. Everything in Phase 2 (ASR ensemble,
full-playlist ingestion, sentence-level verifier) is a deliberate upgrade
on top of that skeleton.

> **Merge notes — inconsistencies fixed while combining these docs:**
> - **`PLAYLIST_ID` naming collision.** Phase 1's `config.py` already uses
>   `PLAYLIST_ID` as a stable *corpus label* (metadata filter, e.g.
>   `"mit-prob"`) — it isn't fetched from anywhere. The original Phase 2
>   doc told you to overwrite that same variable with a real YouTube
>   `PLxxxxx` playlist ID for downloading audio. That would silently break
>   the Phase 1 metadata filter. Fixed by introducing a new variable,
>   `YOUTUBE_PLAYLIST_ID`, for the fetchable playlist, leaving `PLAYLIST_ID`
>   as the corpus label.
> - **Gemini model name drift.** The Phase 2 doc hardcoded
>   `gemini-2.0-flash` everywhere, but the actual `config.py` already
>   defines `GEMINI_MODEL_NAME` / `VERIFIER_MODEL = "gemini-3.6-flash"`.
>   Fixed by having every script read the model name from `config.py`
>   instead of hardcoding a string, so there's one source of truth.
> - **API key env var mismatch.** `config.py` reads
>   `os.environ.get("GEMINI_API_KEY")`, but the Phase 2 doc's code and
>   `.env` instructions used `GOOGLE_API_KEY` — two different env vars
>   pointing at the same key. Fixed to use `GEMINI_API_KEY` everywhere, and
>   scripts now read `config.GEMINI_API_KEY` instead of re-reading the
>   environment directly.
> - **Wrong Gemini SDK.** `requirements.txt` already pins `google-genai`
>   (the current unified SDK), but the Phase 2 doc's code imported the
>   older `google-generativeai` package (`import google.generativeai as
>   genai`) with a different client API. Fixed all Gemini calls to use
>   `from google import genai` / `genai.Client(...)`, matching
>   `requirements.txt`, and dropped the redundant `pip install
>   google-generativeai` step.
> - **Unimportable module name.** Step 7 tried
>   `from 08_verifier import verify_answer` — Python module names can't
>   start with a digit, so this would raise a `SyntaxError`. Fixed by
>   naming the file `verifier.py` (still built in "Step 6" of the
>   checklist) instead of `08_verifier.py`.
> - **Missing dependency.** The `IndicConformerASR` class calls
>   `import librosa`, but the Step 3 install command never installs it.
>   Added it to the `pip install` line.
> - **Chunking params not read from config.** `06_asr_to_chunks.py` called
>   `merge_into_chunks(entries)` and relied on the function's hardcoded
>   defaults (`60`/`10`) instead of `config.CHUNK_SECONDS` /
>   `config.CHUNK_OVERLAP_SECONDS`, which `config.py`'s own docstring says
>   is the single place chunking settings should live. Fixed to pass them
>   explicitly.
> - **Stale ASR-vendor gotchas.** The Troubleshooting section had leftover
>   entries about "Sarvam API timing out" and "WhisperX too slow" — vendors
>   that don't appear anywhere in the actual Phase 2 scripts, which use
>   IndicConformer (via Hugging Face `transformers`) and `faster-whisper`.
>   Replaced with gotchas for the tools actually used.
> - **`03_ask_gemini.py` described as "modify" with no origin.** Step 7
>   said to modify a file that Step 1–6 never created.
>
> **Updated after seeing the actual `01_fetch_transcripts.py` /
> `02_chunk_and_index.py` / `03_ask.py`:**
> - **`03_ask.py` is already Gemini-based, not Claude.** It imports
>   `from google import genai`, reads `GEMINI_API_KEY` directly, and picks
>   its model via `getattr(config, "GEMINI_MODEL", getattr(config,
>   "GEMINI_MODEL_NAME", "gemini-2.5-flash"))` — so it already resolves to
>   `config.GEMINI_MODEL_NAME` (`"gemini-3.6-flash"`). There is no
>   Claude-based ask script anywhere in the codebase; `config.ANTHROPIC_MODEL`
>   is currently **unused** by any script (see note in Prerequisites).
>   Because of this, creating a separate `03_ask_gemini.py` no longer makes
>   sense — Step 7 below now adds the verifier gate directly on top of
>   `03_ask.py`'s existing retrieval + Gemini generation, in a new file
>   called `09_ask_verified.py`, keeping `03_ask.py` untouched as the
>   "no verifier" baseline Step 8 compares against.
> - **Verifier would have been called with the wrong shape.** `03_ask.py`'s
>   `retrieve()` returns Qdrant `ScoredPoint` objects (`r.payload`,
>   `r.score`), but `verifier.py`'s `verify_answer()` expects a plain
>   `list[dict]` with a `"text"` key (it calls `c.get('text', '')`).
>   Passing `results` straight through would throw, since `ScoredPoint`
>   has no `.get()`. Fixed by passing `[r.payload for r in results]`.

---

## Prerequisites

**For Phase 1 (do these first):**
- [ ] Pick 3 videos from **one** playlist. Check they have captions first —
  open the video on YouTube, click the "..." menu, look for "Show
  transcript." If that option isn't there, pick a different video; you
  don't want to debug ASR today.
- [ ] Get a Gemini API key (free tier) at [ai.google.dev](https://ai.google.dev) — no credit card needed. `03_ask.py` calls Gemini directly (`GEMINI_API_KEY`, `config.GEMINI_MODEL_NAME`), so this key is required from the very first run of Phase 1, not just Phase 2.
- [ ] An Anthropic API key from [console.anthropic.com](https://console.anthropic.com) is *not* currently required — `config.py` defines `ANTHROPIC_MODEL = "claude-sonnet-5"`, but no script in the repo references it. Get one only if/when you add a Claude-based ask script for comparison.
- Note: this code was written and logic-tested (chunking + local vector search) in a sandboxed environment without live internet access to YouTube or Hugging Face — run the actual fetch/embed/ask steps on your own machine with normal internet access.

**Additional prerequisites before starting Phase 2:**
- [ ] A YouTube playlist ID (the `PLxxxxx` string from the URL) with 50+ videos
- [ ] `yt-dlp` installed locally: `pip install yt-dlp`
- [ ] `ffmpeg` installed: `ffmpeg -version`. If not: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux)
- [ ] Disk space: ~100 GB for 50+ hours of MP3s + ASR output (temporary; delete after indexing)
- [ ] GPU recommended for ASR (3–4x speedup), but CPU works (slower, ~4–6x realtime)
- [ ] Patience: first ASR run on 50 videos takes 4–8 hours on CPU, 1–2 hours on GPU

Phase 2's extra Python packages (`faster-whisper`, `transformers`,
`torch`, `torchaudio`, `librosa`, `pydub`, `yt-dlp`, optionally
`nemo-toolkit`) are intentionally **not** in the base `requirements.txt` —
they're heavy (multi-GB) and only needed once you're running the free ASR
path. Install them when you get to Phase 2 Step 3.

---

## Phase 1 (MVP-1): Transcript-only QA on 3 pilot videos

This is the smallest possible version of the full architecture that still
proves the core loop works: retrieve evidence from a closed corpus, force
the LLM to cite it, get a timestamped answer back. Everything else in the
full architecture (ASR ensemble, glossary correction, hybrid retrieval,
sentence-level verifier, evidence reels) is a deliberate upgrade on top of
this skeleton.

**What's deliberately NOT in Phase 1** (these are later stages, don't try
to add them today): the multi-engine ASR ensemble (using YouTube's
existing captions instead), the Sanskrit/Telugu glossary correction layer,
sparse (BM25) and entity indexing (dense-only for now), the sentence-level
verifier (the system prompt is your only faithfulness mechanism today —
that's intentional, and you'll feel exactly why the verifier is needed in
Phase 2).

### Steps to run today

**1. Set up the environment (5 min)**
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your real API keys into .env
```

**2. Fill in `config.py` (2 min)**
Open `config.py`. Uncomment `VIDEO_IDS` and replace the placeholders with
your 3 real YouTube video IDs (the string after `watch?v=` in the URL).
`PLAYLIST_ID` is already set to a stable corpus label (`"mit-prob"`) — leave
it as your metadata filter, not a real playlist ID (that comes in Phase 2
as a separate variable). Leave everything else at its default for your
first run.

**3. Fetch transcripts (2 min)**
```bash
python 01_fetch_transcripts.py
```
You should see one `[ok]` line per video with a caption-line count. If you
get `[FAIL]`, that video has no captions — swap in a different one from
the same playlist and re-run.

**4. Chunk and index (5–15 min, mostly model download time)**
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
- A question about something **not in those 3 videos** (even if it's real
  content from elsewhere in the playlist) — confirm it says it doesn't
  have enough information, rather than answering from the model's general
  knowledge. **If it answers anyway, you've just reproduced the exact
  failure mode your verifier is meant to catch** — write that example
  down, it's your best Phase 2 test case.
- A totally unrelated question ("what's the capital of France") — same
  check.

### What "done" looks like today

You should be able to ask a question, get an answer with `[chunk N]`
citations, and click through a printed `youtu.be/...?t=123` link that
actually lands on the right moment. That's the whole Phase 1 bar. Don't
polish beyond this — get it working end to end first, then move to Phase 2.

### Your very next task after this works

Run the "not in those 3 videos" test above a dozen times with different
questions and count how often it fails (answers anyway instead of
abstaining). That failure rate is your baseline "why I need a verifier"
number — write it down before you build Phase 2, so you have a
before/after comparison once the verifier exists.

---

## Transitioning to Phase 2

Before touching new code, reconcile `config.py` with what Phase 2 needs:

- [ ] **Add `YOUTUBE_PLAYLIST_ID`** — a new variable, separate from
  `PLAYLIST_ID` (which stays as your corpus metadata label):
  ```python
  YOUTUBE_PLAYLIST_ID = "PLxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # the real playlist to download
  ```
- [ ] **Confirm the ASR/verifier block already in `config.py` matches
  what's below** (most of it should already be there from Phase 1's
  template):
  ```python
  # ASR Ensemble (completely free, self-hosted)
  ASR_ENGINES = ["indiconformer", "whisper_indic"]  # both free, open source
  INDICONFORMER_ENABLED = True   # AI4Bharat, covers all 22 Indian languages, MIT license
  WHISPER_INDIC_ENABLED = True   # OpenAI Whisper + AI4Bharat fine-tune, free fallback
  ASR_ENSEMBLE_STRATEGY = "highest_confidence"  # pick IndicConformer or Whisper output with best score
  TARGET_LANGUAGE = "te"  # Telugu (change to "sa" for Sanskrit, or "hi" for Hindi)

  VERIFIER_MODEL = "gemini-3.6-flash"   # matches GEMINI_MODEL_NAME; cheapest current tier
  VERIFIER_ENABLED = True
  ABSTAIN_ON_UNSUPPORTED = True  # if True, refuse any answer with unsupported sentences
  GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

  ASR_OUTPUT_DIR = "./data/asr_output"
  ```
- [ ] **Use one env var, `GEMINI_API_KEY`**, everywhere — in `.env`, in
  `config.py`, and in every script that calls Gemini. Don't introduce a
  second `GOOGLE_API_KEY` that points at the same value.
- [ ] **Never hardcode a Gemini model string in a script.** Always read
  `config.GEMINI_MODEL_NAME` (generation) or `config.VERIFIER_MODEL`
  (verification) so there's one place to bump the model later.

**Testing:**
```bash
python -c "import config; print(f'Corpus label: {config.PLAYLIST_ID}'); print(f'YouTube playlist: {config.YOUTUBE_PLAYLIST_ID}'); print(f'Verifier: {config.VERIFIER_ENABLED}')"
```
Should print your corpus label, your real playlist ID, and confirm the
verifier is enabled.

---

## Phase 2: Full Playlist + Free Indic ASR + Gemini Verifier

**Scope:** Take Phase 1 (3 videos, YouTube captions) and upgrade to a
production-ready closed-corpus system on your full playlist using
**completely free, self-hosted ASR** and a **Gemini verifier**. Skip
glossary, BM25, entity indexing — those are Phase 3.

**Outcome:** A system that can ingest a 50–100 hour playlist, transcribe it
with an open-source ASR ensemble (IndicConformer + Whisper), and generate
answers that must pass a Gemini-based sentence-level verifier before
reaching the user.

**Timeline:** 4–5 weeks, 3–4 hours/week if working part-time.

### Cost breakdown: why this matters

| Component | Original plan | Phase 2 (this plan) |
|---|---|---|
| ASR (50+ hours) | $1000–2000 (paid ASR API at scale) | $0 (open source, self-hosted) |
| Verifier (1000+ questions) | $50–100 (a frontier-tier model at scale) | ~$5–10 (Gemini flash-tier is far cheaper) |
| **Total for full corpus work** | **$1050–2100** | **~$5–10** |

By using free, self-hosted ASR and a cheap Gemini tier for verification
instead of paid APIs at scale, you save well over $1000 on the same work.

---

### Step 1: Extend `config.py` for the full playlist

**File to modify:** `config.py`

**Checklist:**
- [ ] Add `YOUTUBE_PLAYLIST_ID` (see "Transitioning to Phase 2" above) —
  don't overwrite `PLAYLIST_ID`, which is already your corpus label.
- [ ] Confirm the ASR + verifier config block is present and uses
  `GEMINI_API_KEY` (not `GOOGLE_API_KEY`) and `VERIFIER_MODEL =
  "gemini-3.6-flash"` (matching `GEMINI_MODEL_NAME`, not a hardcoded
  `gemini-2.0-flash`).
- [ ] Keep everything else from Phase 1 (embedding model, Qdrant,
  chunking parameters, retrieval `TOP_K`) unchanged.

**Testing:**
```bash
python -c "import config; print(f'Playlist: {config.YOUTUBE_PLAYLIST_ID}'); print(f'Verifier: {config.VERIFIER_ENABLED}')"
```
Should print your playlist ID and confirm the verifier is enabled.

---

### Step 2: Download playlist audio

**New file to create:** `04_download_audio.py`

**What it does:** Fetches all videos from the playlist and converts to MP3.

**Checklist:**
- [ ] Create the script with this structure:
  ```python
  import os
  from yt_dlp import YoutubeDL

  import config

  def get_playlist_videos(playlist_id: str) -> list[str]:
      """Fetch all video IDs from a playlist."""
      ydl_opts = {
          'quiet': True,
          'no_warnings': True,
          'extract_flat': 'in_playlist',
      }
      with YoutubeDL(ydl_opts) as ydl:
          result = ydl.extract_info(f'https://www.youtube.com/playlist?list={playlist_id}', download=False)
          return [entry['id'] for entry in result.get('entries', [])]

  def download_audio(video_id: str, output_dir: str) -> str:
      """Download one video as MP3."""
      out_path = os.path.join(output_dir, f"{video_id}.mp3")
      if os.path.exists(out_path):
          print(f"[skip] {video_id} already downloaded")
          return out_path

      ydl_opts = {
          'format': 'bestaudio/best',
          'postprocessors': [{
              'key': 'FFmpegExtractAudio',
              'preferredcodec': 'mp3',
              'preferredquality': '192',
          }],
          'outtmpl': os.path.join(output_dir, f"{video_id}"),
          'quiet': False,
      }
      try:
          with YoutubeDL(ydl_opts) as ydl:
              ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
          print(f"[ok] {video_id} downloaded")
          return out_path
      except Exception as e:
          print(f"[FAIL] {video_id}: {e}")
          return None

  def main():
      os.makedirs(config.ASR_OUTPUT_DIR, exist_ok=True)
      print(f"Fetching playlist {config.YOUTUBE_PLAYLIST_ID}...")
      video_ids = get_playlist_videos(config.YOUTUBE_PLAYLIST_ID)
      print(f"Found {len(video_ids)} videos")

      for i, vid in enumerate(video_ids, start=1):
          print(f"[{i}/{len(video_ids)}] downloading {vid}...")
          download_audio(vid, config.ASR_OUTPUT_DIR)

  if __name__ == "__main__":
      main()
  ```
- [ ] Test on just the first 3 videos before running the full playlist
  (temporarily slice `video_ids` down to `video_ids[:3]` in `main()`):
  ```bash
  python 04_download_audio.py
  ```
- [ ] Verify MP3 files exist:
  ```bash
  ls -lh data/asr_output/*.mp3 | head
  ```

**Gotchas:**
- [ ] `ffmpeg` must be installed (see Prerequisites).
- [ ] The first run takes a while (roughly one video per minute on a
  typical connection).
- [ ] Don't run on the full playlist until you've tested on 3 videos.

**Success criteria:**
- [ ] 3+ MP3 files in `data/asr_output/`
- [ ] Each file is >10MB (confirms it's actual audio, not a corrupt download)
- [ ] Next step: run the full playlist (will take hours, possibly overnight)

---

### Step 3: Run the free ASR ensemble on all audio

**New file to create:** `05_asr_ensemble.py`

**What it does:** Runs IndicConformer (AI4Bharat) + Whisper (via
`faster-whisper`, using an AI4Bharat Indic fine-tune) on all audio files,
picks the best output by confidence score. $0 in API costs — everything
runs locally.

**Checklist:**
- [ ] Install ASR dependencies:
  ```bash
  pip install --break-system-packages faster-whisper transformers torchaudio torch librosa
  # For IndicConformer, also install AI4Bharat NeMo (optional; simpler approach uses HuggingFace):
  pip install --break-system-packages nemo-toolkit
  ```
- [ ] Create the script:
  ```python
  import os
  import json
  from pathlib import Path
  from faster_whisper import WhisperModel
  import torch

  import config

  class IndicConformerASR:
      """AI4Bharat's IndicConformer — free, open-source, covers all 22 Indian languages."""
      def __init__(self, language_code="te"):
          # Load from HuggingFace (simpler than NeMo setup)
          self.language_code = language_code
          try:
              from transformers import AutoModelForCTC, AutoProcessor
              model_id = "ai4bharat/indic-conformer-600m-multilingual"
              self.processor = AutoProcessor.from_pretrained(model_id)
              self.model = AutoModelForCTC.from_pretrained(model_id)
              device = "cuda" if torch.cuda.is_available() else "cpu"
              self.model = self.model.to(device)
              self.device = device
          except Exception as e:
              print(f"[Warning] IndicConformer load failed: {e}. Fallback to Whisper only.")
              self.model = None

      def transcribe(self, audio_path: str) -> dict:
          """Run IndicConformer."""
          if self.model is None:
              return {'text': '', 'confidence': 0.0, 'engine': 'indiconformer', 'error': 'model_load_failed'}

          try:
              import librosa
              audio, sr = librosa.load(audio_path, sr=16000)
              inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
              with torch.no_grad():
                  outputs = self.model(**inputs)
              logits = outputs.logits
              predicted_ids = torch.argmax(logits, dim=-1)
              transcription = self.processor.batch_decode(predicted_ids)[0]
              return {
                  'text': transcription,
                  'confidence': 0.8,  # IndicConformer doesn't expose confidence easily
                  'engine': 'indiconformer'
              }
          except Exception as e:
              print(f"[IndicConformer error] {e}")
              return {'text': '', 'confidence': 0.0, 'engine': 'indiconformer', 'error': str(e)}

  class WhisperIndic:
      """faster-whisper + AI4Bharat fine-tune for Indic languages — free, fast."""
      def __init__(self, language_code="te"):
          self.language_code = language_code
          device = "cuda" if torch.cuda.is_available() else "cpu"
          # Default to vasista22's Telugu fine-tune, available on HuggingFace
          model_id = "vasista22/whisper-telugu-large-v2" if language_code == "te" else "openai/whisper-large-v2"
          self.model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "float32")

      def transcribe(self, audio_path: str) -> dict:
          """Run Whisper locally."""
          try:
              segments, info = self.model.transcribe(audio_path, language=self.language_code)
              text = " ".join([seg.text for seg in segments])
              # Whisper doesn't expose per-utterance confidence; assume 0.75 if it ran
              return {
                  'text': text,
                  'confidence': 0.75,
                  'engine': 'whisper_indic'
              }
          except Exception as e:
              print(f"[Whisper error] {e}")
              return {'text': '', 'confidence': 0.0, 'engine': 'whisper_indic', 'error': str(e)}

  def transcribe_one_video(video_id: str, audio_path: str, indiconformer: IndicConformerASR, whisper: WhisperIndic) -> dict:
      """Run ensemble on one video, pick highest-confidence output."""
      print(f"  [transcribing] {video_id}...")

      outputs = []
      if config.INDICONFORMER_ENABLED:
          outputs.append(indiconformer.transcribe(audio_path))
      if config.WHISPER_INDIC_ENABLED:
          outputs.append(whisper.transcribe(audio_path))

      if not outputs or all(o.get('text', '') == '' for o in outputs):
          print(f"    [WARNING] both engines failed for {video_id}")
          return {'video_id': video_id, 'text': '', 'engine_used': 'none', 'confidence': 0.0, 'all_outputs': outputs}

      # Pick the one with highest confidence (and non-empty text)
      valid_outputs = [o for o in outputs if o.get('text', '') != '']
      best = max(valid_outputs, key=lambda x: x['confidence']) if valid_outputs else outputs[0]

      return {
          'video_id': video_id,
          'text': best['text'],
          'engine_used': best['engine'],
          'confidence': best['confidence'],
          'all_outputs': outputs  # for debugging
      }

  def main():
      os.makedirs(config.ASR_OUTPUT_DIR, exist_ok=True)

      print("Initializing free ASR engines...")
      print(f"  IndicConformer: {config.INDICONFORMER_ENABLED}")
      print(f"  Whisper Indic: {config.WHISPER_INDIC_ENABLED}")

      indiconformer = IndicConformerASR(config.TARGET_LANGUAGE) if config.INDICONFORMER_ENABLED else None
      whisper = WhisperIndic(config.TARGET_LANGUAGE) if config.WHISPER_INDIC_ENABLED else None

      audio_files = sorted(Path(config.ASR_OUTPUT_DIR).glob("*.mp3"))
      print(f"Found {len(audio_files)} audio files to transcribe")
      print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU (will be slower)'}\n")

      for i, audio_path in enumerate(audio_files, start=1):
          video_id = audio_path.stem
          print(f"[{i}/{len(audio_files)}] {video_id}...", end='', flush=True)
          result = transcribe_one_video(video_id, str(audio_path), indiconformer, whisper)

          out_path = os.path.join(config.ASR_OUTPUT_DIR, f"{video_id}_asr.json")
          with open(out_path, 'w', encoding='utf-8') as f:
              json.dump(result, f, ensure_ascii=False, indent=2)

          print(f" {len(result['text'].split())} words, engine={result['engine_used']}")

  if __name__ == "__main__":
      main()
  ```
- [ ] Test on the first 3 videos you downloaded (this will download model
  weights on first run):
  ```bash
  python 05_asr_ensemble.py
  ```
  Should produce `data/asr_output/*_asr.json` files with transcription results.
- [ ] Spot-check one result:
  ```bash
  cat data/asr_output/VIDEO_ID_asr.json | jq '.text' | head -c 200
  ```
  Should see Telugu text (native script).

**Gotchas:**
- [ ] First run downloads models (~5–10 GB for both engines combined).
  This happens once, takes 10–30 min.
- [ ] On CPU, this is slow: IndicConformer ~3–4x realtime, Whisper
  ~2–3x realtime. For 50 hours, expect 150–200 hours of compute on CPU.
  GPU cuts this to 10–20 hours.
- [ ] If you don't have a GPU and CPU speed is blocking, skip
  IndicConformer (`INDICONFORMER_ENABLED = False`) and use Whisper only —
  it's faster and still solid for Telugu.
- [ ] No API costs, ever — all models are free, open-source, self-hosted.

**Success criteria:**
- [ ] All audio files produce `*_asr.json` output files
- [ ] Each output has a `text` field with >100 words
- [ ] One pass through the full playlist completed (CPU: overnight run, GPU: 1–2 hours)

---

### Step 4: Convert ASR output to timestamped chunks

**New file to create:** `06_asr_to_chunks.py`

**What it does:** Takes raw ASR JSON (which has no timestamps, since it's
full-audio transcription), estimates timestamps by spreading the text
across the audio duration, then chunks it using the **same
`config.CHUNK_SECONDS` / `config.CHUNK_OVERLAP_SECONDS` settings as Phase
1** — don't hardcode different values here.

**Checklist:**
- [ ] Create the script:
  ```python
  import os
  import json
  from pathlib import Path
  from pydub import AudioSegment

  import config

  def get_audio_duration(audio_path: str) -> float:
      """Get duration of MP3 in seconds."""
      audio = AudioSegment.from_mp3(audio_path)
      return len(audio) / 1000.0

  def estimate_timestamps(text: str, duration_seconds: float) -> list[dict]:
      """
      Rough timestamp estimation: assume uniform speech rate.
      Split text into words, spread evenly across duration.
      """
      words = text.split()
      if not words:
          return []

      words_per_second = len(words) / duration_seconds
      entries = []
      current_time = 0.0

      for word in words:
          word_duration = 1.0 / words_per_second  # rough estimate
          entries.append({
              'text': word,
              'start': current_time,
              'duration': word_duration
          })
          current_time += word_duration

      return entries

  def merge_into_chunks(entries: list[dict], chunk_seconds: int, overlap_seconds: int):
      """Same chunking logic as Phase 1 — always called with config.CHUNK_SECONDS /
      config.CHUNK_OVERLAP_SECONDS so both phases share one source of truth."""
      if not entries:
          return []
      chunks = []
      window_start_idx = 0
      n = len(entries)

      while window_start_idx < n:
          window_start_time = entries[window_start_idx]['start']
          text_parts = []
          end_idx = window_start_idx
          end_time = window_start_time

          while end_idx < n and (entries[end_idx]['start'] - window_start_time) < chunk_seconds:
              text_parts.append(entries[end_idx]['text'])
              end_time = entries[end_idx]['start'] + entries[end_idx]['duration']
              end_idx += 1

          chunks.append({
              'start': round(window_start_time, 2),
              'end': round(end_time, 2),
              'text': ' '.join(text_parts).strip(),
          })

          next_start_time = window_start_time + (chunk_seconds - overlap_seconds)
          next_idx = window_start_idx
          while next_idx < n and entries[next_idx]['start'] < next_start_time:
              next_idx += 1
          if next_idx <= window_start_idx:
              next_idx = window_start_idx + 1
          window_start_idx = next_idx

      return [c for c in chunks if c['text']]

  def main():
      asr_files = sorted(Path(config.ASR_OUTPUT_DIR).glob("*_asr.json"))
      print(f"Processing {len(asr_files)} ASR outputs into timestamped chunks...")

      for asr_path in asr_files:
          video_id = asr_path.stem.replace('_asr', '')
          audio_path = os.path.join(config.ASR_OUTPUT_DIR, f"{video_id}.mp3")

          if not os.path.exists(audio_path):
              print(f"[skip] {video_id}: no audio file")
              continue

          with open(asr_path, encoding='utf-8') as f:
              asr_data = json.load(f)

          duration = get_audio_duration(audio_path)
          entries = estimate_timestamps(asr_data['text'], duration)
          chunks = merge_into_chunks(entries, config.CHUNK_SECONDS, config.CHUNK_OVERLAP_SECONDS)

          out_path = str(asr_path).replace('_asr.json', '_chunks.json')
          with open(out_path, 'w', encoding='utf-8') as f:
              json.dump({
                  'video_id': video_id,
                  'asr_engine': asr_data.get('engine_used', 'unknown'),
                  'duration_seconds': duration,
                  'num_chunks': len(chunks),
                  'chunks': chunks
              }, f, ensure_ascii=False, indent=2)

          print(f"[ok] {video_id}: {len(entries)} words -> {len(chunks)} chunks")

  if __name__ == "__main__":
      main()
  ```
- [ ] Install pydub: `pip install --break-system-packages pydub`
- [ ] Run on your 3 test videos:
  ```bash
  python 06_asr_to_chunks.py
  ```
- [ ] Spot-check output:
  ```bash
  cat data/asr_output/VIDEO_ID_chunks.json | jq '.chunks[0]'
  ```
  Should see `{start, end, text}` with realistic timestamp ranges.

**Gotchas:**
- [ ] Timestamp estimation is rough — uniform speech rate means
  timestamps are approximate, not precise. Fine for Phase 2; precise
  timestamps come from an alignment tool in Phase 3.
- [ ] `pydub` requires `ffmpeg` (same as Step 2).

**Success criteria:**
- [ ] All videos have `*_chunks.json` files
- [ ] Chunk count looks reasonable (50–100 videos × 40–60 chunks each ≈ 2000–6000 total chunks)

---

### Step 5: Re-index the full corpus into Qdrant with ASR data

**New file to create:** copy `02_chunk_and_index.py` → `07_reindex_full_corpus.py`

**Checklist:**
- [ ] Copy `02_chunk_and_index.py` to `07_reindex_full_corpus.py`
- [ ] Modify it to read from `*_chunks.json` instead of YouTube captions:
  ```python
  def load_chunks_from_asr():
      """Load all ASR-derived chunks from data/asr_output/"""
      chunks_data = []
      for chunks_path in sorted(Path(config.ASR_OUTPUT_DIR).glob("*_chunks.json")):
          with open(chunks_path, encoding='utf-8') as f:
              data = json.load(f)
          chunks_data.extend(data['chunks'])
      return chunks_data
  ```
- [ ] Keep the embedding and Qdrant storage logic the same, including
  tagging each point with `config.PLAYLIST_ID` as the corpus-label
  metadata field (unchanged from Phase 1).
- [ ] Delete the old Qdrant collection first, so you don't mix Phase 1
  YouTube-caption chunks with Phase 2 ASR chunks:
  ```bash
  rm -rf ./qdrant_data
  ```
- [ ] Run the re-indexing:
  ```bash
  python 07_reindex_full_corpus.py
  ```
  Expected output: something like `[ok] indexed 3000 chunks into Qdrant`

**Testing:**
```bash
python -c "from qdrant_client import QdrantClient; c = QdrantClient('./qdrant_data'); print(f'Points in collection: {c.get_collection(\"transcript_chunks\").points_count}')"
```

**Success criteria:**
- [ ] Qdrant collection has 2000+ points (one per chunk)
- [ ] Collection rebuilds from scratch in <10 min on modern hardware

---

### Step 6: Implement the sentence-level verifier (Gemini)

**New file to create:** `verifier.py`
*(Not `08_verifier.py` — Python module names can't start with a digit,
and Step 7 needs to `import` this file, not just run it as a script.)*

**What it does:** Takes a generated answer and the retrieved chunks, scores
each sentence as SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED using
Gemini. Cost is a fraction of a cent per verification.

**Checklist:**
- [ ] No new package install needed — `google-genai` is already in
  `requirements.txt` from Phase 1.
- [ ] Create the verifier script:
  ```python
  import json
  from google import genai

  import config

  VERIFIER_SYSTEM_PROMPT = """You are a fact-checker. Given an answer and a list of \
  evidence chunks it cites, rate each sentence of the answer as:
  - SUPPORTED: the claim is directly stated or clearly entailed by at least one cited chunk
  - PARTIALLY_SUPPORTED: mostly true but missing nuance, or supported only by inference
  - UNSUPPORTED: not found in chunks, or contradicts them

  Output ONLY valid JSON (no preamble):
  {
    "sentences": [
      {"text": "first sentence", "status": "SUPPORTED", "reasoning": "found in chunk 1"},
      {"text": "second sentence", "status": "UNSUPPORTED", "reasoning": "no evidence"}
    ],
    "overall_unsupported_count": 1,
    "recommendation": "ACCEPT or REFUSE"
  }
  """

  def verify_answer(answer_text: str, retrieved_chunks: list[dict]) -> dict:
      """Call Gemini to verify an answer against retrieved chunks."""
      client = genai.Client(api_key=config.GEMINI_API_KEY)

      # Format chunks for the verifier
      chunks_str = "\n\n".join([
          f"[chunk {i+1}] {c.get('text', '')}"
          for i, c in enumerate(retrieved_chunks)
      ])

      user_message = f"""Answer to verify:
  {answer_text}

  Evidence chunks:
  {chunks_str}

  Verify each sentence."""

      try:
          response = client.models.generate_content(
              model=config.VERIFIER_MODEL,
              contents=f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_message}",
          )
          result_text = response.text
          # Extract JSON if wrapped in markdown code blocks
          if "```json" in result_text:
              result_text = result_text.split("```json")[1].split("```")[0].strip()
          elif "```" in result_text:
              result_text = result_text.split("```")[1].split("```")[0].strip()

          result = json.loads(result_text)
          return result
      except Exception as e:
          print(f"[Gemini verifier error] {e}")
          return {
              "sentences": [],
              "overall_unsupported_count": -1,
              "recommendation": "ERROR",
              "error_message": str(e)
          }

  if __name__ == "__main__":
      # Test verifier
      test_answer = "Bhishma took a vow of celibacy."
      test_chunks = [
          {"text": "Bhishma vowed celibacy for his father Shantanu's marriage."}
      ]
      result = verify_answer(test_answer, test_chunks)
      print(json.dumps(result, indent=2))
  ```
- [ ] Confirm `.env` has `GEMINI_API_KEY=your-key-here` (it should already,
  from Phase 1 — no separate `GOOGLE_API_KEY` needed).
- [ ] Test the verifier on a few hand-written answer + chunk pairs:
  ```bash
  python verifier.py
  ```
  Should output valid JSON with sentence-level scores.

**Success criteria:**
- [ ] Verifier runs without errors
- [ ] Output is valid JSON with `sentences`, `overall_unsupported_count`, `recommendation` keys
- [ ] Gemini API calls succeed (check your Google AI Studio dashboard for quota usage)

---

### Step 7: Integrate the verifier into the ask loop

**New file to create:** copy `03_ask.py` → `09_ask_verified.py`

`03_ask.py` already does retrieval + Gemini generation (it's not a Claude
script) — so Phase 2 doesn't need a parallel "Gemini version" of it. It
just needs the verifier gate bolted onto the existing generation step.
Keep `03_ask.py` completely unmodified; it's the "no verifier" baseline
Step 8 compares against.

**Checklist:**
- [ ] Copy `03_ask.py` to `09_ask_verified.py` as your starting point —
  keep everything (`SYSTEM_PROMPT`, `retrieve()`, `format_evidence()`,
  the REPL loop) unchanged.
- [ ] Add one import at the top:
  ```python
  from verifier import verify_answer
  ```
- [ ] In `main()`, insert the verifier gate right after `answer_text =
  response.text` and before it gets printed:
  ```python
  try:
      response = gemini_client.models.generate_content(
          model=gemini_model,
          contents=f"Evidence chunks:\n\n{evidence}\n\nQuestion: {question}",
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_PROMPT,
              temperature=0.0,
              max_output_tokens=1000,
          ),
      )
      answer_text = response.text

      # NEW: verify before showing the answer
      if config.VERIFIER_ENABLED:
          # results are Qdrant ScoredPoint objects — verify_answer expects
          # plain dicts with a "text" key, so pass the payloads, not the points.
          verification = verify_answer(answer_text, [r.payload for r in results])
          unsupported_count = verification.get("overall_unsupported_count", 0)

          if unsupported_count > 0:
              print(f"\n[verifier] Answer contains {unsupported_count} unsupported sentences.")
              if config.ABSTAIN_ON_UNSUPPORTED:
                  print("Refusing to output answer due to unsupported claims.\n")
                  continue  # loop back to the next `> ` prompt
              else:
                  print("(Showing answer anyway, but flagging for review.)\n")

          print("Verification details:")
          for sent in verification.get("sentences", []):
              status_emoji = "✓" if sent["status"] == "SUPPORTED" else "?" if sent["status"] == "PARTIALLY_SUPPORTED" else "✗"
              print(f"  {status_emoji} {sent['text'][:60]}... [{sent['status']}]")
          print()

      print(f"\n{answer_text}\n")
      print("Sources retrieved this turn:")
      for i, r in enumerate(results, start=1):
          p = r.payload
          print(f"  [chunk {i}] {p['video_id']} @ {p['start']:.0f}s "
                f"(score={r.score:.3f}) -> https://youtu.be/{p['video_id']}?t={int(p['start'])}")
      print()

  except Exception as e:
      print(f"\n[FAIL] Error querying Gemini: {e}\n")
  ```
- [ ] Test the full loop on your pilot corpus:
  ```bash
  python 09_ask_verified.py
  ```
- [ ] Run your 30-question test set and measure:
  - [ ] How many answers does the verifier block?
  - [ ] Of those, how many *should* it have blocked (true positives)?
  - [ ] Of those it lets through, how many have unsupported claims (false negatives)?

**Success criteria:**
- [ ] Verifier blocks at least 1–2 obviously hallucinated answers
- [ ] Verifier doesn't block all answers (it's not overly strict)
- [ ] Verified answers feel more trustworthy than unverified Phase 1 answers
- [ ] Total Gemini cost for 30 questions stays well under $1

---

### Step 8: Measure improvement over Phase 1

**Checklist:**
- [ ] Create a test set document: `TEST_RESULTS.md`
- [ ] Run the same 30 questions on Phase 1 (`03_ask.py`, no verifier) and
  Phase 2 (`09_ask_verified.py`, with verifier)
- [ ] For each question, record:
  ```
  Question: "Why did Bhishma take his vow?"

  Phase 1 (no verifier):
  - Answer: [full text]
  - Hallucinated: no/yes

  Phase 2 (with verifier):
  - Answer: [full text] or [REFUSED by verifier]
  - Verifier blocked: yes/no
  - Still hallucinated: no/yes
  ```
- [ ] Compute statistics:
  - [ ] Phase 1 hallucination rate (baseline)
  - [ ] Phase 2 hallucination rate after verifier
  - [ ] Verifier false-positive rate (blocked good answers)
  - [ ] Verifier false-negative rate (missed hallucinations)
- [ ] Write up findings in `TEST_RESULTS.md` with the numbers

**Success criteria:**
- [ ] Verifier reduces hallucination rate by at least 30–50%
- [ ] Verifier doesn't block >20% of good answers

---

### Optional: parallel ASR improvement

While Steps 3–8 are running, start collecting error data from the ASR
ensemble:

- [ ] Manually spot-check 5–10 minutes of ASR output from each video
- [ ] Note which proper nouns are consistently wrong (e.g., "bhisma"
  instead of "Bhishma")
- [ ] Build a small glossary of the top 20 mistakes:
  ```json
  {
    "asr_mistakes": [
      {"wrong": "bhisma", "correct": "Bhishma"},
      {"wrong": "santanu", "correct": "Shantanu"}
    ]
  }
  ```
- [ ] This becomes your Phase 3 glossary seed — don't do anything with it yet, just collect

---

### Gotchas & troubleshooting

**"IndicConformer won't load / runs out of memory"**
- The `AutoModelForCTC` / `AutoProcessor` download is a few GB; make sure
  you have enough RAM/VRAM free.
- If it keeps failing, set `INDICONFORMER_ENABLED = False` in `config.py`
  and run Whisper-only — the ensemble code already falls back gracefully.

**"faster-whisper is too slow on CPU"**
- Expected: ~2–3x realtime on CPU (a 1-hour video takes ~20–30 min).
- If you have GPU access, use it — `device="cuda"` cuts this dramatically.
- For Phase 2, you can skip IndicConformer entirely and run
  `faster-whisper` only if CPU time is the bottleneck.

**"Timestamps are way off"**
- Expected with the rough estimation in Step 4 — timestamps are +/- 5–10
  seconds at best with uniform speech-rate estimation. Fine for Phase 2;
  accurate timestamps come from a proper alignment tool in Phase 3.

**"Verifier is blocking everything"**
- Gemini might be overly strict. Try loosening the prompt:
  ```python
  # Change VERIFIER_SYSTEM_PROMPT to:
  "A claim is SUPPORTED if it's roughly consistent with the chunks, even if not word-for-word identical."
  ```
- Alternatively, try a higher-reasoning-tier Gemini model for
  `config.VERIFIER_MODEL` — check current pricing before switching.
- Set `ABSTAIN_ON_UNSUPPORTED = False` to see all answers even when the
  verifier flags them, useful for tuning before you turn abstention back on.

**"ASR is running on CPU and taking forever"**
- Expected. On CPU, 50 hours of audio takes roughly 150–200 hours of
  compute (4–6x realtime across both engines).
- Options:
  - [ ] Use a GPU if you have access (10–50x faster)
  - [ ] Run Whisper only (skip IndicConformer) — still accurate and faster
  - [ ] Accept the slow timeline and run overnight (still free)

**"Qdrant is huge (50+ GB)"**
- Normal for large playlists — Qdrant stores dense vectors (1024 numbers
  each) plus metadata.
- You can delete `./qdrant_data` between major iterations to save disk space.

---

### Success criteria for Phase 2

You're done when you can:

- [ ] Ingest a full 50+ hour playlist from YouTube
- [ ] Run the multi-engine ASR ensemble (IndicConformer + Whisper) on all audio
- [ ] Generate answers with the verifier gate enabled
- [ ] Ask 30+ test questions and see the verifier blocking at least some hallucinated answers
- [ ] Measure baseline improvement over Phase 1 (hallucination rate drops by 30%+)
- [ ] Have a small glossary of common ASR mistakes ready for Phase 3

---

### Timeline

- **Week 1:** Steps 1–2 (config + download audio)
- **Week 2:** Steps 3–4 (ASR + chunking) — likely runs overnight
- **Week 3:** Steps 5–6 (re-index + verifier) — quick
- **Week 4:** Steps 7–8 (integration + testing) — measure improvement
- **Week 5:** Buffer for debugging + parallel glossary collection

Total: 4–5 weeks if working 3–4 hours/week.

---

## What's NOT in Phase 2 (save for Phase 3+)

- ❌ Glossary correction layer
- ❌ BM25 sparse indexing
- ❌ Entity tagging/indexing
- ❌ Fine-grained UI (CLI only)
- ❌ Incremental video addition
- ❌ RLVR training

Don't try to add these now — they'll distract you from getting Phase 2 solid.

## Next phase hooks

Once Phase 2 is done, Phase 3 adds:
1. Glossary correction (using your collected mistakes from the optional step)
2. BM25 sparse indexing (for rare Sanskrit terms)
3. Entity indexing (person/place/term tags)

You'll have a smooth path forward with the foundation in place.

---

## Combined file map

| File | Phase | What it does |
|---|---|---|
| `config.py` | 1 & 2 | Every setting lives here — corpus label, chunk size, model names, ASR/verifier config |
| `.env` | 1 & 2 | `GEMINI_API_KEY` is the only secret currently read by any script (`ANTHROPIC_MODEL` in `config.py` is unused for now) |
| `01_fetch_transcripts.py` | 1 | Pulls YouTube's own captions for the videos in `config.VIDEO_IDS` |
| `02_chunk_and_index.py` | 1 | Merges captions into timestamped chunks (`config.CHUNK_SECONDS`/`CHUNK_OVERLAP_SECONDS`), embeds with `config.EMBED_MODEL_NAME`, stores in local Qdrant tagged with `config.PLAYLIST_ID` |
| `03_ask.py` | 1 | Retrieval + citation-constrained generation loop — already Gemini-based (`config.GEMINI_MODEL_NAME`), no verifier gate. This is the Step 8 "no verifier" baseline. |
| `04_download_audio.py` | 2 | Downloads every video in `config.YOUTUBE_PLAYLIST_ID` as MP3 |
| `05_asr_ensemble.py` | 2 | Runs IndicConformer + faster-whisper on all audio, keeps the higher-confidence transcript |
| `06_asr_to_chunks.py` | 2 | Estimates timestamps for ASR output and re-chunks it with the same `CHUNK_SECONDS`/`CHUNK_OVERLAP_SECONDS` as Phase 1 |
| `07_reindex_full_corpus.py` | 2 | Rebuilds the Qdrant collection from ASR-derived chunks instead of captions |
| `verifier.py` | 2 | Sentence-level SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED scoring via Gemini |
| `09_ask_verified.py` | 2 | Copy of `03_ask.py`'s retrieval + generation loop with the verifier gate added — this is the Step 8 "with verifier" comparison |
| `TEST_RESULTS.md` | 2 | Your 30-question before/after log comparing Phase 1 vs. Phase 2 hallucination rates |
