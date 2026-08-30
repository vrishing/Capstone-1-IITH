# Phase 2: Full Playlist + Indic ASR + Verifier

**Scope:** Take MVP-1 (3 videos, YouTube captions) and upgrade to a production-ready closed-corpus system on your full playlist using real ASR and a verifier gate. Skip glossary, BM25, entity indexing — those are Phase 3.

**Outcome:** A system that can ingest a 50-100 hour playlist, transcribe it with an ASR ensemble, and generate answers that must pass a sentence-level verifier before reaching the user.

**Timeline:** 4-5 weeks, 3-4 hours/week if working part-time.

---

## Prerequisites (do before starting Phase 2)

- [ ] You have a YouTube playlist ID (the `PLxxxxx` string from the URL) with 50+ videos
- [ ] You have API keys for:
  - [ ] Sarvam Saaras v3 (apply at https://www.sarvam.ai/ — free tier gives 100k calls/month)
  - [ ] Anthropic (Claude API) for the verifier
  - [ ] Google Generative AI (Gemini, for comparison if desired)
- [ ] You have `yt-dlp` installed locally: `pip install yt-dlp`
- [ ] Disk space: ~100 GB for 50+ hours of MP3s + ASR output (temporary; delete after indexing)
- [ ] Patience: first ASR run on 50 videos takes 2-4 hours depending on parallelization

---

## Phase 2 Checklist

### Step 1: Extend config.py for full playlist

**File to modify:** `config.py` (from MVP-1)

**Checklist:**
- [ ] Replace `VIDEO_IDS` with `PLAYLIST_ID = "PLxxxxx"` (the actual YouTube playlist ID)
- [ ] Add ASR configuration:
  ```python
  # ASR Ensemble
  SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
  WHISPERX_ENABLED = True  # set to False if no GPU available
  INDICONFORMER_ENABLED = True
  ASR_ENSEMBLE_STRATEGY = "highest_confidence"  # pick Sarvam/WhisperX/IndicConformer output with best score
  ```
- [ ] Add verifier configuration:
  ```python
  VERIFIER_MODEL = "claude-sonnet-5"
  VERIFIER_ENABLED = True
  ABSTAIN_ON_UNSUPPORTED = True  # if True, refuse any answer with unsupported sentences
  ```
- [ ] Add ASR output directory:
  ```python
  ASR_OUTPUT_DIR = "./data/asr_output"
  ```
- [ ] Keep everything else from MVP-1 (embedding model, Qdrant, chunking parameters, retrieval top_k)

**Testing:** 
```bash
python -c "import config; print(f'Playlist: {config.PLAYLIST_ID}'); print(f'Verifier: {config.VERIFIER_ENABLED}')"
```
Should print your playlist ID and confirm verifier is enabled.

---

### Step 2: Download playlist audio

**New file to create:** `04_download_audio.py`

**What it does:** Fetches all videos from the playlist and converts to MP3 with timestamps.

**Checklist:**
- [ ] Create the script with this structure:
  ```python
  import os
  import json
  import subprocess
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
      print(f"Fetching playlist {config.PLAYLIST_ID}...")
      video_ids = get_playlist_videos(config.PLAYLIST_ID)
      print(f"Found {len(video_ids)} videos")
      
      for i, vid in enumerate(video_ids, start=1):
          print(f"[{i}/{len(video_ids)}] downloading {vid}...")
          download_audio(vid, config.ASR_OUTPUT_DIR)
  
  if __name__ == "__main__":
      main()
  ```
- [ ] Test on just the first 3 videos before running the full playlist:
  ```bash
  # Edit 04_download_audio.py temporarily: change get_playlist_videos to return only first 3
  python 04_download_audio.py
  ```
- [ ] Verify MP3 files exist in `data/asr_output/`:
  ```bash
  ls -lh data/asr_output/*.mp3 | head
  ```

**Gotchas:**
- [ ] FFmpeg must be installed: `ffmpeg -version`. If not: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux)
- [ ] The first run takes a while (one video per minute on a typical connection)
- [ ] Don't run on the full playlist until you've tested on 3 videos

**Success criteria:** 
- [ ] 3+ MP3 files in `data/asr_output/` 
- [ ] Each file is >10MB (confirms it's actual audio, not corrupt download)
- [ ] Next step: run full playlist (will take hours, possibly overnight)

---

### Step 3: Run ASR ensemble on all audio

**New file to create:** `05_asr_ensemble.py`

**What it does:** Runs Sarvam + WhisperX + IndicConformer on all audio files, picks best output by confidence score.

**Checklist:**
- [ ] Install ASR dependencies:
  ```bash
  pip install --break-system-packages faster-whisper openai-whisper
  # Sarvam is accessed via API, no local install needed
  ```
- [ ] Create the script:
  ```python
  import os
  import json
  import concurrent.futures
  from pathlib import Path
  
  import requests
  from faster_whisper import WhisperModel
  
  import config
  
  class SarvamASR:
      def __init__(self, api_key: str):
          self.api_key = api_key
          self.base_url = "https://api.sarvam.ai/speech-to-text"
      
      def transcribe(self, audio_path: str) -> dict:
          """Call Sarvam API."""
          with open(audio_path, 'rb') as f:
              files = {'file': f}
              data = {'language_code': 'te'}  # Telugu
              headers = {'API-Subscription-Key': self.api_key}
              try:
                  response = requests.post(self.base_url, files=files, data=data, headers=headers, timeout=60)
                  result = response.json()
                  return {
                      'text': result.get('transcript', ''),
                      'confidence': result.get('confidence_score', 0.5),
                      'engine': 'sarvam'
                  }
              except Exception as e:
                  print(f"[Sarvam error] {e}")
                  return {'text': '', 'confidence': 0.0, 'engine': 'sarvam'}
  
  class WhisperXASR:
      def __init__(self):
          self.model = WhisperModel("base", device="cuda", compute_type="float16")  # change to "cpu" if no GPU
      
      def transcribe(self, audio_path: str) -> dict:
          """Run WhisperX locally."""
          try:
              segments, info = self.model.transcribe(audio_path, language="te")
              text = " ".join([seg.text for seg in segments])
              # Rough confidence: if model ran without error, assume good
              return {
                  'text': text,
                  'confidence': 0.7,  # WhisperX doesn't expose per-utterance confidence easily
                  'engine': 'whisperx'
              }
          except Exception as e:
              print(f"[WhisperX error] {e}")
              return {'text': '', 'confidence': 0.0, 'engine': 'whisperx'}
  
  class IndiconformerASR:
      """Placeholder — IndicConformer requires Hugging Face model setup."""
      def transcribe(self, audio_path: str) -> dict:
          # TODO: implement via huggingface transformers or AI4Bharat's inference API
          # For MVP-2, you can skip this and just use Sarvam + WhisperX
          return {'text': '', 'confidence': 0.0, 'engine': 'indiconformer'}
  
  def transcribe_one_video(video_id: str, audio_path: str, sarvam: SarvamASR, whisperx: WhisperXASR) -> dict:
      """Run ensemble on one video, pick highest-confidence output."""
      print(f"  [transcribing] {video_id}...")
      
      outputs = []
      outputs.append(sarvam.transcribe(audio_path))
      if config.WHISPERX_ENABLED:
          outputs.append(whisperx.transcribe(audio_path))
      
      # Pick the one with highest confidence
      best = max(outputs, key=lambda x: x['confidence'])
      
      return {
          'video_id': video_id,
          'text': best['text'],
          'engine_used': best['engine'],
          'confidence': best['confidence'],
          'all_outputs': outputs  # for debugging
      }
  
  def main():
      os.makedirs(config.ASR_OUTPUT_DIR, exist_ok=True)
      
      print("Initializing ASR engines...")
      sarvam = SarvamASR(config.SARVAM_API_KEY)
      whisperx = WhisperXASR() if config.WHISPERX_ENABLED else None
      
      audio_files = sorted(Path(config.ASR_OUTPUT_DIR).glob("*.mp3"))
      print(f"Found {len(audio_files)} audio files to transcribe")
      
      for audio_path in audio_files:
          video_id = audio_path.stem
          result = transcribe_one_video(video_id, str(audio_path), sarvam, whisperx)
          
          out_path = os.path.join(config.ASR_OUTPUT_DIR, f"{video_id}_asr.json")
          with open(out_path, 'w', encoding='utf-8') as f:
              json.dump(result, f, ensure_ascii=False, indent=2)
          
          print(f"[ok] {video_id}: {len(result['text'].split())} words, engine={result['engine_used']}")
  
  if __name__ == "__main__":
      main()
  ```
- [ ] Update `.env` with your Sarvam API key:
  ```bash
  echo "SARVAM_API_KEY=your-key-here" >> .env
  ```
- [ ] Test on the first 3 videos you downloaded:
  ```bash
  python 05_asr_ensemble.py
  ```
  Should produce `data/asr_output/*_asr.json` files with transcription results.
- [ ] Spot-check one result:
  ```bash
  cat data/asr_output/VIDEO_ID_asr.json | jq '.text' | head -c 200
  ```
  Should see Telugu text (native script or transliterated).

**Gotchas:**
- [ ] Sarvam API has rate limits (~10 requests/min on free tier). If you hit limits, add `time.sleep(6)` between calls.
- [ ] WhisperX on CPU is *very slow* (~4x realtime). For a 50-hour playlist, this can take 24+ hours. Consider:
  - [ ] Running WhisperX only (skipping Sarvam) if you have GPU
  - [ ] Running Sarvam only (skipping WhisperX) for fast MVP-2, add WhisperX later
  - [ ] Running both in parallel with `concurrent.futures` if you have API quota
- [ ] Confidence scores are rough. You'll want to compute actual WER against a validation set later (Phase 3).

**Success criteria:**
- [ ] All audio files produce `*_asr.json` output files
- [ ] Each output has `text` field with >100 words
- [ ] One pass through full playlist completed (even if it took hours)

---

### Step 4: Convert ASR output to timestamped chunks

**New file to create:** `06_asr_to_chunks.py`

**What it does:** Take raw ASR JSON (which has no timestamps since it's full-audio transcription), estimate timestamps by spreading the text across the audio duration, then chunk it into 60-second evidence spans like MVP-1 did.

**Checklist:**
- [ ] Create the script:
  ```python
  import os
  import json
  import math
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
      
      for i, word in enumerate(words):
          word_duration = 1.0 / words_per_second  # rough estimate
          entries.append({
              'text': word,
              'start': current_time,
              'duration': word_duration
          })
          current_time += word_duration
      
      return entries
  
  def merge_into_chunks(entries: list[dict], chunk_seconds: int = 60, overlap_seconds: int = 10):
      """Same chunking logic as MVP-1."""
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
          chunks = merge_into_chunks(entries)
          
          out_path = asr_path.replace('_asr.json', '_chunks.json')
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
- [ ] Timestamp estimation is rough — a 60-minute video with uniform speech rate will have timestamps, but they're not precise. This is fine for MVP-2; precise timestamps come from enhanced ASR models or alignment tools later.
- [ ] `pydub` requires FFmpeg to be installed (same as Step 2).

**Success criteria:**
- [ ] All videos have `*_chunks.json` files
- [ ] Chunk count looks reasonable (50-100 videos × 40-60 chunks each ≈ 2000-6000 total chunks)

---

### Step 5: Re-index full corpus into Qdrant with ASR data

**New file to modify:** Adapt `02_chunk_and_index.py` → `07_reindex_full_corpus.py`

**Checklist:**
- [ ] Copy `02_chunk_and_index.py` to `07_reindex_full_corpus.py`
- [ ] Modify to read from `*_chunks.json` instead of YouTube captions:
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
- [ ] Keep the embedding and Qdrant storage logic the same
- [ ] Delete the old Qdrant collection first (to avoid mixing MVP-1 YouTube captions with MVP-2 ASR):
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
# Verify chunk count
python -c "from qdrant_client import QdrantClient; c = QdrantClient('./qdrant_data'); print(f'Points in collection: {c.get_collection(\"transcript_chunks\").points_count}')"
```

**Success criteria:**
- [ ] Qdrant collection has 2000+ points (one per chunk)
- [ ] Collection rebuilds from scratch in <10 min on modern hardware

---

### Step 6: Implement the sentence-level verifier

**New file to create:** `08_verifier.py`

**What it does:** Takes a generated answer and retrieves chunks, scores each sentence as SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED.

**Checklist:**
- [ ] Create the verifier script:
  ```python
  import json
  from anthropic import Anthropic
  
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
      """Call Claude to verify an answer against retrieved chunks."""
      client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
      
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
      
      response = client.messages.create(
          model="claude-sonnet-5",
          max_tokens=1000,
          system=VERIFIER_SYSTEM_PROMPT,
          messages=[{"role": "user", "content": user_message}],
      )
      
      result_text = response.content[0].text
      try:
          result = json.loads(result_text)
          return result
      except json.JSONDecodeError:
          # If Claude's response isn't valid JSON, mark it as error
          return {
              "sentences": [],
              "overall_unsupported_count": -1,
              "recommendation": "ERROR",
              "raw_response": result_text
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
- [ ] Test the verifier on a few hand-written answer + chunk pairs:
  ```bash
  python 08_verifier.py
  ```
  Should output valid JSON with sentence-level scores.

**Success criteria:**
- [ ] Verifier runs without errors
- [ ] Output is valid JSON with `sentences`, `overall_unsupported_count`, `recommendation` keys

---

### Step 7: Integrate verifier into the ask loop

**File to modify:** `03_ask.py` (or `03_ask_gemini.py` if using Gemini)

**Checklist:**
- [ ] Import the verifier:
  ```python
  from 08_verifier import verify_answer
  import config
  ```
- [ ] After LLM generation, before printing the answer:
  ```python
  # Generate answer (same as MVP-1)
  answer_text = claude.messages.create(...).content[0].text
  
  # NEW: verify
  if config.VERIFIER_ENABLED:
      verification = verify_answer(answer_text, results)
      unsupported_count = verification.get('overall_unsupported_count', 0)
      
      if unsupported_count > 0:
          print(f"\n[verifier] Answer contains {unsupported_count} unsupported sentences.")
          if config.ABSTAIN_ON_UNSUPPORTED:
              print("Refusing to output answer due to unsupported claims.\n")
              continue  # loop back, ask next question
          else:
              print("(Showing answer anyway, but flagging for review.)\n")
      
      # Print verification details
      print("Verification details:")
      for sent in verification.get('sentences', []):
          status_emoji = "✓" if sent['status'] == "SUPPORTED" else "?" if sent['status'] == "PARTIALLY_SUPPORTED" else "✗"
          print(f"  {status_emoji} {sent['text'][:60]}... [{sent['status']}]")
      print()
  
  # Print answer (same as MVP-1)
  print(f"\n{answer_text}\n")
  ```
- [ ] Test the full loop on your pilot corpus:
  ```bash
  python 03_ask.py
  ```
- [ ] Run your 30-question test set and measure:
  - [ ] How many answers does the verifier block?
  - [ ] Of those, how many should it have blocked (true positives)?
  - [ ] Of those it lets through, how many have unsupported claims (false negatives)?

**Success criteria:**
- [ ] Verifier blocks at least 1-2 obviously hallucinated answers
- [ ] Verifier doesn't block all answers (it's not overly strict)
- [ ] Verified answers feel more trustworthy than unverified MVP-1 answers

---

### Step 8: Measure improvement over MVP-1

**Checklist:**
- [ ] Create a test set document: `TEST_RESULTS.md`
- [ ] Run the same 30 questions on MVP-1 (no verifier) and MVP-2 (with verifier)
- [ ] For each question, record:
  ```
  Question: "Why did Bhishma take his vow?"
  
  MVP-1 (no verifier):
  - Answer: [full text]
  - Hallucinated: no/yes
  
  MVP-2 (with verifier):
  - Answer: [full text] or [REFUSED by verifier]
  - Verifier blocked: yes/no
  - Still hallucinated: no/yes
  ```
- [ ] Compute statistics:
  - [ ] MVP-1 hallucination rate (baseline)
  - [ ] MVP-2 hallucination rate after verifier
  - [ ] Verifier false-positive rate (blocked good answers)
  - [ ] Verifier false-negative rate (missed hallucinations)
- [ ] Write up findings in `TEST_RESULTS.md` with the numbers

**Success criteria:**
- [ ] Verifier reduces hallucination rate by at least 30-50%
- [ ] Verifier doesn't block >20% of good answers

---

## Optional: Parallel ASR improvement

While Step 3-8 are running, **start collecting error data** from the ASR ensemble:

- [ ] Manually spot-check 5-10 minutes of ASR output from each video
- [ ] Note which proper nouns are consistently wrong (e.g., "bhisma" instead of "Bhishma")
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

## Gotchas & Troubleshooting

**"Sarvam API keeps timing out"**
- Add exponential backoff between retries
- Check your API quota at https://www.sarvam.ai/dashboard
- Consider running fewer videos in parallel

**"WhisperX is too slow"**
- Run on CPU: expected speed is 4-6x realtime (1 hour video ≈ 4-6 hours processing)
- If you have GPU access, use it (50x faster)
- For MVP-2, you can skip WhisperX entirely and use Sarvam + IndicConformer only

**"Timestamps are way off"**
- This is expected with the rough estimation in Step 4
- Timestamps are +/- 5-10 seconds at best with uniform speech-rate estimation
- This is fine for MVP-2; accurate timestamps come from a proper alignment tool in Phase 3

**"Verifier is blocking everything"**
- Claude might be overly strict. Try adjusting the prompt to be more lenient:
  ```python
  # Change VERIFIER_SYSTEM_PROMPT to:
  "A claim is SUPPORTED if it's roughly consistent with the chunks, even if not word-for-word identical."
  ```
- Alternatively, set `ABSTAIN_ON_UNSUPPORTED = False` to see all answers even if verifier flags them

**"Qdrant is huge (50+ GB)"**
- This is normal for large playlists. Qdrant stores dense vectors (1024 numbers each) + metadata.
- You can delete `./qdrant_data` between major iterations to save disk space

---

## Success Criteria for Phase 2

You're done when you can:

- [ ] Ingest a full 50+ hour playlist from YouTube
- [ ] Run multi-engine ASR (Sarvam + at least one other) on all audio
- [ ] Generate answers with the verifier gate enabled
- [ ] Ask 30+ test questions and see verifier blocking at least some hallucinated answers
- [ ] Measure baseline improvement over MVP-1 (hallucination rate drops by 30%+)
- [ ] Have a small glossary of common ASR mistakes ready for Phase 3

---

## What's NOT in Phase 2 (save for Phase 3+)

- ❌ Glossary correction layer
- ❌ BM25 sparse indexing
- ❌ Entity tagging/indexing
- ❌ Fine-grained UI (CLI only)
- ❌ Incremental video addition
- ❌ RLVR training

All of these are planned for Phase 3+ — don't try to add them now, they'll distract you.

---

## Timeline

- **Week 1:** Steps 1-2 (config + download audio)
- **Week 2:** Step 3-4 (ASR + chunking) — likely runs overnight
- **Week 3:** Steps 5-6 (re-index + verifier) — quick
- **Week 4:** Steps 7-8 (integration + testing) — measure improvement
- **Week 5:** Buffer for debugging + parallel glossary collection

Total: 4-5 weeks if working 3-4 hours/week.

---

## Next Phase Hooks

Once Phase 2 is done, Phase 3 adds:
1. Glossary correction (using your collected mistakes from the optional step)
2. BM25 sparse indexing (for rare Sanskrit terms)
3. Entity indexing (person/place/term tags)

You'll have a smooth path forward with the foundation in place.