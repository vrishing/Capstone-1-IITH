# Phase 2: Full Playlist + Free Indic ASR + Gemini Verifier

**Scope:** Take MVP-1 (3 videos, YouTube captions) and upgrade to a production-ready closed-corpus system on your full playlist using **completely free, self-hosted ASR** and **Gemini verifier** (cost: ~$3-5 for 50+ hours of verification, vs. $50+ with Claude). Skip glossary, BM25, entity indexing — those are Phase 3.

**Outcome:** A system that can ingest a 50-100 hour playlist, transcribe it with an open-source ASR ensemble (IndicConformer + Whisper), and generate answers that must pass a Gemini-based sentence-level verifier before reaching the user. **$0 for ASR, $5-10 total for Gemini verification on 50+ hours.**

**Timeline:** 4-5 weeks, 3-4 hours/week if working part-time.

---

## Prerequisites (do before starting Phase 2)

- [ ] You have a YouTube playlist ID (the `PLxxxxx` string from the URL) with 50+ videos
- [ ] You have API keys for:
  - [ ] Google Generative AI (Gemini API) — get **free tier** at https://ai.google.dev, no credit card needed, 1500 req/min limit
- [ ] You have `yt-dlp` installed locally: `pip install yt-dlp`
- [ ] Disk space: ~100 GB for 50+ hours of MP3s + ASR output (temporary; delete after indexing)
- [ ] GPU recommended for ASR (3-4x speedup), but CPU works (slower, ~4-6x realtime)
- [ ] Patience: first ASR run on 50 videos takes 4-8 hours on CPU, 1-2 hours on GPU

---

## Cost Breakdown: Why This Matters

| Component | Original Plan | Phase 2 (This Plan) |
|---|---|---|
| ASR (50+ hours) | $1000-2000 (Sarvam API at scale) | $0 (open source, self-hosted) |
| Verifier (1000+ questions) | $50-100 (Claude at scale) | $5-10 (Gemini is 10x cheaper) |
| **Total for full thesis work** | **$1050-2100** | **$5-10** |

**By using free ASR and Gemini instead of paid APIs, you save $1000+ on the same work.**

---

## Phase 2 Checklist

### Step 1: Extend config.py for full playlist

**File to modify:** `config.py` (from MVP-1)

**Checklist:**
- [ ] Replace `VIDEO_IDS` with `PLAYLIST_ID = "PLxxxxx"` (the actual YouTube playlist ID)
- [ ] Add ASR configuration:
  ```python
  # ASR Ensemble (completely free, self-hosted)
  ASR_ENGINES = ["indiconformer", "whisper_indic"]  # both free, open source
  INDICONFORMER_ENABLED = True  # AI4Bharat, covers all 22 Indian languages, MIT license
  WHISPER_INDIC_ENABLED = True  # OpenAI Whisper + AI4Bharat fine-tune, free fallback
  ASR_ENSEMBLE_STRATEGY = "highest_confidence"  # pick IndicConformer or Whisper output with best score
  TARGET_LANGUAGE = "te"  # Telugu (change to "sa" for Sanskrit if needed, or "hi" for Hindi)
  ```
- [ ] Add verifier configuration (using Gemini, much cheaper):
  ```python
  VERIFIER_MODEL = "gemini-2.0-flash"  # ~$0.00001 per request (cheapest option)
  # Alternative: "gemini-1.5-pro" if you need stronger reasoning (still <$0.05 per request)
  VERIFIER_ENABLED = True
  ABSTAIN_ON_UNSUPPORTED = True  # if True, refuse any answer with unsupported sentences
  GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
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

### Step 3: Run free ASR ensemble on all audio

**New file to create:** `05_asr_ensemble.py`

**What it does:** Runs IndicConformer (AI4Bharat) + Whisper (OpenAI, free fine-tune) on all audio files, picks best output by confidence score. **100% free, no API costs.**

**Checklist:**
- [ ] Install ASR dependencies:
  ```bash
  pip install --break-system-packages faster-whisper transformers torchaudio torch
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
      """OpenAI Whisper + AI4Bharat fine-tune for Indic languages — free, fast."""
      def __init__(self, language_code="te"):
          # Use AI4Bharat's fine-tuned Whisper for Telugu/Indic languages
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
- [ ] Test on the first 3 videos you downloaded (this will download model weights on first run):
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
- [ ] **First run downloads models (~5-10 GB for both engines combined)**. This happens once, takes 10-30 min.
- [ ] **On CPU, this is slow:** IndicConformer ~3-4x realtime, Whisper ~2-3x realtime. For 50 hours, expect 150-200 hours of compute on CPU. GPU cuts this to 10-20 hours.
- [ ] If you don't have GPU and CPU speed is blocking, **skip IndicConformer and use Whisper only** — it's faster and still excellent for Telugu.
- [ ] **No API costs, ever.** All models are free, open-source, self-hosted.

**Success criteria:**
- [ ] All audio files produce `*_asr.json` output files
- [ ] Each output has `text` field with >100 words
- [ ] One pass through full playlist completed (CPU: overnight run, GPU: 1-2 hours)

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

### Step 6: Implement the sentence-level verifier (using Gemini — much cheaper)

**New file to create:** `08_verifier.py`

**What it does:** Takes a generated answer and retrieves chunks, scores each sentence as SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED using Gemini. **Cost: ~$0.00001 per verification (50+ hours of answers = $0.50-1.00 total).**

**Checklist:**
- [ ] Install Gemini SDK:
  ```bash
  pip install --break-system-packages google-generativeai
  ```
- [ ] Create the verifier script:
  ```python
  import json
  import os
  import google.generativeai as genai
  
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
      genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
      model = genai.GenerativeModel("gemini-2.0-flash")
      
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
          response = model.generate_content(
              f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_message}",
              generation_config=genai.types.GenerationConfig(max_output_tokens=1000)
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
- [ ] Update `.env` with your Gemini API key:
  ```bash
  echo "GOOGLE_API_KEY=your-key-here" >> .env
  ```
- [ ] Test the verifier on a few hand-written answer + chunk pairs:
  ```bash
  python 08_verifier.py
  ```
  Should output valid JSON with sentence-level scores.

**Success criteria:**
- [ ] Verifier runs without errors
- [ ] Output is valid JSON with `sentences`, `overall_unsupported_count`, `recommendation` keys
- [ ] Gemini API calls succeed (check your Google AI dashboard for quota usage)

---

### Step 7: Integrate verifier into the ask loop (using Gemini for both generation and verification)

**File to modify:** `03_ask_gemini.py` (use Gemini as both the answer generator AND verifier for cost efficiency)

**Checklist:**
- [ ] Update `03_ask_gemini.py` to integrate the verifier:
  ```python
  import google.generativeai as genai
  from 08_verifier import verify_answer
  import config
  
  genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
  
  # ... (retrieval code same as MVP-1)
  
  # Generation with verifier gate
  gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # or gemini-1.5-pro if you need stronger reasoning
  
  answer_prompt = f"""You are answering questions using ONLY the evidence chunks provided below. \
  Each chunk is numbered and comes from a specific video and timestamp in a closed video corpus.
  
  Rules:
  1. Every factual sentence in your answer MUST be supported by at least one of the numbered evidence chunks. \
  Cite the chunk number in square brackets after each factual sentence.
  2. Do NOT use any knowledge you have from outside these chunks.
  3. If the evidence chunks do not contain enough information to answer the question, respond with exactly: \
  "I don't have enough information in this corpus to answer that."
  
  Evidence chunks:
  {evidence}
  
  Question: {question}"""
  
  try:
      response = gemini_model.generate_content(
          answer_prompt,
          generation_config=genai.types.GenerationConfig(max_output_tokens=800)
      )
      answer_text = response.text
  except Exception as e:
      print(f"[Generation error] {e}")
      continue
  
  # NEW: verify with Gemini verifier
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
  
  # Print answer
  print(f"\n{answer_text}\n")
  
  # Print sources
  print("Sources retrieved this turn:")
  for i, r in enumerate(results, start=1):
      p = r.payload
      print(f"  [chunk {i}] {p['video_id']} @ {p['start']:.0f}s "
            f"(score={r.score:.3f}) -> https://youtu.be/{p['video_id']}?t={int(p['start'])}")
  print()
  ```
- [ ] Test the full loop on your pilot corpus:
  ```bash
  python 03_ask_gemini.py
  ```
- [ ] Run your 30-question test set and measure:
  - [ ] How many answers does the verifier block?
  - [ ] Of those, how many should it have blocked (true positives)?
  - [ ] Of those it lets through, how many have unsupported claims (false negatives)?

**Success criteria:**
- [ ] Verifier blocks at least 1-2 obviously hallucinated answers
- [ ] Verifier doesn't block all answers (it's not overly strict)
- [ ] Verified answers feel more trustworthy than unverified MVP-1 answers
- [ ] Total cost for 30 questions: <$0.10 (Gemini is extremely cheap)

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
- Gemini might be overly strict. Try adjusting the prompt to be more lenient:
  ```python
  # Change VERIFIER_SYSTEM_PROMPT to:
  "A claim is SUPPORTED if it's roughly consistent with the chunks, even if not word-for-word identical."
  ```
- Alternatively, try `gemini-1.5-pro` instead of `gemini-2.0-flash` — it's still cheap (~$0.0001 per request) but more calibrated.
- Set `ABSTAIN_ON_UNSUPPORTED = False` to see all answers even if verifier flags them

**"ASR is running on CPU and taking forever"**
- This is expected. On CPU, 50 hours of audio takes 150-200 hours of compute time (4-6x realtime).
- Options:
  - [ ] If you have GPU access, use it (50x faster)
  - [ ] Run Whisper only (skip IndicConformer) — still accurate and faster
  - [ ] Accept the slow timeline and run overnight (still free)

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