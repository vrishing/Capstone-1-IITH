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