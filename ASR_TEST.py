import os
import re
import json
import time
import gc
import numpy as np
import soundfile as sf
import librosa
import torch
import yt_dlp
from jiwer import wer, cer
import config
from asr_engine import IndicConformerASR
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

HF_TOKEN = config.HF_TOKEN

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clean_vtt_subtitles(vtt_text: str) -> str:
    if not vtt_text:
        return ""
    lines = vtt_text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if (not line or
            line.startswith("WEBVTT") or
            line.startswith("Kind:") or
            line.startswith("Language:") or
            "-->" in line or
            line.isdigit()):
            continue
        line = re.sub(r'<[^>]+>', '', line)
        cleaned_lines.append(line)
    deduped = []
    for line in cleaned_lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return " ".join(deduped)

def normalize_indic_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def download_youtube_audio_and_subs(youtube_url: str, output_dir: str = "downloads") -> tuple[str, str, str | None]:
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['te'],
        'skip_download': False,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info['id']
        audio_path = os.path.join(output_dir, f"{video_id}.wav")
        reference_text = None
        requested_sub = info.get('requested_subtitles')
        if requested_sub and 'te' in requested_sub:
            sub_file = requested_sub['te']['filepath']
            if os.path.exists(sub_file):
                with open(sub_file, 'r', encoding='utf-8') as f:
                    raw_sub = f.read()
                    reference_text = clean_vtt_subtitles(raw_sub)
    return video_id, audio_path, reference_text

def add_noise_at_snr(audio_path: str, output_path: str, snr_db: float | None):
    y, sr = librosa.load(audio_path, sr=16000)
    if snr_db is None:
        sf.write(output_path, y, sr)
        return
    s_power = np.mean(y ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    n_power = s_power / snr_linear if snr_linear != 0 else 1e-10
    noise = np.random.normal(0, np.sqrt(n_power), y.shape)
    noisy_y = y + noise
    max_val = np.max(np.abs(noisy_y))
    if max_val > 1.0:
        noisy_y = noisy_y / max_val
    sf.write(output_path, noisy_y, sr)

def evaluate_video_noise_robustness(
    youtube_url: str,
    snr_levels_db: list[float | None],
    ground_truth_text: str | None = None,
    token: str | None = None
):
    print("=" * 70)
    print("STEP 1: Downloading Audio & Ground Truth")
    print("=" * 70)
    video_id, clean_audio_path, yt_sub = download_youtube_audio_and_subs(youtube_url)
    reference_raw = ground_truth_text or yt_sub or ""
    reference_norm = normalize_indic_text(reference_raw)

    print("\n" + "=" * 70)
    print("STEP 2: Generating Noisy Audio Profiles")
    print("=" * 70)
    temp_dir = "temp_noisy_audio"
    os.makedirs(temp_dir, exist_ok=True)
    noisy_files_map = {}
    for snr in snr_levels_db:
        label = "Clean" if snr is None else f"SNR_{snr}dB"
        noisy_path = os.path.join(temp_dir, f"{video_id}_{label}.wav")
        add_noise_at_snr(clean_audio_path, noisy_path, snr)
        noisy_files_map[label] = noisy_path
        print(f" -> Generated audio file for [{label}]")

    print("\n" + "=" * 70)
    print("STEP 3: Running IndicConformer (silence-cut chunks, no overlap)")
    print("=" * 70)
    indiconformer = IndicConformerASR(
        language_code=config.TARGET_LANGUAGE, decoder_type=config.ASR_DECODER, token=token,
        max_chunk_s=config.ASR_MAX_CHUNK_S, top_db=config.ASR_TOP_DB, pad_s=config.ASR_PAD_S,
    )

    benchmark_results = {}
    for label, audio_path in noisy_files_map.items():
        print(f" Transcribing [{label}] using IndicConformer...")
        res = indiconformer.transcribe(audio_path)
        norm_text = normalize_indic_text(res['text'])
        wer_score = round(wer(reference_norm, norm_text), 4) if reference_norm else None
        cer_score = round(cer(reference_norm, norm_text), 4) if reference_norm else None
        benchmark_results[label] = {
            'text': res['text'], 'norm_text': norm_text,
            'latency': res['latency_sec'], 'wer': wer_score, 'cer': cer_score
        }

    del indiconformer
    clear_vram()

    print("\n" + "=" * 80)
    print("                         INDICCONFORMER REPORT")
    print("=" * 80)
    print(f"\n--- GROUND TRUTH (REFERENCE) ---")
    print(f"RAW       : {reference_raw[:400]}..." if len(reference_raw) > 400 else f"RAW       : {reference_raw}")
    print(f"NORMALIZED: {reference_norm[:400]}..." if len(reference_norm) > 400 else f"NORMALIZED: {reference_norm}")

    json_report = []
    for label in noisy_files_map:
        ic_res = benchmark_results[label]
        print("\n" + "-" * 80)
        print(f" NOISE LEVEL PROFILE: [{label}]")
        print("-" * 80)
        print(f" [IndicConformer] (Latency: {ic_res.get('latency')}s | WER: {ic_res.get('wer')} | CER: {ic_res.get('cer')})")
        print(f"  OUTPUT : {ic_res.get('text')}\n")
        json_report.append({'noise_level': label, 'reference': reference_norm, 'indic_conformer': ic_res})

    report_file = f"{video_id}_benchmark_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 80)
    print(f"Saved full analytical report to: {report_file}")
    print("=" * 80)

if __name__ == "__main__":
    TEST_YOUTUBE_URL = "https://www.youtube.com/watch?v=9u1S0xp1-Kc"
    NOISE_STEPS_SNR = [None
                    #    , 20.0, 10.0, 0.0
                       ]
    evaluate_video_noise_robustness(
        youtube_url=TEST_YOUTUBE_URL,
        snr_levels_db=NOISE_STEPS_SNR,
        ground_truth_text=None,
        token=HF_TOKEN
    )