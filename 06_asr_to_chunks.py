from __future__ import annotations
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