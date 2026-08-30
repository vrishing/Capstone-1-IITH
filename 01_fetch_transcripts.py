"""
Step 1 — pull YouTube's own transcript for each pilot video.

This is what makes MVP-1 "transcript-only": we are NOT running Sarvam /
WhisperX / IndicConformer yet (that's MVP-2). We're using whatever caption
track YouTube already has, to prove the retrieval + generation + citation
loop works end to end before investing in the full ASR ensemble.

Run: python 01_fetch_transcripts.py
Needs internet access to youtube.com — run this on your own machine.
"""
import json
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

import config

OUT_DIR = "data/transcripts"


def fetch_one(video_id: str) -> None:
    out_path = os.path.join(OUT_DIR, f"{video_id}.json")
    if os.path.exists(out_path):
        print(f"[skip] {video_id} already fetched")
        return

    try:
        # Handles both youtube-transcript-api v1.0+ and legacy v0.x versions
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(video_id)
    except (TranscriptsDisabled, VideoUnavailable) as e:
        print(f"[FAIL] {video_id}: {e}")
        print("       -> this video has no captions at all. Either pick a")
        print("          different video, or plan to ASR it yourself (MVP-2).")
        return
    except Exception as e:
        print(f"[FAIL] {video_id}: unexpected error listing transcripts: {e}")
        return

    transcript = None
    used_lang = None
    for lang in config.TRANSCRIPT_LANGS:
        try:
            transcript = transcript_list.find_transcript([lang])
            used_lang = lang
            break
        except NoTranscriptFound:
            continue

    if transcript is None:
        # fall back to whatever's available
        try:
            transcript = next(iter(transcript_list))
            used_lang = transcript.language_code
        except StopIteration:
            print(f"[FAIL] {video_id}: no transcript in any language")
            return

    entries_raw = transcript.fetch()

    # Normalize entries to dicts: {"text": ..., "start": ..., "duration": ...}
    normalized_entries = []
    for e in entries_raw:
        if isinstance(e, dict):
            normalized_entries.append({
                "text": e["text"],
                "start": e["start"],
                "duration": e["duration"]
            })
        else:
            normalized_entries.append({
                "text": getattr(e, "text", ""),
                "start": getattr(e, "start", 0.0),
                "duration": getattr(e, "duration", 0.0)
            })

    data = {
        "video_id": video_id,
        "language": used_lang,
        "is_generated": transcript.is_generated,
        "entries": normalized_entries,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[ok] {video_id}: {len(normalized_entries)} caption lines, "
          f"lang={used_lang}, auto-generated={transcript.is_generated}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for vid in config.VIDEO_IDS:
        fetch_one(vid)


if __name__ == "__main__":
    main()