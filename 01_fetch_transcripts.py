"""
Step 1 — pull YouTube's own transcript for EVERY video in the given playlist.

Uses yt-dlp to fetch the list of video IDs from config.YOUTUBE_PLAYLIST_ID,
then fetches the transcript for each one that has captions.
"""
import json
import os
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
import config

OUT_DIR = "data/transcripts"

def get_playlist_video_ids(playlist_id: str) -> list[str]:
    """Fetch all video IDs from a YouTube playlist using yt-dlp."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
    }
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f'https://www.youtube.com/playlist?list={playlist_id}', download=False)
        return [entry['id'] for entry in result.get('entries', [])]

def fetch_one(video_id: str) -> None:
    out_path = os.path.join(OUT_DIR, f"{video_id}.json")
    if os.path.exists(out_path):
        print(f"[skip] {video_id} already fetched")
        return

    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(video_id)
    except (TranscriptsDisabled, VideoUnavailable) as e:
        print(f"[FAIL] {video_id}: {e}")
        print("       -> this video has no captions at all. Skipping.")
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
        try:
            transcript = next(iter(transcript_list))
            used_lang = transcript.language_code
        except StopIteration:
            print(f"[FAIL] {video_id}: no transcript in any language")
            return

    entries_raw = transcript.fetch()
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

    # Instead of config.VIDEO_IDS, fetch from the playlist
    playlist_id = config.YOUTUBE_PLAYLIST_ID
    if not playlist_id:
        print("[FAIL] YOUTUBE_PLAYLIST_ID is not set in config.py")
        return

    print(f"Fetching video list from playlist {playlist_id}...")
    video_ids = get_playlist_video_ids(playlist_id)
    print(f"Found {len(video_ids)} videos in playlist.")

    for vid in video_ids:
        fetch_one(vid)

if __name__ == "__main__":
    main()