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