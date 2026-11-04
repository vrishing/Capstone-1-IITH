"""
One downloader (replaces 01_fetch_transcripts.py, 04_download_audio.py and the
download code in ASR_TEST.py): playlist listing, 16 kHz mono wav audio, and
best-effort captions (baseline only, never ground truth).
"""
from __future__ import annotations

import json
import os
import subprocess

try:  # optional: ffmpeg binaries installed via pip
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

from manifest import Unavailable, atomic_write_json

# yt-dlp error text that means the video will never download (owner made it private, removed it,
# or it needs a membership). These become the manifest status 'unavailable' instead of a retry.
_PERMANENT = ("private video", "video is private", "video unavailable", "has been removed",
              "no longer available", "members-only", "join this channel", "account associated",
              "has been terminated", "copyright claim", "deleted video")
_PLACEHOLDER_TITLES = {"[private video]", "[deleted video]"}


def is_permanent_error(message: str) -> bool:
    msg = (message or "").lower()
    return any(p in msg for p in _PERMANENT)


def list_playlist(playlist_id: str) -> dict:
    """{'playlist_id', 'playlist_name', 'videos': [{'id','title','duration'}]}"""
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/playlist?list={playlist_id}", download=False)
    videos = [
        {"id": e["id"], "title": e.get("title"), "duration": e.get("duration"),
         "listed_unavailable": (e.get("title") or "").strip().lower() in _PLACEHOLDER_TITLES}
        for e in info.get("entries", []) if e and e.get("id")
    ]
    return {"playlist_id": playlist_id, "playlist_name": info.get("title"), "videos": videos}


def download_audio(video_id: str, out_wav: str) -> None:
    """Download best audio, convert to 16 kHz mono 16-bit wav, then atomically move into place."""
    from yt_dlp import YoutubeDL

    out_dir = os.path.dirname(out_wav) or "."
    os.makedirs(out_dir, exist_ok=True)
    opts = {"format": "bestaudio/best", "outtmpl": os.path.join(out_dir, f"{video_id}.src.%(ext)s"),
            "quiet": True, "noprogress": True}
    from yt_dlp.utils import DownloadError

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            src = ydl.prepare_filename(info)
    except DownloadError as e:
        if is_permanent_error(str(e)):
            raise Unavailable(str(e)) from None
        raise
    tmp = out_wav + ".tmp"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", "-f", "wav", tmp],
        check=True,
    )
    os.replace(tmp, out_wav)
    os.remove(src)


def fetch_captions(video_id: str, out_json: str, langs: list[str]) -> bool:
    """Best-effort YouTube captions in the same JSON format 01_fetch_transcripts.py wrote. Never raises."""
    if os.path.exists(out_json):
        return True
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tl = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            tl = YouTubeTranscriptApi().list(video_id)
        transcript = None
        for lang in langs:
            try:
                transcript = tl.find_transcript([lang])
                break
            except Exception:
                continue
        if transcript is None:
            print(f"   [captions] {video_id}: no caption track in {langs}")
            return False
        entries = []
        for e in transcript.fetch():
            if isinstance(e, dict):
                entries.append({"text": e["text"], "start": e["start"], "duration": e["duration"]})
            else:
                entries.append({"text": getattr(e, "text", ""), "start": getattr(e, "start", 0.0),
                                "duration": getattr(e, "duration", 0.0)})
        atomic_write_json(out_json, {"video_id": video_id, "language": transcript.language_code,
                                     "is_generated": transcript.is_generated, "entries": entries})
        return True
    except Exception as e:
        print(f"   [captions] {video_id}: none ({type(e).__name__})")
        return False
