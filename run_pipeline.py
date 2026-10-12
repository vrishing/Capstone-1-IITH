"""
Resumable pipeline runner.

  python run_pipeline.py --ids data/dev_subset.txt          # download + ASR for the dev subset
  python run_pipeline.py                                    # whole playlist (config.YOUTUBE_PLAYLIST_ID)
  python run_pipeline.py --status                           # manifest summary and failures
  python run_pipeline.py --ids data/dev_subset.txt --retry-failed

Kill it at any time (Ctrl+C or kill -9) and run the same command again: finished
stages are skipped, the interrupted one restarts from scratch.
Stages corrected / chunked / indexed are registered here as they get built.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import config
from manifest import (STAGES, Manifest, Stage, atomic_write_json, clean_tmp_files,
                      fingerprint, process_video)

REQUIRED = ["ASR_OUTPUT_DIR", "CAPTIONS_DIR", "TRANSCRIPT_LANGS", "TARGET_LANGUAGE", "HF_TOKEN",
            "ASR_DECODER", "ASR_MAX_CHUNK_S", "ASR_TOP_DB", "ASR_PAD_S", "MANIFEST_PATH", "MAX_ATTEMPTS"]
LOW_WPS = 0.8  # chunks slower than this many words/second are flagged as possible under-transcription

_engine = None


def wav_path(v: str) -> str:
    return os.path.join(config.ASR_OUTPUT_DIR, f"{v}.wav")


def asr_path(v: str) -> str:
    return os.path.join(config.ASR_OUTPUT_DIR, f"{v}_asr.json")


def captions_path(v: str) -> str:
    return os.path.join(config.CAPTIONS_DIR, f"{v}.json")


# ---------- stage implementations ----------

def run_download(v: str) -> None:
    from downloader import download_audio, fetch_captions
    download_audio(v, wav_path(v))
    fetch_captions(v, captions_path(v), config.TRANSCRIPT_LANGS)  # baseline only; never fails the stage


def get_engine():
    global _engine
    if _engine is None:  # load the model only if there is ASR work to do
        from asr_engine import IndicConformerASR
        _engine = IndicConformerASR(config.TARGET_LANGUAGE, config.ASR_DECODER, config.HF_TOKEN,
                                    config.ASR_MAX_CHUNK_S, config.ASR_TOP_DB, config.ASR_PAD_S)
    return _engine


def run_asr(v: str) -> None:
    res = get_engine().transcribe(wav_path(v))   # raises on failure
    segs = res["segments"]
    words = sum(len(s["text"].split()) for s in segs)
    if res["duration_s"] > 120 and words == 0:
        raise RuntimeError("empty transcript for a long recording")
    low = [[s["start"], s["duration"], len(s["text"].split())] for s in segs
           if s["duration"] > 3 and len(s["text"].split()) / s["duration"] < LOW_WPS]
    res["video_id"] = v
    res["engine_used"] = res["engine"]
    res["stats"] = {"n_segments": len(segs), "n_words": words,
                    "rtf": round(res["latency_sec"] / max(res["duration_s"], 1e-6), 3),
                    "low_wps_segments": low}
    atomic_write_json(asr_path(v), res)
    print(f"   [asr] {v}: {words} words, RTF {res['stats']['rtf']}, {len(low)} low-wps chunks")


def build_stages() -> dict[str, Stage]:
    return {
        "downloaded": Stage("downloaded", lambda v: [wav_path(v)], run_download,
                            lambda: {"audio": "wav16k_mono_pcm16"}),
        "asr_done": Stage("asr_done", lambda v: [asr_path(v)], run_asr,
                          lambda: {"model": "indic-conformer-600m", "decoder": config.ASR_DECODER,
                                   "max_chunk_s": config.ASR_MAX_CHUNK_S, "top_db": config.ASR_TOP_DB,
                                   "pad_s": config.ASR_PAD_S, "precision": "fp32",
                                   "language": config.TARGET_LANGUAGE}),
        # "corrected": ..., "chunked": ..., "indexed": ...  (register as each is built)
    }


# ---------- CLI ----------

def read_ids(path: str) -> list[str]:
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", line)
        ids.append(m.group(1) if m else line)
    return list(dict.fromkeys(ids))  # de-duplicate, keep order


def print_status(m: Manifest) -> None:
    print("stage        done  failed")
    for s, c in m.summary().items():
        print(f"{s:<12} {c['done']:>5} {c['failed']:>7}")
    for f in m.failures():
        print(f"  FAILED {f['video_id']} @ {f['stage']} (attempts {f['attempts']}): {f['error']}")


def main(argv=None, stages: dict[str, Stage] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="file with one video id (or URL) per line")
    ap.add_argument("--through", default="asr_done", choices=STAGES)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, help="process only the first N videos")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    missing = [n for n in REQUIRED if not hasattr(config, n)]
    if not args.ids and not args.status and not hasattr(config, "YOUTUBE_PLAYLIST_ID"):
        missing.append("YOUTUBE_PLAYLIST_ID")
    if missing:
        print("config.py is missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    m = Manifest(config.MANIFEST_PATH)
    if args.status:
        print_status(m)
        return 0

    stages = stages or build_stages()
    unbuilt = [s for s in STAGES[: STAGES.index(args.through) + 1] if s not in stages]
    if unbuilt:
        print(f"stages not implemented yet: {unbuilt}. Use --through with a built stage.", file=sys.stderr)
        return 2

    removed = clean_tmp_files(config.ASR_OUTPUT_DIR, config.CAPTIONS_DIR)
    if removed:
        print(f"removed {removed} leftover .tmp file(s) from an interrupted run")

    if args.ids:
        ids = read_ids(args.ids)
    else:
        from downloader import list_playlist
        pl = list_playlist(config.YOUTUBE_PLAYLIST_ID)
        atomic_write_json(os.path.join(os.path.dirname(config.MANIFEST_PATH) or ".", "playlist_meta.json"), pl)
        ids = [v["id"] for v in pl["videos"]]
    if args.limit:
        ids = ids[: args.limit]

    for i, v in enumerate(ids, 1):
        res = process_video(m, stages, v, args.through, config.MAX_ATTEMPTS, args.retry_failed)
        print(f"[{i}/{len(ids)}] {v}: " + ", ".join(f"{k}={r}" for k, r in res.items()), flush=True)

    print()
    print_status(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
