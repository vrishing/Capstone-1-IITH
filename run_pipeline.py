"""
Resumable pipeline runner.

  python run_pipeline.py --add-playlist "<playlist url>" --lang hi --key nptel-dsa   # register a playlist
  python run_pipeline.py --playlist mahabharata --sample 10      # whole playlist, 10 spread videos first
  python run_pipeline.py --playlist nptel-dsa                    # another playlist, its own language
  python run_pipeline.py --playlist mahabharata --status         # summary for one playlist
  python run_pipeline.py --ids data/dev_subset.txt               # explicit IDs (language of --playlist)
  python run_pipeline.py --playlist mahabharata --retry-failed   # also re-checks 'unavailable' videos

Without --playlist the registry's default (playlists.json) is used.

Kill it at any time (Ctrl+C or kill -9) and run the same command again: finished
stages are skipped, the interrupted one restarts from scratch.
Stages: downloaded, asr_done, corrected, chunked are registered; indexed is registered when it is built
(use --through chunked until then).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import config
from manifest import (STAGES, Manifest, Stage, atomic_write_bytes, atomic_write_json,
                      clean_tmp_files, fingerprint, process_video)

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


def _dir(name: str, sub: str) -> str:
    return getattr(config, name, os.path.join(os.path.dirname(config.MANIFEST_PATH) or ".", sub))


def corrected_path(v: str) -> str:
    return os.path.join(_dir("CORRECTED_DIR", "corrected"), f"{v}_corrected.json")


def spans_path(v: str) -> str:
    return os.path.join(_dir("SPANS_DIR", "spans"), f"{v}_spans.json")


def correction_cfg() -> dict:
    return {"auto": getattr(config, "CORRECTION_AUTO_SCORE", 92.0),
            "review": getattr(config, "CORRECTION_REVIEW_SCORE", 86.0),
            "margin": getattr(config, "CORRECTION_MARGIN", 5.0),
            "drop_leading_chars": getattr(config, "CORRECTION_DROP_LEADING_CHARS", 2),
            "use_llm": getattr(config, "CORRECTION_USE_LLM", False),
            "llm_audit_auto": getattr(config, "CORRECTION_LLM_AUDIT_AUTO", True),
            "llm_batch": getattr(config, "CORRECTION_LLM_BATCH", 50),
            "language": getattr(config, "TARGET_LANGUAGE", ""),
            "topic": playlist_topic()}


def playlist_topic() -> str:
    """What the playlist is about, for the LLM: the playlist name plus the glossary pack's topic."""
    name = (getattr(config, "PLAYLIST_NAME", "") or "").strip()
    topic = ""
    pack = getattr(config, "DOMAIN_PACK_PATH", None)
    if pack and os.path.exists(pack):
        try:
            with open(pack, encoding="utf-8") as f:
                topic = (json.load(f).get("topic") or "").strip()
        except (OSError, ValueError):
            pass
    if topic and topic.lower() not in name.lower():
        return f"{name} ({topic})" if name else topic
    return name or topic


def llm_cache_path() -> str:
    return os.path.join(_dir("CORRECTED_DIR", "corrected"), "_llm_cache.json")


def correction_llm():
    """The generate(prompt) function the corrector uses: JSON output, role 'verifier'."""
    from llm_client import generate_default
    return lambda prompt: generate_default(prompt, role="verifier", json_output=True,
                                           max_output_tokens=8000)


def span_settings() -> dict:
    return {"min_s": getattr(config, "SPAN_MIN_S", 45.0), "max_s": getattr(config, "SPAN_MAX_S", 90.0),
            "pause_s": getattr(config, "SPAN_PAUSE_S", 1.0)}


def _glossary():
    import correction
    return correction.get_glossary(getattr(config, "ALIAS_TABLE_PATH", None),
                                   getattr(config, "PROTECTED_WORDS_PATH", None),
                                   getattr(config, "DOMAIN_PACK_PATH", None))


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
    res = get_engine().transcribe(wav_path(v), language_code=config.TARGET_LANGUAGE)  # raises on failure
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


def run_correct(v: str) -> None:
    import correction
    cfg = correction_cfg()
    gen = correction_llm() if cfg["use_llm"] else None
    summ = correction.correct_file(asr_path(v), corrected_path(v), _glossary(), cfg, gen, llm_cache_path())
    print(f"   [correct] {v}: {summ['n_changes']} changes, {summ['n_review']} for review, "
          f"{len(summ['dropped_leading'])} leading noise segment(s) dropped")


def run_chunk(v: str) -> None:
    from spans import spans_from_asr_file, write_spans
    st = span_settings()
    spans = spans_from_asr_file(corrected_path(v), st)   # picks up raw_segments, keeps raw_text per span
    if not spans:
        raise RuntimeError("no spans produced")
    import entities
    entities.tag_spans(spans, _glossary())                # entity ids per span (a retrieval aid only)
    write_spans(spans_path(v), spans, st)
    print(f"   [chunk] {v}: {len(spans)} spans, {sum(1 for s in spans if s['entities'])} tagged")


def build_stages() -> dict[str, Stage]:
    return {
        "downloaded": Stage("downloaded", lambda v: [wav_path(v)], run_download,
                            lambda: {"audio": "wav16k_mono_pcm16"}),
        "asr_done": Stage("asr_done", lambda v: [asr_path(v)], run_asr,
                          lambda: {"model": "indic-conformer-600m", "decoder": config.ASR_DECODER,
                                   "max_chunk_s": config.ASR_MAX_CHUNK_S, "top_db": config.ASR_TOP_DB,
                                   "pad_s": config.ASR_PAD_S, "precision": "fp32", "timing": "speech_bounds_v1",
                                   "language": config.TARGET_LANGUAGE}),
        "corrected": Stage("corrected", lambda v: [corrected_path(v)], run_correct,
                           lambda: __import__("correction").stage_settings(
                               _glossary(), correction_cfg(), getattr(config, "TARGET_LANGUAGE", ""))),
        "chunked": Stage("chunked", lambda v: [spans_path(v)], run_chunk,
                         lambda: {"spans": "spans_v1", "entities": __import__("entities").ENTITY_ALGO,
                                 "glossary": _glossary().version, **span_settings()}),
        # "indexed": ...  (register when it is built)
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


def meta_path() -> str:
    return getattr(config, "PLAYLIST_META_PATH",
                   os.path.join(os.path.dirname(config.MANIFEST_PATH) or ".", "playlist_meta.json"))


def cached_playlist() -> dict | None:
    p = meta_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def fetch_playlist() -> dict:
    """List the playlist online and cache it; fall back to the cached copy if listing fails."""
    from downloader import list_playlist
    try:
        pl = list_playlist(config.YOUTUBE_PLAYLIST_ID)
    except Exception as e:
        pl = cached_playlist()
        if pl is None:
            raise
        print(f"playlist listing failed ({type(e).__name__}); using the cached list of {len(pl['videos'])} videos")
        return pl
    pl["language"] = config.TARGET_LANGUAGE
    atomic_write_json(meta_path(), pl)
    return pl


def order_with_sample(ids: list[str], n: int) -> list[str]:
    """Put a fixed, evenly spread sample first. The sample is saved once and reused, so the
    videos kept out of evaluation never change between runs."""
    import playlists
    path = getattr(config, "SAMPLE_IDS_PATH", None)
    sample: list[str] = []
    if path and os.path.exists(path):
        sample = [v for v in read_ids(path) if v in ids]
    if not sample:
        sample = playlists.spread_sample(ids, n)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            atomic_write_bytes(path, ("# glossary sample: exclude from evaluation\n"
                                      + "\n".join(sample) + "\n").encode("utf-8"))
        print(f"glossary sample ({len(sample)} videos) saved to {path}")
    return sample + [v for v in ids if v not in sample]


def print_status(m: Manifest, ids: list[str] | None = None) -> None:
    label = f" for {getattr(config, 'PLAYLIST_KEY', '')} ({len(ids)} videos)" if ids is not None else ""
    print(f"status{label}")
    print("stage        done  failed  unavailable")
    summ = m.summary(ids)
    for s, c in summ.items():
        print(f"{s:<12} {c['done']:>5} {c['failed']:>7} {c['unavailable']:>12}")
    for f in m.failures(ids, "unavailable"):
        print(f"  UNAVAILABLE {f['video_id']}: {f['error'][:120]}")
    for f in m.failures(ids):
        print(f"  FAILED {f['video_id']} @ {f['stage']} (attempts {f['attempts']}): {f['error']}")
    if ids is not None:
        dl = summ["downloaded"]
        settled = dl["done"] + dl["unavailable"]
        state = "COMPLETE" if settled == len(ids) else f"INCOMPLETE ({len(ids) - settled} not downloaded yet)"
        print(f"downloads: {dl['done']} downloaded + {dl['unavailable']} unavailable of {len(ids)} -> {state}")


def main(argv=None, stages: dict[str, Stage] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", help="registry name, playlist URL or ID (default: playlists.json default)")
    ap.add_argument("--add-playlist", metavar="URL", help="register a new playlist and exit")
    ap.add_argument("--lang", help="with --add-playlist: IndicConformer language code (hi, ta, te, ...)")
    ap.add_argument("--key", help="with --add-playlist: short name to use with --playlist")
    ap.add_argument("--make-default", action="store_true", help="with --add-playlist: make it the default")
    ap.add_argument("--sample", type=int, default=0,
                    help="process N evenly spread videos first (saved as the glossary sample)")
    ap.add_argument("--ids", help="file with one video id (or URL) per line")
    ap.add_argument("--through", default="asr_done", choices=STAGES)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, help="process only the first N videos")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    if args.add_playlist:
        return add_playlist(args)
    if args.playlist:
        import playlists
        try:
            config.select_playlist(args.playlist)
        except playlists.RegistryError as e:
            print(str(e), file=sys.stderr)
            return 2

    missing = [n for n in REQUIRED if not hasattr(config, n)]
    if not args.ids and not args.status and not hasattr(config, "YOUTUBE_PLAYLIST_ID"):
        missing.append("YOUTUBE_PLAYLIST_ID")
    if missing:
        print("config.py is missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    m = Manifest(config.MANIFEST_PATH)
    if args.status:
        if args.ids:
            print_status(m, read_ids(args.ids))
        else:
            pl = cached_playlist() if hasattr(config, "PLAYLIST_META_PATH") else None
            print_status(m, [v["id"] for v in pl["videos"]] if pl else None)
        return 0

    stages = stages or build_stages()
    unbuilt = [s for s in STAGES[: STAGES.index(args.through) + 1] if s not in stages]
    if unbuilt:
        print(f"stages not implemented yet: {unbuilt}. Use --through with a built stage.", file=sys.stderr)
        return 2

    removed = clean_tmp_files(config.ASR_OUTPUT_DIR, config.CAPTIONS_DIR,
                              _dir("CORRECTED_DIR", "corrected"), _dir("SPANS_DIR", "spans"))
    if removed:
        print(f"removed {removed} leftover .tmp file(s) from an interrupted run")

    if args.ids:
        ids = read_ids(args.ids)
    else:
        pl = fetch_playlist()
        ids = [v["id"] for v in pl["videos"]]
    all_ids = list(ids)
    if args.sample:
        ids = order_with_sample(ids, args.sample)
    if args.limit:
        ids = ids[: args.limit]

    if "corrected" in STAGES[: STAGES.index(args.through) + 1] and "corrected" in stages \
            and correction_cfg()["use_llm"]:
        # One playlist-wide LLM pass, so the per-video stage finds every verdict cached.
        import correction
        files = [asr_path(v) for v in ids if os.path.exists(asr_path(v))]
        try:
            correction.prefill_llm_cache(files, _glossary(), correction_cfg(), correction_llm(), llm_cache_path())
        except Exception as e:
            print(f"LLM review failed ({type(e).__name__}: {str(e)[:300]}). Verdicts answered so far are "
                  f"saved. Run again later, or set CORRECTION_USE_LLM=0 to correct without the LLM.",
                  file=sys.stderr)
            return 2

    print(f"playlist {getattr(config, 'PLAYLIST_KEY', 'ids')}: "
          f"{len(ids)} videos, language {config.TARGET_LANGUAGE}, through {args.through}")
    for i, v in enumerate(ids, 1):
        res = process_video(m, stages, v, args.through, config.MAX_ATTEMPTS, args.retry_failed)
        print(f"[{i}/{len(ids)}] {v}: " + ", ".join(f"{k}={r}" for k, r in res.items()), flush=True)

    print()
    print_status(m, all_ids)
    return 0


def add_playlist(args) -> int:
    import playlists
    try:
        pid = playlists.parse_playlist_id(args.add_playlist)
    except playlists.RegistryError as e:
        print(str(e), file=sys.stderr)
        return 2
    pl = None
    try:
        from downloader import list_playlist
        pl = list_playlist(pid)
    except Exception as e:
        print(f"could not list the playlist online ({type(e).__name__}: {e})", file=sys.stderr)
    lang = args.lang
    if not lang and pl:
        lang = playlists.guess_language([pl.get("playlist_name") or ""] + [v["title"] or "" for v in pl["videos"]])
        if lang:
            print(f"language guessed from the titles' script: {lang} ({playlists.ASR_LANGUAGES[lang]}). "
                  f"Pass --lang to override.")
    if not lang:
        print("Cannot tell the language from the titles (English or a shared script such as "
              "Devanagari). Pass --lang, e.g. --lang hi.", file=sys.stderr)
        return 2
    try:
        key, entry = playlists.add(args.add_playlist, lang, key=args.key,
                                   name=(pl or {}).get("playlist_name"), make_default=args.make_default)
        config.select_playlist(key)
    except playlists.RegistryError as e:
        print(str(e), file=sys.stderr)
        return 2
    if pl:
        pl["language"] = entry["language"]
        atomic_write_json(meta_path(), pl)
        gone = sum(1 for v in pl["videos"] if v.get("listed_unavailable"))
        print(f"{len(pl['videos'])} videos listed" + (f" ({gone} private or deleted)" if gone else ""))
    print(f"registered {key!r}: {entry}")
    print(f"next: python run_pipeline.py --playlist {key} --sample 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())