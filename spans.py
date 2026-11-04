"""
Evidence spans: whole ASR segments grouped into 45-90 s pieces, cut at long pauses, no overlap.

    spans = build_spans(video_id, segments, language="te")
    write_spans(path, spans)

Each span is what gets embedded, retrieved and cited:
  span_id       "<video_id>_<start ms, 9 digits>", stable while the segmentation is unchanged
  point_id      deterministic UUID from span_id (Qdrant point ID; reindexing never duplicates)
  video_id, start, end, segment_range [first, last] (indices into the segment list)
  text          joined segment text (corrected text once the corrector runs; raw before that)
  raw_text      the uncorrected text, when corrected segments are passed with their raw ones
  language, type ("content"), entities [], cluster_id None (filled by later stages)

Cutting rule (no threshold tuning needed, so it works for any speaker or language):
  - A span is never longer than max_s, except a single segment that is longer on its own.
  - Once a span is at least min_s long, it ends at the first pause of at least pause_s, or, when
    there is none before max_s, at the LARGEST pause between min_s and max_s. ASR segments are
    padded into the silence around them, so real pauses show up as small gaps; the largest one in
    the window is still the most natural place to cut.
  - Cuts that would leave a final piece shorter than min_s / 2 are avoided when another cut works.
Stdlib only.
"""
from __future__ import annotations

import json
import os
import uuid

SPAN_NAMESPACE = uuid.UUID("6f1c2d4e-8a7b-4c3d-9e2f-1a2b3c4d5e6f")  # never change: point IDs depend on it
DEFAULTS = {"min_s": 45.0, "max_s": 90.0, "pause_s": 0.7}


def point_id(span_id: str) -> str:
    return str(uuid.uuid5(SPAN_NAMESPACE, span_id))


def _end(seg: dict) -> float:
    return float(seg["start"]) + float(seg["duration"])


def _gap(a: dict, b: dict) -> float:
    """Pause between two consecutive segments. Uses the speech bounds the ASR stage records (the
    chunk padding hides the real silence); falls back to the plain gap for older transcripts."""
    if "speech_end" in a and "speech_start" in b:
        return max(0.0, float(b["speech_start"]) - float(a["speech_end"]))
    return max(0.0, float(b["start"]) - _end(a))


def _make(video_id: str, segs: list[dict], first: int, last: int, language: str,
          raw: list[dict] | None) -> dict:
    start = float(segs[first]["start"])
    end = _end(segs[last])
    sid = f"{video_id}_{int(round(start * 1000)):09d}"
    span = {
        "span_id": sid,
        "point_id": point_id(sid),
        "video_id": video_id,
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "segment_range": [first, last],
        "text": " ".join(s["text"].strip() for s in segs[first:last + 1] if s["text"].strip()),
        "language": language,
        "type": "content",
        "entities": [],
        "cluster_id": None,
    }
    if raw is not None:
        span["raw_text"] = " ".join(s["text"].strip() for s in raw[first:last + 1] if s["text"].strip())
    return span


def build_spans(video_id: str, segments: list[dict], language: str = "",
                min_s: float = DEFAULTS["min_s"], max_s: float = DEFAULTS["max_s"],
                pause_s: float = DEFAULTS["pause_s"], raw_segments: list[dict] | None = None) -> list[dict]:
    """segments: [{'text', 'start', 'duration'}] in time order (the ASR or corrector output)."""
    if min_s <= 0 or max_s < min_s:
        raise ValueError("need 0 < min_s <= max_s")
    if raw_segments is not None and len(raw_segments) != len(segments):
        raise ValueError("raw_segments must line up one-to-one with segments")
    segs = sorted(segments, key=lambda s: float(s["start"]))
    if raw_segments is not None and segs != list(segments):
        raise ValueError("segments must already be in time order when raw_segments is given")
    groups: list[tuple[int, int]] = []
    n, a = len(segs), 0
    last_end = _end(segs[-1]) if segs else 0.0
    while a < n:
        start = float(segs[a]["start"])
        if last_end - start <= max_s:               # the rest fits in one span
            groups.append((a, n - 1))
            break
        cands = []                                  # (j, gap after j, tail left after cutting at j)
        for j in range(a, n - 1):
            length = _end(segs[j]) - start
            if length > max_s:
                break
            if length >= min_s:
                cands.append((j, _gap(segs[j], segs[j + 1]),
                              last_end - float(segs[j + 1]["start"])))
        if not cands:                               # no segment boundary inside [min_s, max_s]
            j = a
            while j + 1 < n and _end(segs[j + 1]) - start <= max_s:
                j += 1
            groups.append((a, j))
            a = j + 1
            continue
        # Never leave a short scrap at the end of the video if a cut can avoid it.
        pool = [c for c in cands if c[2] >= min_s / 2] or cands
        long_pause = [c for c in pool if c[1] >= pause_s]
        j = long_pause[0][0] if long_pause else max(pool, key=lambda c: (c[1], -c[0]))[0]
        groups.append((a, j))
        a = j + 1
    return [_make(video_id, segs, a, b, language, raw_segments) for a, b in groups]


def write_spans(path: str, spans: list[dict], settings: dict | None = None) -> None:
    from manifest import atomic_write_json
    atomic_write_json(path, {"settings": settings or DEFAULTS, "n_spans": len(spans), "spans": spans})


def spans_from_asr_file(asr_json_path: str, settings: dict | None = None) -> list[dict]:
    """Spans for one *_asr.json (or a corrected file with the same 'segments' shape)."""
    with open(asr_json_path, encoding="utf-8") as f:
        res = json.load(f)
    st = {**DEFAULTS, **(settings or {})}
    vid = res.get("video_id") or os.path.basename(asr_json_path).split("_")[0]
    raw = res.get("raw_segments")                    # present once the corrector writes both
    return build_spans(vid, res["segments"], res.get("language", ""), st["min_s"], st["max_s"],
                       st["pause_s"], raw_segments=raw)


if __name__ == "__main__":
    # Dry run on transcripts you already have:  python spans.py data/asr_output/*_asr.json
    import sys
    try:
        import config
        cli = {"min_s": config.SPAN_MIN_S, "max_s": config.SPAN_MAX_S, "pause_s": config.SPAN_PAUSE_S}
    except Exception:
        cli = None
    used = {**DEFAULTS, **(cli or {})}
    print(f"settings: min {used['min_s']} s, max {used['max_s']} s, pause {used['pause_s']} s"
          + ("" if cli else "  (config.py not readable: defaults)"))
    for p in sys.argv[1:]:
        sp = spans_from_asr_file(p, cli)
        d = [s["duration"] for s in sp]
        with open(p, encoding="utf-8") as f:
            sg = sorted(json.load(f)["segments"], key=lambda s: float(s["start"]))
        gaps = sorted(_gap(a, b) for a, b in zip(sg, sg[1:])) or [0.0]
        has = "speech_end" in sg[0] if sg else False
        q = lambda f: gaps[min(len(gaps) - 1, int(f * len(gaps)))]
        print(f"{os.path.basename(p)}: {len(sp)} spans, {min(d, default=0):.0f}-{max(d, default=0):.0f} s, "
              f"mean {sum(d) / max(len(d), 1):.0f} s | {'true pauses' if has else 'padded gaps (re-run ASR for true pauses)'} median {q(0.5):.2f} s, p90 {q(0.9):.2f} s, "
              f"max {gaps[-1]:.2f} s, {sum(g >= used['pause_s'] for g in gaps)} >= {used['pause_s']} s")
