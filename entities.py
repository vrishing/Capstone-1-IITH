"""
Entity tags and near-duplicate clusters for evidence spans.

    tag_spans(spans, glossary)      -> fills span["entities"] (sorted entity ids found in the text)
    cluster_spans(all_spans)        -> fills span["cluster_id"] (same id = near-duplicate text)
    query_entities(text, glossary)  -> entity ids a question mentions (for retrieval boosts)

Tags are a retrieval AID, never a filter and never evidence: a wrong or missing tag can only
change ranking, the verifier still checks the cited text. Works for any script the glossary
covers. An empty glossary tags nothing. Stdlib only (plus correction/romanize helpers).

Matching, per span:
  - a token equal to a glossary spelling or known variant;
  - an inflected token: a glossary stem (>= min_stem_chars) followed by a short ending;
  - a phrase of 2-4 consecutive tokens equal to a multi-word glossary term.
Ordinary words that merely start like a name are the main risk, so stems must be >= 3 chars and
the remaining ending <= max_suffix chars (same limits as the corrector).

Clusters: word 3-gram shingles, Jaccard >= threshold, union-find. cluster_id is the span_id of the
earliest member (video id, start), so it is deterministic. Singletons get their own span_id.
"""
from __future__ import annotations

import unicodedata

from correction import DEFAULTS as _CD, MAX_PHRASE_WORDS, _split_punct, stem_ends

ENTITY_ALGO = "entities_v1"
CLUSTER_ALGO = "cluster_v1"


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _tokens(text: str) -> list[str]:
    out = []
    for raw in _nfc(text).split():
        core = _split_punct(raw)[1]
        if core:
            out.append(core)
    return out


def _ids(gl, key: str) -> set[str]:
    return {e for e in gl.term_entities.get(key, ()) if e}


def _word_entities(gl, w: str) -> set[str]:
    ids = _ids(gl, w)
    for t in gl.variants.get(w, ()):
        if t.entity_id:
            ids.add(t.entity_id)
    if ids:
        return ids
    min_stem = _CD["min_stem_chars"]
    max_suf = _CD["max_suffix"]
    for e in stem_ends(w):
        if e < min_stem or len(w) - e > max_suf or e == len(w):
            continue
        ids |= _ids(gl, "\x01" + w[:e])
    return ids


def text_entities(text: str, gl) -> list[str]:
    """Sorted entity ids found in text (whole words, inflected forms, and multi-word terms)."""
    if not gl:
        return []
    toks = _tokens(text)
    found: set[str] = set()
    i = 0
    while i < len(toks):
        hit = 0
        for n in range(min(MAX_PHRASE_WORDS, len(toks) - i), 1, -1):
            if " ".join(toks[i:i + n]) in gl.phrase_texts:
                found |= _ids(gl, " ".join(toks[i:i + n]))
                hit = n
                break
        if hit:
            i += hit
            continue
        found |= _word_entities(gl, toks[i])
        i += 1
    return sorted(found)


def tag_spans(spans: list[dict], gl) -> list[dict]:
    for s in spans:
        s["entities"] = text_entities(s["text"], gl)
    return spans


def query_entities(text: str, gl) -> list[str]:
    return text_entities(text, gl)


# ---------------------------------------------------------------------------------------
# near-duplicate clusters
# ---------------------------------------------------------------------------------------
def _shingles(text: str, k: int = 3) -> set[str]:
    w = _tokens(text)
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def cluster_spans(spans: list[dict], threshold: float = 0.8) -> list[dict]:
    """Group near-duplicate spans across videos. Fills cluster_id on every span."""
    ordered = sorted(spans, key=lambda s: (s["video_id"], s["start"]))
    sh = [_shingles(s["text"]) for s in ordered]
    parent = list(range(len(ordered)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    index: dict[str, list[int]] = {}
    for i, s in enumerate(sh):
        if not s:
            continue
        shared: dict[int, int] = {}
        for g in s:
            for j in index.get(g, ()):
                shared[j] = shared.get(j, 0) + 1
        for j, c in shared.items():
            if c / (len(s) + len(sh[j]) - c) >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)       # the earliest span is the root
        for g in s:
            index.setdefault(g, []).append(i)
    for i, sp in enumerate(ordered):
        sp["cluster_id"] = ordered[find(i)]["span_id"]
    return spans