"""
Correction layer: fixes mis-heard glossary names in ASR text, conservatively, and logs every change.

    gl = load_glossary("glossary/<id>/aliases.csv")          # missing or empty file -> empty glossary
    correct_file("data/asr_output/X_asr.json", "data/corrected/X_corrected.json", gl, cfg)

Rules (frozen in the handoff):
  - Whole words only (multi-word glossary terms match whole word groups). Never inside a word.
  - A word that already equals a glossary term, or is an inflected form of one, is left alone.
  - Matching uses romanize.phonetic_key, so one glossary works for every Brahmic script.
  - Inflection is kept: a mis-heard stem is fixed and the case ending stays as the speaker said it. This
    path is strict: the stem key must be at least 6 letters and identical to the term's stem key, and the
    ending must be one this same transcript shows on a correctly spelled form of a glossary name.
  - A key shorter than 5 letters carries too little information (శ్రమ "effort" and శ్రీరామ share the key
    "sram"): it is auto-applied only when the spelling itself differs by one character, else it is reviewed.
  - Score is normalised edit similarity of the phonetic keys (0-100). Auto-apply at cfg["auto"] (92) when
    the best entity leads the runner-up entity by cfg["margin"] (5). From cfg["review"] (86) up, or when
    two entities are too close, the proposal goes to the review list and the text is NOT changed.
  - Rows with needs_context=1 and types listed in domain_pack.json "no_fuzzy_types" match exactly only.
  - Protected words are never rewritten: every glossary spelling, every word inside a multi-word
    spelling, plus the lines of protected_words.txt when that file exists.
  - Other names of the same entity are separate rows. They are never rewritten into each other.
  - LLM review is off by default. When on, it only accepts or rejects proposals the code made
    (mid-confidence ones). It never writes text. A parse error rejects; an API error propagates so the
    stage fails and retries. Decisions are cached.
  - Leading 1-2 character segments (jingle or noise before the speech starts) are dropped, and listed
    under correction.dropped_leading. Set drop_leading_chars to 0 to keep them.
  - The corrected file has the ASR file's shape plus `raw_segments` (one to one with `segments`),
    `changes`, `review` and a `correction` summary. The raw text is never lost.

Stdlib plus romanize.py. Dry run and probe:
    python correction.py dry-run data/asr_output/*_asr.json
    python correction.py probe ordinary_words.txt        # words that WOULD be proposed: should be none
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata

from romanize import phonetic_key, script_of

ALGO_VERSION = "correction_v1"
DEFAULTS = {"auto": 92.0, "review": 86.0, "margin": 5.0, "min_key_len": 4, "max_suffix": 5,
            "min_stem_chars": 3, "min_inflect_key_len": 6, "min_auto_key_len": 5, "drop_leading_chars": 2, "drop_leading_max": 3, "use_llm": False}
MAX_PHRASE_WORDS = 4


# ======================================================================================
# Text helpers
# ======================================================================================
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Mc")


def _is_virama(ch: str) -> bool:
    return unicodedata.combining(ch) == 9


def stem_ends(word: str) -> list[int]:
    """Positions where a stem may end: never before a vowel sign, anusvara or inside a conjunct."""
    out = [i for i in range(1, len(word)) if not _is_mark(word[i]) and not _is_virama(word[i - 1])]
    out.append(len(word))
    return out


def _split_punct(token: str) -> tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation) by Unicode category, so marks stay in words."""
    def punct(c: str) -> bool:
        return unicodedata.category(c)[0] in "PS"
    a, b = 0, len(token)
    while a < b and punct(token[a]):
        a += 1
    while b > a and punct(token[b - 1]):
        b -= 1
    return token[:a], token[a:b], token[b:]


def _lev(a: str, b: str, k: int) -> int | None:
    """Edit distance, or None when it is larger than k."""
    if abs(len(a) - len(b)) > k:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > k:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= k else None


def similarity(a: str, b: str, review: float) -> float:
    """Normalised edit similarity 0-100; 0 when it is below the review band (not worth computing)."""
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    d = _lev(a, b, int(m * (1 - review / 100.0) + 1e-9))
    return 0.0 if d is None else 100.0 * (1 - d / m)


# ======================================================================================
# Glossary
# ======================================================================================
class Term:
    __slots__ = ("term", "entity_id", "roman", "type", "key", "stem", "stem_key", "script", "nwords")

    def __init__(self, row: dict):
        self.term = _nfc(row["term"])
        self.entity_id = (row.get("entity_id") or "").strip()
        self.roman = (row.get("roman") or "").strip()
        self.type = (row.get("type") or "").strip()
        self.key = phonetic_key(self.term)
        self.script = script_of(self.term)
        self.nwords = len(self.term.split())
        self.stem, self.stem_key = "", ""
        if self.nwords == 1:
            ends = stem_ends(self.term)
            if len(ends) >= 2:                       # drop the last syllable: the nominative ending
                self.stem = self.term[:ends[-2]]
                self.stem_key = phonetic_key(self.stem)


class Glossary:
    def __init__(self):
        self.exact: set[str] = set()
        self.protected: set[str] = set()
        self.variants: dict[str, list[Term]] = {}
        self.single: list[Term] = []                 # fuzzy-matchable single-word terms
        self.phrases: list[Term] = []                # fuzzy-matchable multi-word terms
        self.phrase_texts: set[str] = set()
        self.stems: set[str] = set()                 # stems of every single-word term
        self.protected_stems: set[str] = set()       # stems of the words in protected_words.txt
        self.version = "empty"
        self.n_rows = 0

    def __bool__(self) -> bool:
        return bool(self.exact or self.variants)


def _split_variants(cell: str) -> list[str]:
    return [_nfc(v) for v in re.split(r"[|;,]", cell or "") if _nfc(v)]


def _read_lines(path: str | None) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [_nfc(line.split("#")[0]) for line in f if _nfc(line.split("#")[0])]


def load_glossary(alias_path: str | None, protected_path: str | None = None,
                  pack_path: str | None = None, min_key_len: int = DEFAULTS["min_key_len"]) -> Glossary:
    """A missing, unreadable or empty alias table gives an empty glossary: text passes through."""
    gl = Glossary()
    if not alias_path or not os.path.exists(alias_path) or os.path.getsize(alias_path) == 0:
        return gl
    with open(alias_path, "rb") as f:
        blob = f.read()
    with open(alias_path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if _nfc(r.get("term", ""))]
    if not rows:
        return gl
    no_fuzzy = {"other"}
    if pack_path and os.path.exists(pack_path):
        try:
            with open(pack_path, encoding="utf-8") as f:
                no_fuzzy |= set(json.load(f).get("no_fuzzy_types", []))
        except ValueError:
            pass
    extra = _read_lines(protected_path)
    for r in rows:
        t = Term(r)
        gl.exact.add(t.term)
        gl.protected.add(t.term)
        if t.nwords > 1:
            gl.protected.update(t.term.split())
            gl.phrase_texts.add(t.term)
        elif t.stem:
            gl.stems.add(t.stem)
        exact_only = (r.get("needs_context") or "").strip() == "1" or t.type in no_fuzzy
        if not exact_only and len(t.key) >= min_key_len and t.script:
            (gl.phrases if t.nwords > 1 else gl.single).append(t)
        for v in _split_variants(r.get("variants", "")):
            gl.variants.setdefault(v, []).append(t)
    for v in list(gl.variants):                      # a variant that is itself a spelling is not a misspelling
        if v in gl.exact:
            del gl.variants[v]
    gl.protected.update(extra)
    for w in extra:                                  # a protected name is also protected when inflected
        ends = stem_ends(w)
        if " " not in w and len(ends) >= 2 and ends[-2] >= DEFAULTS["min_stem_chars"]:
            gl.protected_stems.add(w[:ends[-2]])
    gl.n_rows = len(rows)
    h = hashlib.sha256(blob)
    h.update("\n".join(sorted(extra)).encode("utf-8"))
    h.update(",".join(sorted(no_fuzzy)).encode("utf-8"))
    gl.version = h.hexdigest()[:12]
    return gl


_CACHE: dict = {}


def get_glossary(alias_path, protected_path=None, pack_path=None) -> Glossary:
    """load_glossary with a per-process cache keyed by file mtimes and sizes."""
    def sig(p):
        try:
            s = os.stat(p)
            return (p, s.st_mtime_ns, s.st_size)
        except (OSError, TypeError):
            return (p, None, None)
    k = (sig(alias_path), sig(protected_path), sig(pack_path))
    if k not in _CACHE:
        _CACHE.clear()
        _CACHE[k] = load_glossary(alias_path, protected_path, pack_path)
    return _CACHE[k]


# ======================================================================================
# Matching
# ======================================================================================
class Corrector:
    def __init__(self, gl: Glossary, cfg: dict | None = None):
        self.gl = gl
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._keys: dict[str, str] = {}
        self._stemkeys: dict[str, list[tuple[int, str]]] = {}
        self._memo: dict[str, tuple] = {}
        self.endings: set[str] = set()               # case endings seen on correct forms in this file

    def learn_endings(self, segments: list[dict]) -> None:
        """Endings (the part after a glossary stem) that the transcript itself uses on correct forms."""
        ms, mx = self.cfg["min_stem_chars"], self.cfg["max_suffix"]
        words = {_nfc(_split_punct(t)[1]) for seg in segments for t in seg["text"].split()}
        ends = set()
        for w in words:
            for st in self.gl.stems:
                if len(st) >= ms and w.startswith(st) and 1 <= len(w) - len(st) <= mx:
                    ends.add(w[len(st):])
        self.endings = ends
        self._memo.clear()

    def key(self, s: str) -> str:
        k = self._keys.get(s)
        if k is None:
            k = self._keys[s] = phonetic_key(s)
        return k

    def _stem_keys(self, w: str) -> list[tuple[int, str]]:
        r = self._stemkeys.get(w)
        if r is None:
            r = []
            for i in stem_ends(w):
                suffix = len(w) - i
                if 1 <= suffix <= self.cfg["max_suffix"] and i >= self.cfg["min_stem_chars"]:
                    k = self.key(w[:i])
                    if len(k) >= self.cfg["min_key_len"]:
                        r.append((i, k))
            self._stemkeys[w] = r
        return r

    def _inflected_form_of_known(self, w: str) -> bool:
        ms = self.cfg["min_stem_chars"]
        for s in self.gl.stems:
            if len(s) >= ms and w.startswith(s) and 1 <= len(w) - len(s) <= self.cfg["max_suffix"]:
                return True
        return False

    def _protected_inflection(self, w: str) -> bool:
        mx = self.cfg["max_suffix"]
        return any(w.startswith(st) and 1 <= len(w) - len(st) <= mx for st in self.gl.protected_stems)

    def _decide(self, hits: list[tuple], raw: str | None = None, key: str = "") -> tuple | None:
        """hits: (score, Term, new_text, kind). Returns (action, best, runner_up_score, reason) or None."""
        if not hits:
            return None
        per_entity: dict[str, tuple] = {}
        for h in hits:
            cur = per_entity.get(h[1].entity_id)
            if cur is None or h[0] > cur[0]:
                per_entity[h[1].entity_id] = h
        ranked = sorted(per_entity.values(), key=lambda h: -h[0])
        best = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else 0.0
        if best[0] >= self.cfg["auto"]:
            if best[0] - runner < self.cfg["margin"]:
                return ("review", best, runner, "ambiguous: two entities score within the margin")
            if (raw is not None and best[3] != "variant" and len(key) < self.cfg["min_auto_key_len"]
                    and _lev(raw, best[2], 1) is None):
                return ("review", best, runner, "short key: too little information to auto-apply")
            return ("apply", best, runner, "")
        if best[0] >= self.cfg["review"]:
            return ("review", best, runner, "mid-confidence")
        return None

    def propose_word(self, w: str) -> tuple | None:
        """Decision for one word, or None to leave it alone. Memoised per distinct word."""
        if w in self._memo:
            return self._memo[w]
        out = self._propose_word(w)
        self._memo[w] = out
        return out

    def _propose_word(self, w: str) -> tuple | None:
        gl = self.gl
        if not w or w in gl.protected or w in gl.exact or not script_of(w) or self._protected_inflection(w):
            return None
        if w in gl.variants:                          # a known misspelling: exact, score 100
            ents = {t.entity_id: t for t in gl.variants[w]}
            hits = [(100.0, t, t.term, "variant") for t in ents.values()]
            return self._decide(hits)
        wk = self.key(w)
        if len(wk) < self.cfg["min_key_len"] or self._inflected_form_of_known(w):
            return None
        review = self.cfg["review"]
        ws = script_of(w)
        stems = self._stem_keys(w)
        hits = []
        for t in gl.single:
            if t.script != ws:
                continue
            best = (similarity(wk, t.key, review), t.term, "whole")
            if t.stem_key and self.endings and len(t.stem_key) >= self.cfg["min_inflect_key_len"]:
                for i, sk in stems:
                    if sk == t.stem_key and w[i:] in self.endings and 100.0 > best[0]:
                        best = (100.0, t.stem + w[i:], "inflected")
            if best[0] >= review:
                hits.append((best[0], t, best[1], best[2]))
        return self._decide(hits, w, wk)

    def propose_phrase(self, phrase: str) -> tuple | None:
        key = "\x00" + phrase
        if key in self._memo:
            return self._memo[key]
        out = None
        gl = self.gl
        if phrase not in gl.phrase_texts and script_of(phrase):
            pk, ps = self.key(phrase), script_of(phrase)
            n = len(phrase.split())
            hits = []
            for t in gl.phrases:
                if t.nwords == n and t.script == ps:
                    s = similarity(pk, t.key, self.cfg["review"])
                    if s >= self.cfg["review"]:
                        hits.append((s, t, t.term, "phrase"))
            out = self._decide(hits)
        self._memo[key] = out
        return out


# ======================================================================================
# Optional LLM review (off by default)
# ======================================================================================
def _llm_prompt(raw: str, term: Term, context: str) -> str:
    return ("A speech recogniser may have misheard a name. Decide if the word is a mis-transcription "
            f"of the glossary name.\nSentence: {context}\nWord: {raw}\nGlossary name: {term.term} "
            f"({term.roman})\nReply with JSON only: {{\"accept\": true}} if the word is clearly the "
            "glossary name misheard, otherwise {\"accept\": false}. When unsure, reject.")


def llm_accepts(raw: str, term: Term, context: str, generate, cache: dict) -> bool:
    key = hashlib.sha256(f"{raw}|{term.term}|{context}".encode("utf-8")).hexdigest()
    if key in cache:
        return bool(cache[key])
    text = generate(_llm_prompt(raw, term, context))     # API errors propagate: the stage retries
    try:
        from llm_client import extract_json
        verdict = json.loads(extract_json(text)).get("accept") is True
    except Exception:                                    # parse error counts as reject
        verdict = False
    cache[key] = verdict
    return verdict


# ======================================================================================
# Correcting segments and files
# ======================================================================================
def drop_leading_noise(segments: list[dict], max_chars: int, max_n: int) -> tuple[int, list[dict]]:
    """How many leading segments are 1..max_chars-character noise (never all of them)."""
    if max_chars <= 0:
        return 0, []
    n = 0
    while n < min(max_n, len(segments) - 1) and len(segments[n]["text"].strip()) <= max_chars:
        n += 1
    return n, [dict(s) for s in segments[:n]]


def correct_segments(segments: list[dict], corrector: Corrector, generate=None,
                     llm_cache: dict | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (corrected segments, changes, review). Input segments are not modified."""
    gl, cfg = corrector.gl, corrector.cfg
    if not corrector.endings:
        corrector.learn_endings(segments)
    out, changes, review = [], [], []
    use_llm = bool(cfg["use_llm"] and generate is not None)
    cache = llm_cache if llm_cache is not None else {}
    for si, seg in enumerate(segments):
        new = dict(seg)
        parts = re.findall(r"\S+|\s+", seg["text"])
        widx = [i for i, p in enumerate(parts) if not p.isspace()]
        splits = {i: _split_punct(parts[i]) for i in widx}          # token index -> (pre, core, post)
        cores = [_nfc(splits[i][1]) for i in widx]
        used = [False] * len(widx)
        edits: dict[int, str] = {}                                   # position in widx -> new core text
        blanks: set[int] = set()                                     # words absorbed into a phrase edit

        def record(pos0, pos1, raw, best, runner, reason, action):
            hit_score, t, new_text, kind = best
            info = {"segment": si, "word": pos0, "raw": raw, "term": t.term, "new": new_text,
                    "entity_id": t.entity_id, "score": round(hit_score, 1), "rule": kind}
            if action == "apply":
                changes.append(info)
                edits[pos0] = new_text
                blanks.update(range(pos0 + 1, pos1))
                for p in range(pos0, pos1):
                    used[p] = True
                return
            info["reason"] = reason
            info["runner_up"] = round(runner, 1)
            ctx = " ".join(cores[max(0, pos0 - 5):pos1 + 5])
            if use_llm and reason == "mid-confidence" and llm_accepts(raw, t, ctx, generate, cache):
                info.pop("reason"), info.pop("runner_up")
                info["rule"] = "llm_review"
                changes.append(info)
                edits[pos0] = new_text
                blanks.update(range(pos0 + 1, pos1))
                for p in range(pos0, pos1):
                    used[p] = True
                return
            review.append(info)

        for n in range(min(MAX_PHRASE_WORDS, len(cores)), 1, -1):    # multi-word terms first
            if not gl.phrases and not gl.phrase_texts:
                break
            p = 0
            while p + n <= len(cores):
                if any(used[p:p + n]):
                    p += 1
                    continue
                phrase = " ".join(cores[p:p + n])
                if phrase in gl.phrase_texts:                        # already correct: leave it alone
                    for q in range(p, p + n):
                        used[q] = True
                    p += n
                    continue
                d = corrector.propose_phrase(phrase)
                if d:
                    record(p, p + n, phrase, d[1], d[2], d[3], d[0])
                p += 1
        for p, w in enumerate(cores):
            if used[p]:
                continue
            d = corrector.propose_word(w)
            if d:
                record(p, p + 1, w, d[1], d[2], d[3], d[0])

        if edits:
            for pos, text in edits.items():
                ti = widx[pos]
                pre, _core, post = splits[ti]
                parts[ti] = pre + text + post
            for q in blanks:                                         # a phrase edit absorbs its later words
                parts[widx[q]] = ""
            text = "".join(parts)
            new["text"] = re.sub(r"\s{2,}", " ", text).strip() if blanks else text
        out.append(new)
    return out, changes, review


def correct_file(asr_path: str, out_path: str, gl: Glossary, cfg: dict | None = None,
                 generate=None, llm_cache_path: str | None = None) -> dict:
    """Read one *_asr.json, write the corrected file atomically. Returns the 'correction' summary."""
    from manifest import atomic_write_json
    cfg = {**DEFAULTS, **(cfg or {})}
    with open(asr_path, encoding="utf-8") as f:
        res = json.load(f)
    segs = res["segments"]
    n_drop, dropped = drop_leading_noise(segs, cfg["drop_leading_chars"], cfg["drop_leading_max"])
    raw = [dict(s) for s in segs[n_drop:]]
    cache: dict = {}
    if cfg["use_llm"] and llm_cache_path and os.path.exists(llm_cache_path):
        with open(llm_cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    cor = Corrector(gl, cfg)
    cor.learn_endings(raw)
    fixed, changes, review = correct_segments(raw, cor, generate, cache)
    out = copy.deepcopy({k: v for k, v in res.items() if k != "segments"})
    out["segments"] = fixed
    out["raw_segments"] = raw
    summary = {"version": ALGO_VERSION, "glossary_version": gl.version, "glossary_rows": gl.n_rows,
               "thresholds": {k: cfg[k] for k in ("auto", "review", "margin")},
               "n_changes": len(changes), "n_review": len(review),
               "dropped_leading": dropped, "llm_review": bool(cfg["use_llm"] and generate is not None)}
    out["correction"] = summary
    out["changes"] = changes
    out["review"] = review
    atomic_write_json(out_path, out)
    if cfg["use_llm"] and llm_cache_path and cache:
        atomic_write_json(llm_cache_path, cache)
    return summary


def stage_settings(gl: Glossary, cfg: dict, language: str = "") -> dict:
    """What defines the 'corrected' stage fingerprint: change any of these and the stage reruns."""
    c = {**DEFAULTS, **cfg}
    return {"algo": ALGO_VERSION, "glossary": gl.version, "auto": c["auto"], "review": c["review"],
            "margin": c["margin"], "max_suffix": c["max_suffix"], "min_key_len": c["min_key_len"],
            "min_inflect_key_len": c["min_inflect_key_len"], "min_auto_key_len": c["min_auto_key_len"],
            "drop_leading_chars": c["drop_leading_chars"], "llm": bool(c["use_llm"]), "language": language}


# ======================================================================================
# Command line
# ======================================================================================
def _glossary_from_config():
    import config
    return get_glossary(config.ALIAS_TABLE_PATH, config.PROTECTED_WORDS_PATH, config.DOMAIN_PACK_PATH)


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("dry-run", "probe"):
        print(__doc__)
        return 2
    gl = _glossary_from_config()
    print(f"glossary: {gl.n_rows} rows, version {gl.version}, {len(gl.single)} fuzzy words, "
          f"{len(gl.phrases)} fuzzy phrases")
    if not gl:
        print("empty glossary: text would pass through unchanged")
    cor = Corrector(gl)
    if argv[0] == "probe":
        words = _read_lines(argv[1])
        hits = [(w, cor.propose_word(_nfc(w))) for w in words]
        hits = [(w, d) for w, d in hits if d]
        print(f"{len(words)} ordinary words, {len(hits)} proposed (should be 0)")
        for w, d in hits:
            print(f"  {w} -> {d[1][2]} [{d[0]}, {d[1][0]:.0f}, {d[3] or d[1][3]}]")
        return 0
    total_c = total_r = 0
    seen: dict[str, int] = {}
    for p in argv[1:]:
        with open(p, encoding="utf-8") as f:
            segs = json.load(f)["segments"]
        _, ch, rv = correct_segments(segs, cor)
        total_c += len(ch)
        total_r += len(rv)
        for c in ch:
            seen[f"{c['raw']} -> {c['new']}"] = seen.get(f"{c['raw']} -> {c['new']}", 0) + 1
        print(f"{os.path.basename(p)}: {len(ch)} changes, {len(rv)} for review")
    print(f"total: {total_c} changes, {total_r} for review")
    for k, n in sorted(seen.items(), key=lambda x: -x[1])[:25]:
        print(f"  {n:>4}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))