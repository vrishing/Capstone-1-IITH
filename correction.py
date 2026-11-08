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
  - Three guards against ordinary words that sound like names (అనుమతి "permission" and భానుమతి):
      onset   the heard word and the name must start with the same sound (first letter of the key,
              for every word of a phrase). ASR rarely mishears the first sound of a name it knows.
      short   a key under 5 letters (శ్రమ "effort" and శ్రీరామ both give "sram") is fixed only when the
              spelling differs by one character; otherwise it is dropped, not even reviewed.
      present a proposal below the auto score is kept for review only when the same name also appears
              correctly spelled (any inflection, any spelling of that entity) elsewhere in the video.
  - Score is normalised edit similarity of the phonetic keys (0-100). Auto-apply at cfg["auto"] (92) when
    the best entity leads the runner-up entity by cfg["margin"] (5). From cfg["review"] (86) up, or when
    two entities are too close, the proposal goes to the review list and the text is NOT changed.
  - Rows with needs_context=1 and types listed in domain_pack.json "no_fuzzy_types" match exactly only.
  - Protected words are never rewritten: every glossary spelling, every word inside a multi-word
    spelling, plus the lines of protected_words.txt when that file exists.
  - Other names of the same entity are separate rows. They are never rewritten into each other.
  - LLM review is off by default (use_llm). When on, every distinct (heard word, proposed fix) pair the
    code generated is judged once: mid-confidence pairs and, with llm_audit_auto, auto-band pairs too.
    "misheard" applies the fix; anything else keeps the speaker's word. Pairs go in batches of
    llm_batch with two example sentences each, and verdicts are cached per (language, heard, fix), so
    a playlist costs a handful of calls and a rerun costs none. The LLM never writes text. A parse
    error or a missing answer rejects (not cached, so it is asked again next run); an API error
    propagates so the stage fails and retries. Exact variants and ambiguous pairs are never sent.
  - Leading 1-2 character segments (jingle or noise before the speech starts) are dropped, and listed
    under correction.dropped_leading. Set drop_leading_chars to 0 to keep them.
  - The corrected file has the ASR file's shape plus `raw_segments` (one to one with `segments`),
    `changes`, `review` and a `correction` summary. The raw text is never lost.

Stdlib plus romanize.py. Commands (PLAYLIST=<key> picks the playlist):
    python correction.py dry-run data/asr_output/*_asr.json   # what would change, no LLM, nothing written
    python correction.py probe ordinary_words.txt             # words that WOULD be proposed: should be none
    python correction.py llm-review                           # LLM verdict per distinct pair, fills the cache
    python correction.py review-export / review-apply         # decide pairs by hand in a CSV

Turning the review list into decisions (set PLAYLIST=<key> first):
    python correction.py review-export     # glossary/<id>/review_decisions.csv, frequent pairs prefilled accept
    # edit the decision column: accept / reject / blank
    python correction.py review-apply      # accept -> variants in aliases.csv, reject -> protected_words.txt
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
            "min_stem_chars": 3, "min_inflect_key_len": 6, "min_auto_key_len": 5, "drop_leading_chars": 2, "drop_leading_max": 3, "use_llm": False,
            "llm_audit_auto": True, "llm_batch": 50, "language": "", "topic": ""}
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


def same_onset(a: str, b: str) -> bool:
    """Every word of the two keys starts with the same letter (and the word counts match)."""
    wa, wb = a.split(), b.split()
    return len(wa) == len(wb) and all(x[:1] == y[:1] for x, y in zip(wa, wb))


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
    __slots__ = ("term", "entity_id", "roman", "type", "key", "stem", "stem_key", "script", "nwords", "desc")

    def __init__(self, row: dict):
        self.term = _nfc(row["term"])
        self.entity_id = (row.get("entity_id") or "").strip()
        self.roman = (row.get("roman") or "").strip()
        self.type = (row.get("type") or "").strip()
        self.desc = (row.get("notes") or "").split("|")[0].strip()[:90]   # Wikidata description
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
        self.term_entities: dict[str, set[str]] = {}  # every spelling (and its stem) -> entity ids
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
        gl.term_entities.setdefault(t.term, set()).add(t.entity_id)
        gl.exact.add(t.term)
        gl.protected.add(t.term)
        if t.nwords > 1:
            gl.protected.update(t.term.split())
            gl.phrase_texts.add(t.term)
        elif t.stem:
            gl.stems.add(t.stem)
            gl.term_entities.setdefault("\x01" + t.stem, set()).add(t.entity_id)
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
        self.present: set[str] | None = None         # entities spelled correctly in this file (None: unknown)

    def learn(self, segments: list[dict]) -> None:
        """Learn from one transcript: the case endings it uses on correctly spelled names, and which
        names it spells correctly at least once. Call once per video, before correcting it."""
        gl = self.gl
        ms, mx = self.cfg["min_stem_chars"], self.cfg["max_suffix"]
        words = {_nfc(_split_punct(t)[1]) for seg in segments for t in seg["text"].split()}
        ends, present = set(), set()
        for w in words:
            present |= gl.term_entities.get(w, set())
            for st in gl.stems:
                if len(st) >= ms and w.startswith(st) and 1 <= len(w) - len(st) <= mx:
                    ends.add(w[len(st):])
                    present |= gl.term_entities.get("\x01" + st, set())
        if gl.phrase_texts:
            joined = " " + " ".join(_nfc(seg["text"]) for seg in segments) + " "
            for ph in gl.phrase_texts:
                if f" {ph} " in joined:
                    present |= gl.term_entities.get(ph, set())
        self.endings, self.present = ends, present
        self._memo.clear()

    learn_endings = learn                            # old name

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
                return None                          # short key: too little information, not even review
            return ("apply", best, runner, "")
        if best[0] >= self.cfg["review"]:
            if self.present is not None and best[1].entity_id not in self.present:
                return None                          # the name never appears correctly in this video
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
            if t.script != ws or t.key[:1] != wk[:1]:
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
                if t.nwords == n and t.script == ps and same_onset(pk, t.key):
                    s = similarity(pk, t.key, self.cfg["review"])
                    if s >= self.cfg["review"]:
                        hits.append((s, t, t.term, "phrase"))
            out = self._decide(hits, phrase, pk)
        self._memo[key] = out
        return out


# ======================================================================================
# Optional LLM review (off by default)
# ======================================================================================
LLM_INSTRUCTIONS = """{intro} A matcher found words that sound like names or terms
from that topic's glossary. For each item decide:
  "misheard": the heard word is the glossary term mis-transcribed (same thing, wrong spelling).
  "keep": the heard word is an ordinary word of the language, a different person, place or thing
          (even one with a similar name), or a correct alternative form or spelling of the same
          name. It must not be rewritten.
Judge each word in the setting of the playlist topic, using the glossary description and the example
sentences: ask whether the speaker, talking about this topic, most plausibly meant the glossary term.
When unsure, answer "keep".
Reply with JSON only: {{"items": [{{"id": 1, "verdict": "misheard"}}, {{"id": 2, "verdict": "keep"}}, ...]}}
with one entry for every id."""


def llm_key(language: str, raw: str, new: str, topic: str = "") -> str:
    """Cache key. A different topic gives different keys, so verdicts never carry over between topics."""
    scope = language if not topic else f"{language}~{hashlib.sha1(topic.encode('utf-8')).hexdigest()[:6]}"
    return f"{scope}\t{raw}\t{new}"


def _llm_batch_prompt(items: list[dict], language: str, topic: str = "") -> str:
    intro = (f"A speech recogniser transcribed spoken discourse from a playlist about: {topic}."
             if topic else "A speech recogniser transcribed a lecture or discourse.")
    lines = [LLM_INSTRUCTIONS.format(intro=intro), f"Language code: {language or 'unknown'}", ""]
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. heard: {it['raw']} | glossary term: {it['term']}"
                     + (f" ({it['roman']})" if it.get("roman") else "")
                     + (f", {it['desc']}" if it.get("desc") else "")
                     + (f" | fix would give: {it['new']}" if it["new"] != it["term"] else ""))
        for ex in it["examples"][:2]:
            lines.append(f"   example: {ex}")
    return "\n".join(lines)


def llm_review_pairs(items: list[dict], generate, cache: dict, language: str = "",
                     batch: int = 40, on_batch=None, topic: str = "") -> dict[str, bool]:
    """items: [{raw, new, term, roman, examples}] (distinct by raw/new). Fills cache with verdicts
    (True = misheard) and returns {llm_key: verdict} for every item. on_batch(cache) runs after every
    answered batch, so an API error later (quota, network) never loses verdicts already paid for.
    API errors propagate."""
    from llm_client import extract_json
    out: dict[str, bool] = {}
    todo = []
    for it in items:
        k = llm_key(language, it["raw"], it["new"], topic)
        if k in cache:
            out[k] = bool(cache[k])
        else:
            todo.append(it)
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        text = generate(_llm_batch_prompt(chunk, language, topic))   # API error: raise, stage retries
        try:
            answers = json.loads(extract_json(text)).get("items", [])
            got = {int(a["id"]): a.get("verdict") for a in answers if isinstance(a, dict) and "id" in a}
        except Exception:                                            # parse error: reject, ask again next run
            got = {}
        for i, it in enumerate(chunk, 1):
            k = llm_key(language, it["raw"], it["new"], topic)
            v = got.get(i)
            if v in ("misheard", "keep"):
                cache[k] = out[k] = (v == "misheard")
            else:
                out[k] = False                                       # missing or odd answer: reject, not cached
        if on_batch is not None:
            on_batch(cache)
    return out


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


def find_proposals(segments: list[dict], corrector: Corrector) -> tuple[list[tuple], list[tuple]]:
    """Pass 1: every proposal in the transcript, nothing applied. Call corrector.learn() first.
    Returns (props, tokens): props are (segment, word, end word, info, action, reason, runner_up, Term)."""
    gl = corrector.gl
    props = []                                       # (si, pos0, pos1, info, action, reason, runner)
    tokens = []                                      # per segment: (parts, widx, splits, cores)
    for si, seg in enumerate(segments):
        parts = re.findall(r"\S+|\s+", seg["text"])
        widx = [i for i, p in enumerate(parts) if not p.isspace()]
        splits = {i: _split_punct(parts[i]) for i in widx}
        cores = [_nfc(splits[i][1]) for i in widx]
        tokens.append((parts, widx, splits, cores))
        used = [False] * len(widx)

        def found(pos0, pos1, raw, d):
            action, best, runner, reason = d
            hit_score, t, new_text, kind = best
            info = {"segment": si, "word": pos0, "raw": raw, "term": t.term, "new": new_text,
                    "entity_id": t.entity_id, "score": round(hit_score, 1), "rule": kind}
            props.append((si, pos0, pos1, info, action, reason, runner, t))
            if action == "apply":                    # claimed words are not matched again
                for q in range(pos0, pos1):
                    used[q] = True

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
                    found(p, p + n, phrase, d)
                p += 1
        for p, w in enumerate(cores):
            if used[p]:
                continue
            d = corrector.propose_word(w)
            if d:
                found(p, p + 1, w, d)

    return props, tokens


def llm_items(props: list[tuple], tokens: list[tuple], cfg: dict, items: dict | None = None) -> dict:
    """The distinct (heard, fix) pairs that go to the LLM, with up to two example sentences and an
    occurrence count. Pass `items` to merge several transcripts into one set."""
    items = {} if items is None else items
    lang, topic = cfg.get("language", ""), cfg.get("topic", "")
    for si, pos0, pos1, info, action, reason, runner, t in props:
        if info["rule"] == "variant":
            continue                                 # curated exact rewrite: trusted
        if not (reason == "mid-confidence" or (action == "apply" and cfg["llm_audit_auto"])):
            continue                                 # ambiguous pairs stay with a human
        k = llm_key(lang, info["raw"], info["new"], topic)
        it = items.setdefault(k, {"raw": info["raw"], "new": info["new"], "term": t.term, "roman": t.roman,
                                  "desc": t.desc, "score": info["score"], "examples": [], "n": 0})
        it["n"] += 1
        if len(it["examples"]) < 2:
            it["examples"].append(" ".join(tokens[si][3][max(0, pos0 - 8):pos1 + 8]))
    return items


def load_llm_cache(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def prefill_llm_cache(asr_files: list[str], gl: "Glossary", cfg: dict, generate, cache_path: str,
                      log=print) -> dict:
    """Ask about every distinct pair of a whole playlist at once, so the per-video stage finds every
    verdict cached and makes no calls. A playlist costs ceil(new pairs / llm_batch) calls. The cache
    is saved after every batch. API errors propagate after saving what was answered."""
    from manifest import atomic_write_json
    cfg = {**DEFAULTS, **cfg}
    items: dict[str, dict] = {}
    lang_used = [""]
    for p in asr_files:
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        c = {**cfg, "language": cfg.get("language") or res.get("language", "")}
        lang_used[0] = lang_used[0] or c["language"]
        n_drop, _ = drop_leading_noise(res["segments"], c["drop_leading_chars"], c["drop_leading_max"])
        segs = res["segments"][n_drop:]
        cor = Corrector(gl, c)
        cor.learn(segs)
        props, tokens = find_proposals(segs, cor)
        llm_items(props, tokens, c, items)
    cache = load_llm_cache(cache_path)
    before = len(cache)
    new = [it for k, it in items.items() if k not in cache]
    batch = int(cfg["llm_batch"])
    log(f"llm review: {len(asr_files)} transcripts, {len(items)} distinct pairs, {len(new)} not cached "
        f"-> {-(-len(new) // batch)} call(s) of up to {batch}")
    calls = [0]

    def save(c):
        calls[0] += 1
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        atomic_write_json(cache_path, c)
    todo = [it for k, it in items.items() if k not in cache]
    if todo:
        llm_review_pairs(todo, generate, cache, lang_used[0], batch, on_batch=save, topic=cfg.get("topic", ""))
    return {"items": items, "cache": cache, "calls": calls[0], "new_verdicts": len(cache) - before}


def correct_segments(segments: list[dict], corrector: Corrector, generate=None,
                     llm_cache: dict | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (corrected segments, changes, review). Input segments are not modified.

    Pass 1 finds proposals in every segment, pass 2 (optional) asks the LLM about the distinct pairs,
    pass 3 applies the accepted ones."""
    gl, cfg = corrector.gl, corrector.cfg
    corrector.learn(segments)
    use_llm = bool(cfg["use_llm"] and generate is not None)
    cache = llm_cache if llm_cache is not None else {}
    lang, topic = cfg.get("language", ""), cfg.get("topic", "")

    props, tokens = find_proposals(segments, corrector)

    # ---- pass 2: LLM verdicts for distinct pairs (normally all cached by prefill_llm_cache) ----
    verdicts: dict[str, bool] = {}
    if use_llm:
        items = llm_items(props, tokens, cfg)
        verdicts = llm_review_pairs(list(items.values()), generate, cache, lang, int(cfg["llm_batch"]),
                                    topic=topic)

    # ---- pass 3: decide and apply ----
    changes, review = [], []
    edits = [dict() for _ in segments]               # per segment: word position -> new text
    blanks = [set() for _ in segments]
    for si, pos0, pos1, info, action, reason, runner, t in props:
        k = llm_key(lang, info["raw"], info["new"], topic)
        if k in verdicts:
            if verdicts[k]:
                if action != "apply":
                    info["rule"] = "llm_review"
                info["llm"] = "misheard"
                action = "apply"
            else:
                info["llm"] = "keep"
                action, reason = "review", "llm: keep the heard word"
        if action == "apply":
            changes.append(info)
            edits[si][pos0] = info["new"]
            blanks[si].update(range(pos0 + 1, pos1))
        else:
            info["reason"] = reason
            info["runner_up"] = round(runner, 1)
            review.append(info)
    out = []
    for si, seg in enumerate(segments):
        new = dict(seg)
        if edits[si]:
            parts, widx, splits, _ = tokens[si]
            parts = list(parts)
            for pos, text in edits[si].items():
                pre, _core, post = splits[widx[pos]]
                parts[widx[pos]] = pre + text + post
            for q in blanks[si]:                                    # a phrase edit absorbs its later words
                parts[widx[q]] = ""
            text = "".join(parts)
            new["text"] = re.sub(r"\s{2,}", " ", text).strip() if blanks[si] else text
        out.append(new)
    changes.sort(key=lambda c: (c["segment"], c["word"]))
    review.sort(key=lambda c: (c["segment"], c["word"]))
    return out, changes, review


def correct_file(asr_path: str, out_path: str, gl: Glossary, cfg: dict | None = None,
                 generate=None, llm_cache_path: str | None = None) -> dict:
    """Read one *_asr.json, write the corrected file atomically. Returns the 'correction' summary."""
    from manifest import atomic_write_json
    cfg = {**DEFAULTS, **(cfg or {})}
    with open(asr_path, encoding="utf-8") as f:
        res = json.load(f)
    cfg["language"] = cfg.get("language") or res.get("language", "")
    segs = res["segments"]
    n_drop, dropped = drop_leading_noise(segs, cfg["drop_leading_chars"], cfg["drop_leading_max"])
    raw = [dict(s) for s in segs[n_drop:]]
    cache: dict = {}
    if cfg["use_llm"] and llm_cache_path and os.path.exists(llm_cache_path):
        with open(llm_cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    cor = Corrector(gl, cfg)
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
        os.makedirs(os.path.dirname(llm_cache_path) or ".", exist_ok=True)
        atomic_write_json(llm_cache_path, cache)
    return summary


def stage_settings(gl: Glossary, cfg: dict, language: str = "") -> dict:
    """What defines the 'corrected' stage fingerprint: change any of these and the stage reruns."""
    c = {**DEFAULTS, **cfg}
    return {"algo": ALGO_VERSION, "glossary": gl.version, "auto": c["auto"], "review": c["review"],
            "margin": c["margin"], "max_suffix": c["max_suffix"], "min_key_len": c["min_key_len"],
            "min_inflect_key_len": c["min_inflect_key_len"], "min_auto_key_len": c["min_auto_key_len"],
            "guards": "onset+short+present_v1",
            "drop_leading_chars": c["drop_leading_chars"], "language": language,
            "llm": (f"pairs_v2:audit={bool(c['llm_audit_auto'])}:topic={c['topic']}" if c["use_llm"] else False)}


# ======================================================================================
# Command line
# ======================================================================================
# ======================================================================================
# Review decisions: turn the review list into variants (accept) or protected words (reject)
# ======================================================================================
DECISION_COLUMNS = ["decision", "count", "raw", "term", "entity_id", "score", "videos", "example"]


def export_review(corrected_files: list[str], out_csv: str, min_count: int = 3) -> dict:
    """Collect review proposals from corrected files into one CSV to decide on.

    One row per (heard word, glossary term). Rows seen at least min_count times are prefilled
    'accept'; the rest are left blank (blank rows are ignored by apply). Write 'reject' for an
    ordinary word. Only whole-word proposals are exported: a variant replaces one exact word.
    An existing file keeps its decisions; new pairs are added."""
    agg: dict[tuple, dict] = {}
    for p in corrected_files:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        vid = d.get("video_id") or os.path.basename(p).split("_")[0]
        segs = d.get("segments", [])
        for r in d.get("review", []):
            if r.get("rule") != "whole" or " " in r["raw"] or r["new"] != r["term"]:
                continue
            k = (r["raw"], r["term"], r["entity_id"])
            a = agg.setdefault(k, {"count": 0, "score": r["score"], "videos": set(), "example": ""})
            a["count"] += 1
            a["videos"].add(vid)
            if not a["example"] and r["segment"] < len(segs):
                a["example"] = segs[r["segment"]]["text"][:120]
    old: dict[tuple, str] = {}
    if os.path.exists(out_csv):
        with open(out_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                old[(r["raw"], r["term"], r["entity_id"])] = (r.get("decision") or "").strip()
    rows = []
    for k, a in sorted(agg.items(), key=lambda x: (-x[1]["count"], x[0])):
        dec = old.get(k, "accept" if a["count"] >= min_count else "")
        rows.append({"decision": dec, "count": a["count"], "raw": k[0], "term": k[1], "entity_id": k[2],
                     "score": a["score"], "videos": len(a["videos"]), "example": a["example"]})
    from manifest import atomic_write_bytes
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=DECISION_COLUMNS)
    w.writeheader()
    w.writerows(rows)
    atomic_write_bytes(out_csv, buf.getvalue().encode("utf-8"))
    return {"pairs": len(rows), "prefilled_accept": sum(r["decision"] == "accept" for r in rows),
            "kept_decisions": sum(1 for r in rows if (r["raw"], r["term"], r["entity_id"]) in old)}


def apply_decisions(decisions_csv: str, alias_path: str, protected_path: str) -> dict:
    """accept -> the heard word becomes a variant of that term's row in aliases.csv (exact rewrite).
    reject -> the heard word goes into protected_words.txt (never rewritten, inflections included).
    Blank rows are skipped. Backs up aliases.csv once as aliases.csv.bak. Safe to run twice."""
    with open(decisions_csv, encoding="utf-8", newline="") as f:
        decs = list(csv.DictReader(f))
    with open(alias_path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames, list(rd)
    if "variants" not in (cols or []):
        raise ValueError(f"{alias_path} has no 'variants' column")
    protected = set(_read_lines(protected_path))
    exact = {_nfc(r["term"]) for r in rows}
    out = {"accepted": 0, "already_variant": 0, "rejected": 0, "already_protected": 0,
           "skipped_blank": 0, "skipped_conflict": [], "term_not_found": []}
    new_protected = []
    for d in decs:
        dec = (d.get("decision") or "").strip().lower()
        raw, term, eid = _nfc(d["raw"]), _nfc(d["term"]), (d.get("entity_id") or "").strip()
        if dec == "reject":
            if raw in protected:
                out["already_protected"] += 1
            else:
                protected.add(raw)
                new_protected.append(raw)
                out["rejected"] += 1
            continue
        if dec != "accept":
            out["skipped_blank"] += 1
            continue
        if raw in exact or raw in protected:            # a real spelling or a protected word: never a variant
            out["skipped_conflict"].append(raw)
            continue
        target = next((r for r in rows if _nfc(r["term"]) == term and r.get("entity_id", "").strip() == eid), None)
        if target is None:
            out["term_not_found"].append(f"{raw} -> {term}")
            continue
        vs = _split_variants(target.get("variants", ""))
        if raw in vs:
            out["already_variant"] += 1
            continue
        target["variants"] = "|".join(vs + [raw])
        out["accepted"] += 1
    from manifest import atomic_write_bytes
    import io
    if out["accepted"]:
        bak = alias_path + ".bak"
        if not os.path.exists(bak):
            with open(alias_path, "rb") as f:
                atomic_write_bytes(bak, f.read())
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        atomic_write_bytes(alias_path, buf.getvalue().encode("utf-8"))
    if new_protected:
        prev = b""
        if os.path.exists(protected_path):
            with open(protected_path, "rb") as f:
                prev = f.read()
        if prev and not prev.endswith(b"\n"):
            prev += b"\n"
        atomic_write_bytes(protected_path, prev + ("\n".join(new_protected) + "\n").encode("utf-8"))
    return out


def _playlist_asr_files() -> list[str]:
    import glob
    import config
    files = sorted(glob.glob(os.path.join(config.ASR_OUTPUT_DIR, "*_asr.json")))
    meta = getattr(config, "PLAYLIST_META_PATH", None)
    if meta and os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            ids = {v["id"] for v in json.load(f)["videos"]}
        files = [p for p in files if os.path.basename(p)[:-len("_asr.json")] in ids]
    return files


def _llm_review_cli(files: list[str]) -> int:
    """Ask the LLM about every distinct pair in these transcripts (default: this playlist's ASR files),
    fill the cache the pipeline uses, and print the verdicts. Changes no transcript."""
    import run_pipeline as rp
    files = files or _playlist_asr_files()
    gl = _glossary_from_config()
    cfg = {**DEFAULTS, **rp.correction_cfg(), "use_llm": True}
    try:
        res = prefill_llm_cache(files, gl, cfg, rp.correction_llm(), rp.llm_cache_path())
    except Exception as e:                           # quota, network: what was answered is saved
        print(f"LLM call failed: {type(e).__name__}: {str(e)[:300]}")
        print("Verdicts answered so far are saved; run the same command again later to continue.")
        return 1
    items, cache = res["items"], res["cache"]
    print(f"{res['calls']} LLM call(s), {res['new_verdicts']} new verdict(s), cache {len(cache)} "
          f"({rp.llm_cache_path()})")
    for verdict, want in (("KEEP", False), ("MISHEARD", True)):
        rows = sorted(((it["n"], it) for k, it in items.items() if k in cache and bool(cache[k]) == want),
                      key=lambda x: -x[0])
        print(f"\n{verdict} ({len(rows)} pairs)")
        for n, it in rows:
            print(f"  {n:>4}  {it['score']:>5}  {it['raw']} -> {it['new']}")
    missing = [it for k, it in items.items() if k not in cache]
    if missing:
        print(f"\nNO ANSWER YET ({len(missing)} pairs): run again to ask about them.")
    print("\nWrong verdict? Put the word in protected_words.txt (never fix) or in the term's variants "
          "column (always fix). Then set CORRECTION_USE_LLM=1 in .env and run the pipeline.")
    return 0


def _glossary_from_config():
    import config
    return get_glossary(config.ALIAS_TABLE_PATH, config.PROTECTED_WORDS_PATH, config.DOMAIN_PACK_PATH)


def _main(argv: list[str]) -> int:
    if argv[:1] in (["review-export"], ["review-apply"]):
        import glob
        import config
        dec_path = os.path.join(config.GLOSSARY_DIR, "review_decisions.csv")
        if argv[0] == "review-export":
            files = argv[1:] or sorted(glob.glob(os.path.join(
                getattr(config, "CORRECTED_DIR", "data/corrected"), "*_corrected.json")))
            ids = None
            meta = getattr(config, "PLAYLIST_META_PATH", None)
            if meta and os.path.exists(meta):              # only this playlist's videos
                with open(meta, encoding="utf-8") as f:
                    ids = {v["id"] for v in json.load(f)["videos"]}
                files = [p for p in files if os.path.basename(p).rsplit("_", 1)[0] in ids]
            res = export_review(files, dec_path)
            print(f"{len(files)} corrected files -> {dec_path}: {res}")
            print("Edit the decision column: accept / reject / blank (skip). Then: "
                  "python correction.py review-apply")
        else:
            res = apply_decisions(dec_path, config.ALIAS_TABLE_PATH, config.PROTECTED_WORDS_PATH)
            print(json.dumps(res, ensure_ascii=False, indent=1))
            print("Glossary changed: the corrected and chunked stages rerun on the next pipeline run.")
        return 0
    if argv[:1] == ["llm-review"]:
        return _llm_review_cli(argv[1:])
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