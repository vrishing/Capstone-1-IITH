"""
Checking answer sentences against the passages they cite. Fail-closed: anything not shown to be
supported is "unsupported", and an LLM failure raises (the caller abstains).

    verdicts, debug = verify(question, sents, hits, gen, mode="locate_judge")

sents: [{"text", "cites": [n, ...]}] (every sentence cites at least one passage); hits: the
retrieved passages, hits[n - 1]["text"]; gen(prompt, role, system, json_output, max_output_tokens).

Modes (config.VERIFY_MODE):

  "locate_judge" (default), two LLM calls per answer, both batched over its sentences:
    1. locate: for each sentence the model copies the words of a cited passage that state it. Code
       then finds those words in the REAL passage text, tolerating an inflection or a slip
       (fuzzy, character level). Nothing found -> unsupported, and the judge never sees it.
    2. judge: the model sees the question, the sentence, and only a short window of real passage
       text around each located quote. It first restates in English who did what to whom in that
       window, then gives the verdict.
    Why: NLI-style checking is far more reliable on a short premise than on a long document
    (SummaC, Laban et al. 2022), and the main error of attribution checkers is missing
    fine-grained differences such as who did what (AttributionBench, Li et al. 2024). A 60-90 s
    run-on ASR passage is exactly the long, noisy premise these papers warn about.

  "single", one call: verdict plus a quote per sentence; with require_quote the quote must be
    found in a cited passage. The earlier design, kept for comparison.
"""
from __future__ import annotations

import json
import unicodedata
from difflib import SequenceMatcher

VERDICTS = ("supported", "partial", "unsupported")

SINGLE_SYSTEM = """You check whether statements are supported by passages. Each statement comes with the \
passages it cites (speech-to-text transcripts: names may be misspelled; the statement may be in a different \
language from the passages). Judge only against those passages, never against what you know.
supported   = the passages clearly say what the statement says.
partial     = the passages support part of it, or only imply it, or the statement adds a detail they lack.
unsupported = the passages do not say it, or contradict it.
For every statement judged supported or partial, also give "quote": the exact words copied from one of its \
cited passages (3 to 25 words, no translation, no changes) that show the statement. For unsupported give "".
Reply with JSON only: {"verdicts": [{"i": 0, "verdict": "supported", "quote": "..."}, ...]} with one entry per statement."""

LOCATE_SYSTEM = """You find evidence. Each numbered statement comes with the passages it cites \
(speech-to-text transcripts of lectures: no punctuation, names may be misspelled; the statement may be in \
another language). For each statement, copy the words of ONE cited passage that state what the statement \
says: 3 to 30 consecutive words, exactly as written in the passage, no translation, no changes. A statement \
that needs two places may get two quotes. If no part of its passages states it, give no quote for it.
Reply with JSON only: {"quotes": [{"i": 0, "passage": 3, "quote": "..."}]}"""

JUDGE_SYSTEM = """You check claimed answers against short excerpts of speech-to-text transcripts of lectures \
(no punctuation, names may be misspelled, the excerpt may be in another language than the claim). For each item:
1. "says": one English sentence stating exactly what the excerpt says about the people or things in the claim: \
who did what to whom, keeping each one's role. Use only the excerpt.
2. "verdict", comparing the claim with what the excerpt says:
   supported   = the excerpt states the claim, with the same people in the same roles.
   partial     = the excerpt states part of the claim and nothing in it contradicts the claim.
   unsupported = the excerpt does not state the claim: it is about another person or another role, it \
describes something only planned, vowed, feared, asked or rumoured, it says the opposite, or the claim adds \
facts the excerpt lacks. Shared names or words alone are not support.
The question tells you what the claim is meant to answer. Judge the claim, never what you know of the story.
Reply with JSON only: {"items": [{"i": 0, "says": "...", "verdict": "supported"}]}"""

WINDOW_PAD = 25          # real passage words kept on each side of a located quote
MAX_QUOTES = 2           # per statement


# ---------------------------------------------------------------------------------------
# parsing (tolerant: local models vary the shape)
# ---------------------------------------------------------------------------------------
def _json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        pass
    from llm_client import extract_json
    try:
        return json.loads(extract_json(text))
    except Exception as e:
        raise ValueError(f"not JSON: {str(e)[:80]}")


def _items(d, keys=("items", "verdicts", "results", "quotes")) -> list:
    if isinstance(d, dict):
        for k in keys:
            if isinstance(d.get(k), list):
                return d[k]
        return []
    return d if isinstance(d, list) else []


def parse_verdicts(text: str, n: int) -> list[str]:
    """One verdict per statement. Accepts {"verdicts"|"items": [...]}, a bare list, items like
    {"i": 0, "verdict": "supported"} or plain strings (taken in order). Raises if any is missing."""
    d = _json(text)
    items = _items(d)
    if not isinstance(items, list) or (not items and n):
        raise ValueError("verifier reply has no list of verdicts")
    got: dict[int, str] = {}
    for pos, v in enumerate(items):
        if isinstance(v, str):
            idx, verdict = pos, v
        elif isinstance(v, dict):
            idx = v.get("i", v.get("index", pos))
            verdict = v.get("verdict", v.get("label"))
        else:
            continue
        verdict = str(verdict).strip().lower() if verdict is not None else ""
        if isinstance(idx, int) and verdict in VERDICTS:
            got[idx] = verdict
    if set(got) != set(range(n)):
        raise ValueError("verifier did not answer every statement")
    return [got[i] for i in range(n)]


def parse_field(text: str, n: int, field: str) -> list[str]:
    """Per-statement string field ("quote", "says"); '' when missing. Never raises."""
    try:
        items = _items(_json(text))
    except ValueError:
        return [""] * n
    out = [""] * n
    for pos, v in enumerate(items):
        if isinstance(v, dict):
            idx = v.get("i", v.get("index", pos))
            if isinstance(idx, int) and 0 <= idx < n and not out[idx]:
                out[idx] = str(v.get(field) or "")
    return out


def parse_quotes(text: str, n: int) -> list[list[tuple[int | None, str]]]:
    """For each statement: [(passage number or None, quote), ...]. Never raises."""
    try:
        items = _items(_json(text))
    except ValueError:
        return [[] for _ in range(n)]
    out: list[list] = [[] for _ in range(n)]
    for v in items:
        if not isinstance(v, dict):
            continue
        idx, q = v.get("i", v.get("index")), str(v.get("quote") or "").strip()
        p = v.get("passage")
        if isinstance(idx, int) and 0 <= idx < n and q:
            out[idx].append((p if isinstance(p, int) else None, q))
    return out


# ---------------------------------------------------------------------------------------
# finding a quote in the real passage text
# ---------------------------------------------------------------------------------------
_EDGE = ".,;:!?\"'()[]{}…-–—“”‘’|‌‍"


def _norm(w: str) -> str:
    return unicodedata.normalize("NFC", w).lower().strip(_EDGE)


def locate(quote: str, passage: str, min_words: int = 3, threshold: float = 0.8) -> tuple[int, int] | None:
    """(start, end) word positions in passage.split() of the best fuzzy match of the quote, or None.
    Character-level similarity over windows of about the quote's length, so an inflected first word
    (అర్జునుడి for అర్జునుడు) or one changed vowel sign still matches, while a sentence the model made
    up does not."""
    q = [w for w in (_norm(x) for x in quote.split()) if w]
    p = [_norm(x) for x in unicodedata.normalize("NFC", passage).split()]
    if len(q) < min_words or not p:
        return None
    qs = " ".join(q)
    best, span = 0.0, None
    for w in range(max(1, len(q) - 2), len(q) + 3):
        for s in range(0, max(1, len(p) - w + 1)):
            sm = SequenceMatcher(None, qs, " ".join(p[s:s + w]), autojunk=False)
            floor = max(best, threshold)
            if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
                continue
            r = sm.ratio()
            if r > best:
                best, span = r, (s, min(s + w, len(p)))
    return span if best >= threshold else None


def quote_in_passage(quote: str, passage: str, **kw) -> bool:
    return locate(quote, passage, **kw) is not None


def window(passage: str, span: tuple[int, int], pad: int = WINDOW_PAD) -> str:
    words = unicodedata.normalize("NFC", passage).split()
    a, b = max(0, span[0] - pad), min(len(words), span[1] + pad)
    return ("... " if a else "") + " ".join(words[a:b]) + (" ..." if b < len(words) else "")


# ---------------------------------------------------------------------------------------
# the two modes
# ---------------------------------------------------------------------------------------
def _statements_with_passages(sents: list[dict], hits: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sents):
        passages = "\n".join(f"  passage [{c}]: {hits[c - 1]['text']}" for c in s["cites"])
        parts.append(f"Statement {i}: {s['text']}\n{passages}")
    return "\n\n".join(parts)


def _single(question, sents, hits, gen, require_quote, max_tokens):
    raw = gen(_statements_with_passages(sents, hits), role="verifier", system=SINGLE_SYSTEM,
              json_output=True, max_output_tokens=max_tokens)
    verdicts = parse_verdicts(raw, len(sents))
    debug: dict = {"verification": raw}
    if require_quote:
        quotes = parse_field(raw, len(sents), "quote")
        for i, s in enumerate(sents):
            if verdicts[i] in ("supported", "partial") and not any(
                    quote_in_passage(quotes[i], hits[c - 1]["text"]) for c in s["cites"]):
                debug.setdefault("quote_failures", []).append({"i": i, "quote": quotes[i]})
                verdicts[i] = "unsupported"
    return verdicts, debug


def _locate_judge(question, sents, hits, gen, max_tokens):
    raw = gen(_statements_with_passages(sents, hits), role="verifier", system=LOCATE_SYSTEM,
              json_output=True, max_output_tokens=max_tokens)
    debug: dict = {"locate": raw, "windows": {}}
    quotes = parse_quotes(raw, len(sents))
    windows: dict[int, list[str]] = {}
    for i, s in enumerate(sents):
        for p, q in quotes[i]:
            where = ([p] + [c for c in s["cites"] if c != p]) if p in s["cites"] else s["cites"]
            for c in where:
                span = locate(q, hits[c - 1]["text"])
                if span:
                    windows.setdefault(i, []).append(window(hits[c - 1]["text"], span))
                    break
            if len(windows.get(i, [])) >= MAX_QUOTES:
                break
        if i not in windows:
            debug.setdefault("quote_failures", []).append({"i": i, "quote": " | ".join(q for _, q in quotes[i])})
    verdicts = ["unsupported"] * len(sents)
    todo = sorted(windows)
    debug["windows"] = {i: windows[i] for i in todo}
    if not todo:
        return verdicts, debug
    items = []
    for j, i in enumerate(todo):
        ex = "\n".join(f"Excerpt: {w}" for w in windows[i])
        items.append(f"Item {j}\nClaim: {sents[i]['text']}\n{ex}")
    prompt = f"Question: {question}\n\n" + "\n\n".join(items)
    jraw = gen(prompt, role="verifier", system=JUDGE_SYSTEM, json_output=True, max_output_tokens=max_tokens)
    debug["judge"] = jraw
    got = parse_verdicts(jraw, len(todo))
    says = parse_field(jraw, len(todo), "says")
    debug["says"] = {i: says[j] for j, i in enumerate(todo)}
    for j, i in enumerate(todo):
        verdicts[i] = got[j]
    return verdicts, debug


def verify(question: str, sents: list[dict], hits: list[dict], gen, mode: str = "locate_judge",
           require_quote: bool = True, max_tokens: int = 3000) -> tuple[list[str], dict]:
    """Verdict per sentence and a debug record for the trace. Raises on an LLM failure."""
    if not sents:
        return [], {}
    if mode == "single":
        return _single(question, sents, hits, gen, require_quote, max_tokens)
    if mode == "locate_judge":
        return _locate_judge(question, sents, hits, gen, max_tokens)
    raise ValueError(f"unknown verify mode {mode!r}: use 'locate_judge' or 'single'")