"""
Cited answers with a fail-closed verifier.

    python answer.py "Who killed Drona?" --playlist mahabharata [--json] [--no-cache] [--no-rerank] [--k 8]
    python answer.py --playlist mahabharata            # no question: asks in a loop

    result = answer_question(question, retrieve_fn, gen, titles, ...)

Flow (plain functions, explicit inputs and outputs):
  1. retrieve_fn(question) -> hits (retrieval.retrieve). No hits: abstain.
  2. Generate: the LLM sees numbered evidence [1]..[n] and returns JSON
     {"sufficient": bool, "sentences": [{"text", "cites": [n, ...]}]}. Not sufficient: abstain.
  3. Check in code: every sentence must cite at least one existing evidence number.
  4. Verify: one LLM call judges each sentence against ONLY the passages it cites:
     supported | partial | unsupported.
  5. All supported: answer. Some partial: answer with a caution. Any unsupported (or uncited): regenerate
     with that feedback, up to max_attempts in total (1 + 2 regenerations by default). Still unsupported:
     abstain (strict, the default) or, with strict=False, keep only the verified sentences plus a warning.
  6. Any generator or verifier failure: abstain and mark the result as an error (never cached).
Citation format: [Video <title>, hh:mm:ss-hh:mm:ss] after each sentence, plus an evidence table with
span_id and a timestamped link. Every question writes a trace file; finished answers are cached.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

import config

ANSWER_ALGO = "answer_v1"
ABSTAIN_TEXT = "I could not find enough support in the specified playlist."
VERDICTS = ("supported", "partial", "unsupported")

GEN_SYSTEM = """You answer questions using ONLY the numbered evidence passages provided. They are speech-to-text \
transcripts of a closed set of videos ({topic}). Names may be misspelled by the transcription; judge by meaning.

Rules:
1. Use nothing from outside the passages, even if you are sure it is true.
2. Every sentence of your answer must be supported by the passage(s) you cite for it.
3. Write the answer in {lang}. The passages may be in another language: translate faithfully, add nothing.
4. Keep it short and direct: one to four sentences, each stating one claim.
5. If the passages do not contain enough to answer the question, set "sufficient" to false.
Reply with JSON only:
{{"sufficient": true, "sentences": [{{"text": "...", "cites": [1, 3]}}]}}"""

VERIFY_SYSTEM = """You check whether statements are supported by passages. Each statement comes with the \
passages it cites (speech-to-text transcripts: names may be misspelled; the statement may be in a different \
language from the passages). Judge only against those passages, never against what you know.
supported   = the passages clearly say what the statement says.
partial     = the passages support part of it, or only imply it, or the statement adds a detail they lack.
unsupported = the passages do not say it, or contradict it.
Reply with JSON only: {"verdicts": [{"i": 0, "verdict": "supported"}, ...]} with one entry per statement."""


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------
def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def question_language(q: str) -> str:
    from romanize import script_of
    return "the same language as the question" if script_of(q) else "English"


def evidence_block(hits: list[dict], titles: dict) -> str:
    out = []
    for n, h in enumerate(hits, 1):
        out.append(f"[{n}] ({titles.get(h['video_id'], h['video_id'])}, {hms(h['start'])}-{hms(h['end'])})\n{h['text']}")
    return "\n\n".join(out)


def _json(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        from llm_client import extract_json
        return json.loads(extract_json(text))
    except Exception as e:                             # LLMError and friends: treat as unreadable
        raise ValueError(f"not JSON: {str(e)[:80]}")


def parse_generation(text: str, n_hits: int) -> tuple[bool, list[dict]]:
    """(sufficient, sentences). Raises ValueError on anything malformed."""
    d = _json(text)
    if not isinstance(d, dict) or not isinstance(d.get("sufficient"), bool):
        raise ValueError("generator JSON needs a boolean 'sufficient'")
    if not d["sufficient"]:
        return False, []
    sents = []
    for s in d.get("sentences") or []:
        t = str(s.get("text", "")).strip()
        cites = [c for c in s.get("cites", []) if isinstance(c, int) and 1 <= c <= n_hits]
        if t:
            sents.append({"text": t, "cites": sorted(set(cites))})
    if not sents:
        raise ValueError("no sentences")
    return True, sents


def parse_verdicts(text: str, n: int) -> list[str]:
    d = _json(text)
    got = {}
    for v in d.get("verdicts", []):
        if isinstance(v.get("i"), int) and v.get("verdict") in VERDICTS:
            got[v["i"]] = v["verdict"]
    if set(got) != set(range(n)):
        raise ValueError("verifier did not answer every statement")
    return [got[i] for i in range(n)]


def verify_prompt(sents: list[dict], hits: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sents):
        passages = "\n".join(f"  passage [{c}]: {hits[c - 1]['text']}" for c in s["cites"])
        parts.append(f"Statement {i}: {s['text']}\n{passages}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------------------
# the answer loop
# ---------------------------------------------------------------------------------------
def answer_question(question: str, retrieve_fn, gen, titles: dict | None = None, topic: str = "",
                    max_attempts: int = 3, strict: bool = True, trace: dict | None = None) -> dict:
    """gen(prompt, role, system, json_output, max_output_tokens) -> str. Returns the result dict."""
    titles = titles or {}
    tr = trace if trace is not None else {}
    tr.update({"question": question, "attempts": []})
    ret = retrieve_fn(question)
    hits = ret["hits"]
    tr["retrieval"] = {k: v for k, v in ret.items() if k != "hits"}
    tr["evidence"] = hits
    base = {"question": question, "evidence": hits, "query_entities": ret.get("query_entities", []),
            "answer": ABSTAIN_TEXT, "sentences": [], "warnings": [], "attempts": 0, "error": False}
    if not hits:
        return {**base, "outcome": "abstained", "reason": "nothing retrieved"}

    system = GEN_SYSTEM.format(topic=topic or "a video playlist", lang=question_language(question))
    feedback = ""
    last_bad: list[str] = []
    last = None
    for attempt in range(1, max_attempts + 1):
        prompt = f"Question: {question}\n\nEvidence:\n{evidence_block(hits, titles)}{feedback}"
        rec = {"attempt": attempt}
        tr["attempts"].append(rec)
        base["attempts"] = attempt
        try:
            raw = gen(prompt, role="answer", system=system, json_output=True, max_output_tokens=2000)
            rec["generation"] = raw
            sufficient, sents = parse_generation(raw, len(hits))
        except ValueError as e:                        # malformed output: try again
            rec["error"] = f"generation unreadable: {e}"
            feedback = "\n\nYour previous reply was not valid JSON in the required shape. Reply with JSON only."
            continue
        except Exception as e:                         # API failure: stop, never guess
            rec["error"] = f"generation failed: {type(e).__name__}: {str(e)[:200]}"
            return {**base, "outcome": "abstained", "error": True, "reason": rec["error"]}
        if not sufficient:
            return {**base, "outcome": "abstained", "reason": "the evidence does not answer the question"}

        verdicts = ["unsupported" if not s["cites"] else None for s in sents]
        todo = [i for i, v in enumerate(verdicts) if v is None]
        if todo:
            try:
                vraw = gen(verify_prompt([sents[i] for i in todo], hits), role="verifier",
                           system=VERIFY_SYSTEM, json_output=True, max_output_tokens=1000)
                rec["verification"] = vraw
                for i, v in zip(todo, parse_verdicts(vraw, len(todo))):
                    verdicts[i] = v
            except Exception as e:                     # fail closed
                rec["error"] = f"verifier failed: {type(e).__name__}: {str(e)[:200]}"
                return {**base, "outcome": "abstained", "error": True, "reason": rec["error"]}
        rec["verdicts"] = verdicts
        bad = [i for i, v in enumerate(verdicts) if v == "unsupported"]
        if not bad:
            return _finish(base, sents, verdicts, hits, titles, attempt)
        last_bad = [sents[i]["text"] for i in bad]
        feedback = ("\n\nA checker found these statements of your previous answer NOT supported by the passages "
                    "they cited:\n" + "\n".join(f"- {t}" for t in last_bad)
                    + "\nWrite the answer again. Drop or rephrase them so every sentence is supported by its citations.")
        last = (sents, verdicts)
    if not strict and last is not None and last_bad:
        sents, verdicts = last
        keep = [i for i, v in enumerate(verdicts) if v != "unsupported"]
        if keep:
            res = _finish(base, [sents[i] for i in keep], [verdicts[i] for i in keep], hits, titles, max_attempts)
            res["warnings"].append("Some statements could not be verified and were removed.")
            return res
    reason = "statements could not be verified against the evidence" if last_bad else "no readable answer produced"
    return {**base, "outcome": "abstained", "reason": reason}


def _cite(n: int, hits: list[dict], titles: dict) -> dict:
    h = hits[n - 1]
    return {"n": n, "span_id": h["span_id"], "video_id": h["video_id"],
            "title": titles.get(h["video_id"], h["video_id"]), "start": h["start"], "end": h["end"],
            "url": h["url"]}


def _finish(base: dict, sents: list[dict], verdicts: list[str], hits: list[dict], titles: dict, attempt: int) -> dict:
    out_s, lines = [], []
    for s, v in zip(sents, verdicts):
        cites = [_cite(c, hits, titles) for c in s["cites"]]
        out_s.append({"text": s["text"], "verdict": v, "cites": cites})
        tag = " ".join(f"[Video {c['title']}, {hms(c['start'])}–{hms(c['end'])}]" for c in cites)
        lines.append(f"{s['text']} {tag}")
    partial = any(v == "partial" for v in verdicts)
    warnings = ["Only partly supported by the cited passages; check the excerpts before relying on it."] if partial else []
    return {**base, "outcome": "answered_with_caution" if partial else "answered", "reason": "",
            "answer": " ".join(lines), "sentences": out_s, "warnings": warnings, "attempts": attempt}


# ---------------------------------------------------------------------------------------
# cache and trace
# ---------------------------------------------------------------------------------------
def cache_key(question: str, playlist_id: str, k: int, models: str, corpus_stamp: str) -> str:
    q = " ".join(question.lower().split())
    blob = json.dumps([ANSWER_ALGO, q, playlist_id, k, models, corpus_stamp], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _dir(name: str, sub: str) -> str:
    return getattr(config, name, os.path.join("data", sub))


def read_cache(key: str) -> dict | None:
    p = os.path.join(_dir("ANSWER_CACHE_DIR", "answers"), f"{key}.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_cache(key: str, result: dict) -> None:
    if result.get("error"):
        return                                         # never cache a failure
    from manifest import atomic_write_json
    d = _dir("ANSWER_CACHE_DIR", "answers")
    os.makedirs(d, exist_ok=True)
    atomic_write_json(os.path.join(d, f"{key}.json"), result)


def write_trace(trace: dict, result: dict) -> str:
    from manifest import atomic_write_json
    d = _dir("TRACE_DIR", "traces")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha256(trace.get("question", "").encode("utf-8")).hexdigest()[:8]
    p = os.path.join(d, f"{stamp}_{h}.json")
    atomic_write_json(p, {**trace, "result": {k: v for k, v in result.items() if k != "evidence"}})
    return p


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def video_titles() -> dict:
    try:
        with open(config.PLAYLIST_META_PATH, encoding="utf-8") as f:
            return {v["id"]: (v.get("title") or v["id"]) for v in json.load(f)["videos"]}
    except (OSError, ValueError, KeyError):
        return {}


def make_gen():
    import llm_client

    def gen(prompt, role="answer", system=None, json_output=True, max_output_tokens=1000):
        return llm_client.generate_default(prompt, role=role, system_instruction=system,
                                           json_output=json_output, max_output_tokens=max_output_tokens)
    return gen


def render(res: dict) -> str:
    lines = [res["answer"]]
    for w in res["warnings"]:
        lines.append(f"CAUTION: {w}")
    if res["outcome"] == "abstained" and res.get("reason"):
        lines.append(f"({res['reason']})")
    if res["outcome"] != "abstained":
        lines.append("\nEvidence")
        seen = set()
        for s in res["sentences"]:
            for c in s["cites"]:
                if c["span_id"] not in seen:
                    seen.add(c["span_id"])
                    lines.append(f"  {c['span_id']}  {hms(c['start'])}–{hms(c['end'])}  {c['title']}  {c['url']}")
    return "\n".join(lines)


def ask(question: str, stack, gen, use_cache: bool, k: int) -> dict:
    import index_store as ix
    import llm_client
    import retrieval
    client, emb, rr, gl = stack
    st = [llm_client.settings("answer"), llm_client.settings("verifier")]
    models = "|".join(f"{s.get('backend')}:{s.get('model')}" for s in st) + f"|rerank={rr is not None}"
    key = cache_key(question, config.PLAYLIST_ID, k, models, str(ix.count(client, config.PLAYLIST_ID)))
    if use_cache:
        hit = read_cache(key)
        if hit:
            hit["cached"] = True
            return hit
    trace: dict = {}
    from run_pipeline import playlist_topic
    res = answer_question(
        question, lambda q: retrieval.retrieve(q, client, emb, rr, gl, config.PLAYLIST_ID, k=k), gen,
        titles=video_titles(), topic=playlist_topic(),
        max_attempts=getattr(config, "ANSWER_MAX_ATTEMPTS", 3), strict=getattr(config, "ANSWER_STRICT", True),
        trace=trace)
    res["cached"] = False
    res["trace"] = write_trace(trace, res)
    write_cache(key, res)
    return res


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--playlist")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args(argv)
    if a.playlist:
        config.select_playlist(a.playlist)
    import retrieval
    stack = retrieval.load_stack(not a.no_rerank)
    gen = make_gen()
    rc = 0
    qs = [a.question] if a.question else None
    while True:
        if qs is None:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit"):
                break
            if not q:
                continue
        else:
            if not qs:
                break
            q = qs.pop(0)
        res = ask(q, stack, gen, not a.no_cache, a.k)
        print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else render(res))
        if res.get("error"):
            rc = 2
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))