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
  4. Verify (verify.py, VERIFY_MODE): locate the words of a cited passage that state each sentence
     (checked in code against the real text), then judge the sentence on a short window around them:
     supported | partial | unsupported. Nothing located = unsupported.
  5. All supported (each with a quote found in its cited passages): answer. Some partial: answer with a caution. Any unsupported (or uncited): regenerate
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

ANSWER_ALGO = "answer_v4"
ABSTAIN_TEXT = "I could not find enough support in the specified playlist."
VERDICTS = ("supported", "partial", "unsupported")

GEN_SYSTEM = """You answer questions using ONLY the numbered evidence passages provided. They are speech-to-text \
transcripts of a closed set of videos ({topic}). Names may be misspelled by the transcription; judge by meaning.

Rules:
1. Use nothing from outside the passages, even if you are sure it is true.
2. Every sentence of your answer must be supported by the passage(s) you cite for it.
3. Write the answer in {lang}. The passages may be in another language: translate faithfully, add nothing.
4. Keep it short and direct: one to four sentences, each stating a different claim. Never say the same thing twice.
   Answer the question that was asked with the best-supported answer. Do not add other candidates, \
alternatives or background the question did not ask for. Give several items only when the question asks \
for a list, a comparison or several people.
5. Names: write a name the way the passage spells it (transliterated if you answer in another language). \
The transcription may garble a name, but never replace it with a different, better-known character; \
if you cannot tell who is meant, say "the one named <name as written>".
6. If the passages answer only part of the question (for example one of several people it asks about), answer \
that part and list the parts they do not cover in "not_covered". Set "sufficient" to false only when they \
answer no part of it.
7. If the question assumes something the passages do not say happened (a false premise), or the passages \
say the opposite, set "sufficient" to false. Never answer by accepting the assumption.
Reply with JSON only:
{{"sufficient": true, "sentences": [{{"text": "...", "cites": [1, 3]}}], "not_covered": []}}"""

from verify import SINGLE_SYSTEM as VERIFY_SYSTEM  # noqa: E402  (kept for callers of the old name)
from verify import parse_quotes, parse_verdicts, quote_in_passage  # noqa: E402,F401


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


def parse_not_covered(text: str) -> list[str]:
    """Parts of the question the generator says the passages do not cover. Never raises."""
    try:
        d = _json(text)
    except ValueError:
        return []
    v = d.get("not_covered") if isinstance(d, dict) else None
    return [str(x).strip() for x in v if str(x).strip()][:5] if isinstance(v, list) else []


# ---------------------------------------------------------------------------------------
def answer_question(question: str, retrieve_fn, gen, titles: dict | None = None, topic: str = "",
                    max_attempts: int = 3, strict: bool = True, trace: dict | None = None,
                    require_quote: bool = False, verify_mode: str = "single",
                    verifier_max_tokens: int = 3000) -> dict:
    """gen(prompt, role, system, json_output, max_output_tokens) -> str. Returns the result dict."""
    titles = titles or {}
    tr = trace if trace is not None else {}
    tr.update({"question": question, "attempts": []})
    ret = retrieve_fn(question)
    hits = ret["hits"]
    tr["retrieval"] = {k: v for k, v in ret.items() if k != "hits"}
    tr["evidence"] = hits
    base = {"question": question, "evidence": hits, "query_entities": ret.get("query_entities", []),
            "answer": ABSTAIN_TEXT, "sentences": [], "warnings": [], "attempts": 0, "error": False,
            "not_covered": []}
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
            raw = gen(prompt, role="answer", system=system, json_output=True, max_output_tokens=4000)
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
            import verify
            try:
                got, dbg = verify.verify(question, [sents[i] for i in todo], hits, gen, mode=verify_mode,
                                         require_quote=require_quote, max_tokens=verifier_max_tokens)
            except Exception as e:                     # fail closed
                rec["error"] = f"verifier failed: {type(e).__name__}: {str(e)[:200]}"
                return {**base, "outcome": "abstained", "error": True, "reason": rec["error"]}
            for i, v in zip(todo, got):
                verdicts[i] = v
            if "verification" in dbg:
                rec["verification"] = dbg.pop("verification")
            for f in dbg.get("quote_failures", []):
                f["i"] = todo[f["i"]]
            for key in ("windows", "says"):
                if key in dbg:
                    dbg[key] = {todo[int(k)]: v for k, v in dbg[key].items()}
            rec.update(dbg)
        base["not_covered"] = parse_not_covered(raw)
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
    if res["outcome"] != "abstained" and res.get("not_covered"):
        lines.append("Not covered by the passages found: " + "; ".join(res["not_covered"]))
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
    models = "|".join(f"{s.get('backend')}:{s.get('model')}" for s in st) + f"|rerank={rr is not None}|strict={getattr(config, 'ANSWER_STRICT', True)}|quote={getattr(config, 'ANSWER_REQUIRE_QUOTE', True)}"
    models += f"|verify={getattr(config, 'VERIFY_MODE', 'locate_judge')}|think={getattr(config, 'LLM_VERIFIER_THINKING', False)}"
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
        trace=trace, require_quote=getattr(config, "ANSWER_REQUIRE_QUOTE", True),
        verify_mode=getattr(config, "VERIFY_MODE", "locate_judge"),
        verifier_max_tokens=getattr(config, "VERIFIER_MAX_TOKENS", 3000))
    res["cached"] = False
    res["trace"] = write_trace(trace, res)
    write_cache(key, res)
    return res


def run_eval(questions_path: str, out_path: str, stack, gen, k: int, use_cache: bool = True) -> int:
    """Full-pipeline run file (eval_harness format). Rows already in out_path are kept, so a run
    stopped by the LLM quota can be resumed; error rows are redone."""
    from eval_harness import load_questions, read_jsonl, write_jsonl
    qs = load_questions(questions_path)
    done = {}
    if os.path.exists(out_path):
        done = {r["id"]: r for r in read_jsonl(out_path) if not r.get("error")}
    rows = []
    for q in qs:
        if q["id"] in done:
            rows.append(done[q["id"]])
            continue
        res = ask(q["question"], stack, gen, use_cache, k)
        rows.append({"id": q["id"],
                     "retrieved": [{"video_id": h["video_id"], "start": h["start"], "end": h["end"],
                                    "span_id": h["span_id"]} for h in res["evidence"]],
                     "answer": res["answer"] if res["outcome"] != "abstained" else "",
                     "abstained": res["outcome"] == "abstained" and not res.get("error"),
                     "citations": [{"video_id": c["video_id"], "start": c["start"], "end": c["end"],
                                    "span_id": c["span_id"]} for s in res["sentences"] for c in s["cites"]],
                     "outcome": res["outcome"], **({"error": res["reason"]} if res.get("error") else {})})
        write_jsonl(out_path, rows)                  # after every question: a crash loses nothing
    bad = [r["id"] for r in rows if r.get("error")]
    print(f"{len(rows)} questions -> {out_path}; {len(bad)} with errors" + (f": {', '.join(bad)}" if bad else ""))
    return 2 if bad else 0


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--playlist")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--trim", action="store_true",
                    help="keep the verified sentences instead of abstaining when others fail (default: abstain)")
    ap.add_argument("--verify-mode", choices=["locate_judge", "single"],
                    help="how sentences are checked (default: VERIFY_MODE in config, locate_judge)")
    ap.add_argument("--verifier-thinking", action="store_true",
                    help="let the verifier reason before answering (Qwen3 thinking mode; much slower)")
    ap.add_argument("--eval", metavar="QUESTIONS", help="answer every question in this file, write a run file")
    ap.add_argument("--out", help="with --eval: run file to write")
    a = ap.parse_args(argv)
    if a.playlist:
        config.select_playlist(a.playlist)
    if a.trim:
        config.ANSWER_STRICT = False
    if a.verify_mode:
        config.VERIFY_MODE = a.verify_mode
    if a.verifier_thinking:
        config.LLM_VERIFIER_THINKING = True
    import retrieval
    stack = retrieval.load_stack(not a.no_rerank)
    gen = make_gen()
    if a.eval:
        if not a.out:
            print("--eval needs --out", file=sys.stderr)
            return 2
        return run_eval(a.eval, a.out, stack, gen, a.k, not a.no_cache)
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