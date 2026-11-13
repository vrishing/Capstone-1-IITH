"""
Gold evidence from a pool of candidates: draft questions -> labelled pool -> eval question file.

    python gold_pool.py build   --draft eval/draft_mahabharata.jsonl --playlist mahabharata \
                                --out eval/pool_mahabharata.csv
    python gold_pool.py prelabel --pool eval/pool_mahabharata.csv [--max-calls 15]    # optional, LLM
    python gold_pool.py apply   --draft eval/draft_mahabharata.jsonl --pool eval/pool_mahabharata.csv \
                                --out eval/mahabharata.questions.jsonl [--accept-llm]

Why a pool: the right evidence for a question is spread over a playlist and no one has time to
read 60 hours. Candidates come from several retrievers at once (hybrid, dense only, sparse only,
and a wider hybrid list for questions that need several videos), so a span only one of them finds
still gets judged, and no single retriever sets its own gold (the usual bias in such sets).

Labelling (the CSV, one row per candidate span): write y in the `label` column when the passage
contains something needed for the answer. Leave it blank otherwise. The `llm` column, when you
ran prelabel, is a suggestion only: it counts only with --accept-llm, and a human label always wins.

apply merges neighbouring y spans of a video into one time range, writes the eval question file
(eval_harness format), and reports what needs a decision:
  - an answerable question with no y span: fix the question, or mark it unanswerable on purpose
  - a question drafted as unanswerable that got y spans: the corpus does cover it
  - a multi-video question (hops >= 2) whose gold sits in fewer videos than hops
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

POOL_COLS = ["qid", "question", "rank", "found_by", "span_id", "video_id", "start", "end", "link",
             "text", "llm", "label", "note"]
MERGE_GAP_S = 5.0


def read_draft(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------
def pool_for(question: dict, retrieve_fn) -> list[dict]:
    """retrieve_fn(text, mode, k) -> hits. Unique spans in first-seen order, with who found them."""
    wide = int(question.get("hops", 1) or 1) >= 2
    plan = [("hybrid", 25 if wide else 12), ("dense", 8), ("sparse", 8)]
    pooled: dict[str, dict] = {}
    for mode, k in plan:
        for rank, h in enumerate(retrieve_fn(question["question"], mode, k), 1):
            e = pooled.setdefault(h["span_id"], {**h, "rank": rank, "found_by": []})
            e["rank"] = min(e["rank"], rank)
            e["found_by"].append(mode)
    return list(pooled.values())


def build(draft: list[dict], retrieve_fn, out_csv: str) -> int:
    n = 0
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLS)
        w.writeheader()
        for q in draft:
            # unanswerable drafts are pooled too: the corpus might cover them after all
            for h in sorted(pool_for(q, retrieve_fn), key=lambda h: (h["video_id"], h["start"])):
                w.writerow({"qid": q["id"], "question": q["question"], "rank": h["rank"],
                            "found_by": "+".join(sorted(set(h["found_by"]))), "span_id": h["span_id"],
                            "video_id": h["video_id"], "start": h["start"], "end": h["end"],
                            "link": h.get("url", ""), "text": h["text"].replace("\n", " ")[:700],
                            "llm": "", "label": "", "note": ""})
                n += 1
    return n


# ---------------------------------------------------------------------------------------
# prelabel (optional, LLM)
# ---------------------------------------------------------------------------------------
PRELABEL_SYSTEM = """You help build an answer key. Given a question and numbered passages (speech-to-text \
transcripts of lectures; names may be misspelled; the question may be in another language), list every passage \
that contains a fact needed for the answer. Be inclusive: a passage that gives only PART of the answer counts \
(one step of an explanation, one of several items, the definition or explanation of a term the question asks \
about, one side of a comparison). Leave out a passage that only mentions the same person or topic without \
saying something that helps answer. Judge only by the passages. Reply with JSON only: {"relevant": [1, 4]} \
(an empty list if none)."""
PRELABEL_VERSION = "v2"
PRELABEL_CHUNK = 10          # passages per call: small models lose track in a list of 25


def _parse(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError:
        from llm_client import extract_json            # fenced or chatty replies
        return json.loads(extract_json(text))


def read_pool(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_pool(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def prelabel(pool_path: str, gen, max_calls: int = 15, cache_path: str | None = None) -> dict:
    """One call per question. Answers are cached (by question and candidate set), so a run cut short
    by a quota continues where it stopped. Returns {"asked", "cached", "left"}."""
    rows = read_pool(pool_path)
    cache_path = cache_path or pool_path + ".llm.json"
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    asked = cached = 0
    labelled: set[str] = set()
    for qid, rs in by_q.items():
        key = hashlib.sha256((PRELABEL_VERSION + str(PRELABEL_CHUNK) + qid + "|" + rs[0]["question"] + "|" + ",".join(r["span_id"] for r in rs)).encode()).hexdigest()[:20]
        if key in cache:
            rel, cached = cache[key], cached + 1
        else:
            if asked >= max_calls:
                continue
            raw, rel = "", []
            try:
                for a in range(0, len(rs), PRELABEL_CHUNK):
                    chunk = rs[a:a + PRELABEL_CHUNK]
                    prompt = f"Question: {rs[0]['question']}\n\nPassages:\n" + "\n\n".join(
                        f"[{i}] {r['text'][:600]}" for i, r in enumerate(chunk, 1))
                    raw = gen(prompt, role="verifier", system=PRELABEL_SYSTEM, json_output=True,
                              max_output_tokens=4000)
                    d = _parse(raw)
                    if isinstance(d, list):              # a bare list means the same as {"relevant": [...]}
                        d = {"relevant": d}
                    rel += [a + i for i in d.get("relevant", []) if isinstance(i, int) and 1 <= i <= len(chunk)]
                rel = sorted(set(rel))
            except Exception as e:                    # quota or bad JSON: keep what we have, stop here
                print(f"prelabel stopped at {qid}: {type(e).__name__}: {str(e)[:120]}\n  raw reply: {raw[:300]!r}",
                      file=sys.stderr)
                break
            cache[key] = rel
            asked += 1
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        labelled.add(qid)
        for i, r in enumerate(rs, 1):
            r["llm"] = "y" if i in rel else "n"
    write_pool(pool_path, rows)
    return {"asked": asked, "cached": cached, "left": len(by_q) - len(labelled)}


# ---------------------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------------------
def merge_ranges(spans: list[tuple[str, float, float]], gap: float = MERGE_GAP_S) -> list[dict]:
    out: list[dict] = []
    for vid, s, e in sorted(spans):
        if out and out[-1]["video_id"] == vid and s - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], e)
        else:
            out.append({"video_id": vid, "start": s, "end": e})
    return [{"video_id": g["video_id"], "start": round(g["start"], 2), "end": round(g["end"], 2)} for g in out]


def apply(draft: list[dict], pool: list[dict], accept_llm: bool = False) -> tuple[list[dict], dict]:
    marked: dict[str, list] = {}
    for r in pool:
        lab = (r.get("label") or "").strip().lower()
        if not lab and accept_llm:
            lab = (r.get("llm") or "").strip().lower()
        if lab == "y":
            marked.setdefault(r["qid"], []).append((r["video_id"], float(r["start"]), float(r["end"])))
    out, issues = [], {"no_gold": [], "unanswerable_has_evidence": [], "too_few_videos": []}
    for q in draft:
        gold = merge_ranges(marked.get(q["id"], []))
        row = {"id": q["id"], "question": q["question"], "language": q.get("language", ""),
               "category": q.get("category", "other"), "answerable": bool(q.get("answerable", True)),
               "gold": [], "must_include": q.get("must_include", []),
               "difficulty": q.get("difficulty"), "hops": q.get("hops")}
        if row["answerable"]:
            if not gold:
                issues["no_gold"].append(q["id"])
                continue                               # not written: the harness needs gold for these
            row["gold"] = gold
            if len({g["video_id"] for g in gold}) < int(q.get("hops", 1) or 1):
                issues["too_few_videos"].append(q["id"])
        elif gold:
            issues["unanswerable_has_evidence"].append(q["id"])
        out.append(row)
    return out, issues


def write_questions(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--draft", required=True)
    b.add_argument("--playlist", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--no-rerank", action="store_true")
    p = sub.add_parser("prelabel")
    p.add_argument("--pool", required=True)
    p.add_argument("--max-calls", type=int, default=15)
    a = sub.add_parser("apply")
    a.add_argument("--draft", required=True)
    a.add_argument("--pool", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--accept-llm", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        import config
        import retrieval
        config.select_playlist(args.playlist)
        client, emb, rr, gl = retrieval.load_stack(not args.no_rerank)

        def fn(text, mode, k):
            return retrieval.retrieve(text, client, emb, rr, gl, config.PLAYLIST_ID, k=k, mode=mode)["hits"]
        n = build(read_draft(args.draft), fn, args.out)
        print(f"{n} candidate rows -> {args.out}\nOpen it, write y in the label column for every passage "
              f"that contains part of the answer, then run: gold_pool.py apply")
        return 0
    if args.cmd == "prelabel":
        import answer
        r = prelabel(args.pool, answer.make_gen(), args.max_calls)
        print(f"asked {r['asked']}, from cache {r['cached']}, still to do {r['left']}")
        return 0
    draft = read_draft(args.draft)
    rows, issues = apply(draft, read_pool(args.pool), args.accept_llm)
    write_questions(args.out, rows)
    print(f"{len(rows)} of {len(draft)} questions written to {args.out}")
    for k, v in issues.items():
        if v:
            print(f"  {k}: {', '.join(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))