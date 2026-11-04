"""
Evaluation harness (stdlib only; uses romanize.py when present).

Commands (python eval_harness.py <command> --help):
  init      write eval/questions.jsonl and eval/cer_samples.jsonl templates
  suggest   find candidate spans for a question, to mark gold evidence faster
  validate  check the question file; report counts and leakage against the glossary sample
  score     retrieval + answer metrics for one run file, with 95% bootstrap intervals
  compare   the same metrics for several run files side by side (ablations)
  cer       character / word / akshara error rate before and after correction
  glossary  how much of the curated glossary an automatic glossary recovers

Question file (JSON lines). Gold evidence is hand-marked time ranges, never span IDs, so it
survives any re-chunking:
  {"id": "q01", "question": "...", "language": "te", "category": "factoid",
   "answerable": true, "gold": [{"video_id": "abcdefghijk", "start": "12:30", "end": "13:45"}],
   "must_include": ["కర్ణుడు"]}
  Unanswerable questions have "answerable": false and no gold. Times: seconds, "mm:ss" or "hh:mm:ss".

Run file (JSON lines, one per question; written by retrieval/generation code, or collect_run()):
  {"id": "q01", "retrieved": [{"video_id": "...", "start": 750, "end": 820, "span_id": "..."}],
   "answer": "...", "abstained": false, "citations": [{"video_id": "...", "start": 750, "end": 820}]}
  "retrieved" is in rank order. A question with no run row counts as a failure.

A retrieved or cited span HITS a gold range when it is in the same video and overlaps it by
min(--min-overlap seconds, half the gold range). Metrics:
  recall@k           answerable questions with a hit in the top k
  mrr                mean 1/rank of the first hit
  timestamp_hit@1    the top-ranked span overlaps the gold range
  evidence_recall@k  fraction of a question's gold ranges hit in the top k (multi-hop)
  citation_hit       answered, and at least one citation hits gold
  citation_precision share of cited spans that hit gold
  end_to_end         citation_hit and every must_include term present in the answer
  abstain_rate       unanswerable questions correctly abstained on
  false_abstain      answerable questions wrongly abstained on
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    from romanize import phonetic_key as _pk
except Exception:  # pragma: no cover - the harness still works without it
    def _pk(s: str) -> str:
        return ""

DEFAULT_KS = (1, 3, 5, 10)
EVAL_DIR = Path("eval")


class EvalError(Exception):
    """Shown to the user as a plain message."""


# ======================================================================================
# Reading files
# ======================================================================================
def parse_time(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise EvalError(f"Cannot read time {v!r}; use seconds, mm:ss or hh:mm:ss.") from None
    if not 1 <= len(nums) <= 3:
        raise EvalError(f"Cannot read time {v!r}; use seconds, mm:ss or hh:mm:ss.")
    t = 0.0
    for n in nums:
        t = t * 60 + n
    return t


def fmt_time(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        raise EvalError(f"File not found: {p}")
    for n, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise EvalError(f"{p} line {n} is not valid JSON: {e}") from None
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, p)


def read_ids(path: str | Path) -> set[str]:
    """Video IDs from a file with one ID or URL per line (# comments allowed)."""
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", line)
        ids.add(m.group(1) if m else line)
    return ids


def load_questions(path: str | Path) -> list[dict]:
    rows = [r for r in read_jsonl(path) if not r.get("example")]
    seen, out = set(), []
    for i, r in enumerate(rows, 1):
        qid = str(r.get("id", "")).strip()
        where = f"question {qid or '#' + str(i)}"
        if not qid:
            raise EvalError(f"{where}: missing id.")
        if qid in seen:
            raise EvalError(f"{where}: duplicate id.")
        seen.add(qid)
        if not str(r.get("question", "")).strip():
            raise EvalError(f"{where}: empty question text.")
        answerable = r.get("answerable", True)
        if not isinstance(answerable, bool):
            raise EvalError(f"{where}: answerable must be true or false.")
        gold = []
        for g in r.get("gold", []) or []:
            vid = str(g.get("video_id", "")).strip()
            if not vid:
                raise EvalError(f"{where}: a gold range has no video_id.")
            s, e = parse_time(g.get("start")), parse_time(g.get("end"))
            if e <= s:
                raise EvalError(f"{where}: gold range {g} ends before it starts.")
            gold.append({"video_id": vid, "start": s, "end": e})
        if answerable and not gold:
            raise EvalError(f"{where}: answerable questions need at least one gold range.")
        if not answerable and gold:
            raise EvalError(f"{where}: unanswerable questions must not have gold ranges.")
        out.append({"id": qid, "question": r["question"], "language": r.get("language", ""),
                    "category": r.get("category") or ("unanswerable" if not answerable else "other"),
                    "answerable": answerable, "gold": gold,
                    "must_include": [str(t) for t in r.get("must_include", []) or []]})
    return out


def drop_leaked(questions: list[dict], sample_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Questions whose gold evidence is in a glossary-sample video are excluded (leakage)."""
    keep, dropped = [], []
    for q in questions:
        if any(g["video_id"] in sample_ids for g in q["gold"]):
            dropped.append(q["id"])
        else:
            keep.append(q)
    return keep, dropped


def load_runs(path: str | Path) -> dict[str, dict]:
    runs = {}
    for r in read_jsonl(path):
        if "id" not in r:
            raise EvalError(f"{path}: a run row has no id.")
        runs[str(r["id"])] = r
    return runs


def collect_run(questions: list[dict], answer_fn, out_path: str | Path) -> list[dict]:
    """Run the system on every question and write a run file.
    answer_fn(question_dict) -> {"retrieved": [...], "answer": str, "abstained": bool, "citations": [...]}
    An exception is recorded as an error row so one bad question never stops the run."""
    rows = []
    for q in questions:
        try:
            res = dict(answer_fn(q))
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            res = {"retrieved": [], "answer": "", "abstained": False, "citations": [],
                   "error": f"{type(e).__name__}: {e}"}
        res["id"] = q["id"]
        rows.append(res)
    write_jsonl(out_path, rows)
    return rows


# ======================================================================================
# Retrieval and answer metrics
# ======================================================================================
def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = "".join(c for c in s if c not in "​‌‍﻿")
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s.lower())
    return " ".join(s.split())


def _hit(span: dict, gold: dict, min_overlap_s: float) -> bool:
    if str(span.get("video_id", "")) != gold["video_id"]:
        return False
    try:
        s, e = parse_time(span.get("start")), parse_time(span.get("end"))
    except EvalError:
        return False
    ov = min(e, gold["end"]) - max(s, gold["start"])
    return ov > 0 and ov >= min(min_overlap_s, 0.5 * (gold["end"] - gold["start"]))


def _has_terms(answer: str, terms: list[str]) -> bool:
    if not terms:
        return True
    na, pa = norm_text(answer), _pk(answer or "")
    for t in terms:
        nt = norm_text(t)
        if nt and nt in na:
            continue
        pt = _pk(t)
        if pt and pt in pa:
            continue
        return False
    return True


def boot_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _mean(v: list[float]) -> float | None:
    return sum(v) / len(v) if v else None


def score(questions: list[dict], runs: dict[str, dict], ks=DEFAULT_KS, min_overlap_s: float = 10.0,
          primary_k: int = 5) -> dict:
    ks = sorted(set(ks) | {primary_k})
    per = []                                   # one record per question
    missing, errors = [], []
    for q in questions:
        run = runs.get(q["id"])
        if run is None:
            missing.append(q["id"])
        elif run.get("error"):
            errors.append(q["id"])
        run = run or {}
        retrieved = run.get("retrieved") or []
        abstained = bool(run.get("abstained", False)) if run else False
        rec = {"id": q["id"], "category": q["category"], "answerable": q["answerable"],
               "abstained": abstained, "has_run": bool(run)}
        if q["answerable"]:
            rank_of_gold = []
            for g in q["gold"]:
                r = next((i for i, s in enumerate(retrieved, 1) if _hit(s, g, min_overlap_s)), None)
                rank_of_gold.append(r)
            hits = [r for r in rank_of_gold if r is not None]
            first = min(hits) if hits else None
            for k in ks:
                rec[f"recall@{k}"] = 1.0 if first is not None and first <= k else 0.0
                rec[f"evidence_recall@{k}"] = sum(1 for r in rank_of_gold if r is not None and r <= k) / len(q["gold"])
            rec["mrr"] = 1.0 / first if first else 0.0
            rec["timestamp_hit@1"] = 1.0 if first == 1 else 0.0
            cites = run.get("citations") or []
            cite_hits = [any(_hit(c, g, min_overlap_s) for g in q["gold"]) for c in cites]
            rec["citation_hit"] = 1.0 if (not abstained and any(cite_hits)) else 0.0
            rec["citation_precision"] = (sum(cite_hits) / len(cite_hits)) if (cites and not abstained) else None
            rec["end_to_end"] = 1.0 if (rec["citation_hit"] and _has_terms(run.get("answer", ""), q["must_include"])) else 0.0
            rec["false_abstain"] = 1.0 if abstained else 0.0
        else:
            rec["abstain_rate"] = 1.0 if abstained else 0.0
        per.append(rec)

    A = [r for r in per if r["answerable"]]
    U = [r for r in per if not r["answerable"]]
    metrics, ci = {}, {}

    def put(name, rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        metrics[name] = _mean(vals)
        ci[name] = boot_ci(vals)

    for k in ks:
        put(f"recall@{k}", A, f"recall@{k}")
    put("mrr", A, "mrr")
    put("timestamp_hit@1", A, "timestamp_hit@1")
    put(f"evidence_recall@{primary_k}", A, f"evidence_recall@{primary_k}")
    put("citation_hit", A, "citation_hit")
    put("citation_precision", A, "citation_precision")
    put("end_to_end", A, "end_to_end")
    put("false_abstain", A, "false_abstain")
    put("abstain_rate", U, "abstain_rate")

    by_cat = {}
    for cat in sorted({r["category"] for r in per}):
        rows = [r for r in per if r["category"] == cat]
        ans = [r for r in rows if r["answerable"]]
        by_cat[cat] = {"n": len(rows),
                       f"recall@{primary_k}": _mean([r[f"recall@{primary_k}"] for r in ans]),
                       "end_to_end": _mean([r["end_to_end"] for r in ans]),
                       "abstain_rate": _mean([r["abstain_rate"] for r in rows if not r["answerable"]])}
    return {"n_questions": len(per), "n_answerable": len(A), "n_unanswerable": len(U),
            "primary_k": primary_k, "min_overlap_s": min_overlap_s, "metrics": metrics, "ci95": ci,
            "by_category": by_cat, "missing_runs": missing, "errored": errors,
            "failed_ids": [r["id"] for r in A if r[f"recall@{primary_k}"] == 0.0]}


def _pct(v) -> str:
    return "  n/a " if v is None else f"{100 * v:5.1f}%"


def format_score(rep: dict, title: str = "") -> str:
    pk = rep["primary_k"]
    lines = [f"{title}questions: {rep['n_questions']} ({rep['n_answerable']} answerable, "
             f"{rep['n_unanswerable']} unanswerable); hit = overlap >= min({rep['min_overlap_s']:g} s, half the gold range)"]
    for name, v in rep["metrics"].items():
        ci = rep["ci95"].get(name)
        scale, dec = (1, 3) if name == "mrr" else (100, 1)
        extra = f"   95% CI {scale * ci[0]:.{dec}f}-{scale * ci[1]:.{dec}f}" if ci else ""
        val = f"{v:6.3f}" if name == "mrr" and v is not None else _pct(v)
        lines.append(f"  {name:<22}{val}{extra}")
    lines.append(f"  by category (recall@{pk} / end_to_end / abstain):")
    for cat, c in rep["by_category"].items():
        lines.append(f"    {cat:<16} n={c['n']:<3} {_pct(c[f'recall@{pk}'])} {_pct(c['end_to_end'])} {_pct(c['abstain_rate'])}")
    if rep["missing_runs"]:
        lines.append(f"  WARNING: no run row for {len(rep['missing_runs'])} question(s): {', '.join(rep['missing_runs'][:10])}")
    if rep["errored"]:
        lines.append(f"  WARNING: {len(rep['errored'])} question(s) errored: {', '.join(rep['errored'][:10])}")
    if rep["failed_ids"]:
        lines.append(f"  missed at k={pk}: {', '.join(rep['failed_ids'])}")
    if rep["n_answerable"] < 30:
        lines.append("  note: with fewer than 30 answerable questions the intervals are wide; read them before comparing runs.")
    return "\n".join(lines)


def format_compare(named: dict[str, dict]) -> str:
    names = list(named)
    first = named[names[0]]
    keys = list(first["metrics"])
    width = max(12, *(len(n) for n in names)) + 2
    lines = ["metric".ljust(22) + "".join(n.rjust(width) for n in names)]
    for k in keys:
        cells = []
        for n in names:
            v = named[n]["metrics"].get(k)
            cells.append((f"{v:.3f}" if k == "mrr" and v is not None else _pct(v).strip()).rjust(width))
        lines.append(k.ljust(22) + "".join(cells))
    lines.append("")
    lines.append("Differences smaller than the 95% intervals from `score` are not evidence of a real change.")
    return "\n".join(lines)


# ======================================================================================
# Error rates (hand-corrected reference vs raw ASR vs corrected)
# ======================================================================================
def aksharas(s: str) -> list[str]:
    """Orthographic syllable units: a base letter with its vowel signs, and conjuncts joined by virama."""
    units: list[str] = []
    prev = ""
    for ch in s:
        attach = bool(units) and ch != " " and (unicodedata.category(ch)[0] == "M"
                                                or (prev and unicodedata.combining(prev) == 9))
        if attach:
            units[-1] += ch
        else:
            units.append(ch)
        prev = ch
    return units


def edit_distance(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(reference: str, hypothesis: str) -> dict:
    ref, hyp = norm_text(reference), norm_text(hypothesis)
    if not ref:
        raise EvalError("A reference text is empty after normalising.")
    out = {}
    for name, ru, hu in (("cer", list(ref), list(hyp)), ("wer", ref.split(), hyp.split()),
                         ("aer", aksharas(ref), aksharas(hyp))):
        d = edit_distance(ru, hu)
        out[name] = d / len(ru)
        out[name + "_edits"], out[name + "_len"] = d, len(ru)
    return out


def cer_report(samples: list[dict]) -> dict:
    rows = []
    for s in samples:
        for k in ("id", "reference", "raw", "corrected"):
            if not str(s.get(k, "")).strip():
                raise EvalError(f"CER sample {s.get('id', '?')}: missing {k!r}.")
        raw, cor = error_rates(s["reference"], s["raw"]), error_rates(s["reference"], s["corrected"])
        rows.append({"id": s["id"], "raw": raw, "corrected": cor})
    agg = {}
    for m in ("cer", "wer", "aer"):
        e_raw = sum(r["raw"][m + "_edits"] for r in rows)
        e_cor = sum(r["corrected"][m + "_edits"] for r in rows)
        n = sum(r["raw"][m + "_len"] for r in rows)
        agg[m] = {"raw": e_raw / n, "corrected": e_cor / n, "relative_change": (e_cor - e_raw) / e_raw if e_raw else 0.0}
    worse = [r["id"] for r in rows if r["corrected"]["cer"] > r["raw"]["cer"]]
    return {"n_samples": len(rows), "micro": agg, "per_sample": rows, "worse_after_correction": worse}


def format_cer(rep: dict) -> str:
    lines = [f"{rep['n_samples']} sample(s), micro-averaged (total edits / total reference length)",
             "  metric   raw      corrected   change"]
    for m, label in (("cer", "CER"), ("wer", "WER"), ("aer", "akshara")):
        a = rep["micro"][m]
        lines.append(f"  {label:<8}{100 * a['raw']:5.1f}%   {100 * a['corrected']:5.1f}%     {100 * a['relative_change']:+.1f}% relative")
    for r in rep["per_sample"]:
        lines.append(f"  {r['id']:<10} CER {100 * r['raw']['cer']:5.1f}% -> {100 * r['corrected']['cer']:5.1f}%")
    if rep["worse_after_correction"]:
        lines.append(f"  WARNING: correction made these worse: {', '.join(rep['worse_after_correction'])}")
    if rep["n_samples"] < 5:
        lines.append("  note: only a few short samples; treat the change as indicative.")
    return "\n".join(lines)


# ======================================================================================
# Glossary recovery
# ======================================================================================
def _read_alias_csv(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise EvalError(f"Alias table not found: {p}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"term", "entity_id"} <= set(rows[0]):
        raise EvalError(f"{p} has no term / entity_id columns.")
    return rows


def glossary_recovery(auto_path: str | Path, curated_path: str | Path, show: int = 15) -> dict:
    auto, cur = _read_alias_csv(auto_path), _read_alias_csv(curated_path)
    nfc = lambda s: unicodedata.normalize("NFC", s.strip())
    a_terms, c_terms = {nfc(r["term"]) for r in auto}, {nfc(r["term"]) for r in cur}
    a_ents, c_ents = {r["entity_id"] for r in auto}, {r["entity_id"] for r in cur}
    missing = sorted(c_terms - a_terms)
    return {"curated_rows": len(cur), "auto_rows": len(auto),
            "term_recall": len(c_terms & a_terms) / len(c_terms) if c_terms else None,
            "entity_recall": len(c_ents & a_ents) / len(c_ents) if c_ents else None,
            "auto_only_rows": len(a_terms - c_terms),
            "missing_terms_sample": missing[:show]}


# ======================================================================================
# Helping hand-marking gold ranges
# ======================================================================================
def _load_spans(path: Path) -> list[dict]:
    if path.name.endswith("_asr.json"):
        from spans import spans_from_asr_file
        return spans_from_asr_file(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["spans"] if isinstance(data, dict) and "spans" in data else []


def suggest(query: str, files: list[Path], limit: int = 10) -> list[dict]:
    """Spans containing every query word (as written, or by phonetic key). Not a retriever:
    a plain lookup to help you find the passage and mark its time range."""
    words = [w for w in norm_text(query).split() if w]
    if not words:
        raise EvalError("Empty query.")
    wk = [_pk(w) for w in words]
    out = []
    for f in files:
        for sp in _load_spans(f):
            text = norm_text(sp.get("text", ""))
            keyed = _pk(sp.get("text", ""))
            n = sum(1 for w, k in zip(words, wk) if w in text or (k and k in keyed))
            if n == len(words):
                pos = min((text.find(w) for w in words if w in text), default=0)
                out.append({"video_id": sp["video_id"], "start": sp["start"], "end": sp["end"],
                            "when": f"{fmt_time(sp['start'])}-{fmt_time(sp['end'])}",
                            "link": f"https://youtu.be/{sp['video_id']}?t={int(sp['start'])}",
                            "snippet": text[max(0, pos - 40):pos + 80]})
    return out[:limit]


# ======================================================================================
# Templates and command line
# ======================================================================================
def write_templates(directory: Path, force: bool = False) -> list[Path]:
    q = [{"example": True, "id": "q01", "question": "Who taught Arjuna archery?", "language": "en",
          "category": "factoid", "answerable": True,
          "gold": [{"video_id": "abcdefghijk", "start": "12:30", "end": "13:45"}], "must_include": ["Drona"]},
         {"example": True, "id": "q02", "question": "What is the capital of Brazil?", "language": "en",
          "category": "unanswerable", "answerable": False}]
    c = [{"example": True, "id": "s1", "video_id": "abcdefghijk", "start": "05:00", "end": "07:00",
          "reference": "hand-corrected text of this 2-minute stretch", "raw": "raw ASR text", "corrected": "corrected text"}]
    written = []
    for name, rows in (("questions.jsonl", q), ("cer_samples.jsonl", c)):
        p = directory / name
        if p.exists() and not force:
            continue
        write_jsonl(p, rows)
        written.append(p)
    return written


def _ks(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _load_for_scoring(args):
    qs = load_questions(args.questions)
    note = ""
    if args.exclude_ids:
        qs, dropped = drop_leaked(qs, read_ids(args.exclude_ids))
        if dropped:
            note = f"excluded {len(dropped)} question(s) whose evidence is in the glossary sample: {', '.join(dropped)}\n"
    if not qs:
        raise EvalError("No questions left to score.")
    return qs, note


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation harness.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--dir", default=str(EVAL_DIR)); p.add_argument("--force", action="store_true")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--questions", default=str(EVAL_DIR / "questions.jsonl"))
    common.add_argument("--exclude-ids", help="glossary sample file; questions with gold in these videos are dropped")
    common.add_argument("--k", default=",".join(map(str, DEFAULT_KS)), help="comma-separated cutoffs")
    common.add_argument("--primary-k", type=int, default=5)
    common.add_argument("--min-overlap", type=float, default=10.0, help="seconds of overlap that count as a hit")
    p = sub.add_parser("validate", parents=[common])
    p = sub.add_parser("score", parents=[common]); p.add_argument("--run", required=True); p.add_argument("--out")
    p = sub.add_parser("compare", parents=[common]); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out")
    p = sub.add_parser("cer"); p.add_argument("--samples", default=str(EVAL_DIR / "cer_samples.jsonl")); p.add_argument("--out")
    p = sub.add_parser("glossary"); p.add_argument("--auto", required=True); p.add_argument("--curated", required=True)
    p = sub.add_parser("suggest"); p.add_argument("query"); p.add_argument("--dir", default="data/asr_output")
    p.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "init":
            made = write_templates(Path(args.dir), args.force)
            print("wrote: " + ", ".join(map(str, made)) if made else "templates already exist (use --force)")
        elif args.cmd == "validate":
            qs, note = _load_for_scoring(args)
            cats = defaultdict(int)
            for q in qs:
                cats[q["category"]] += 1
            print(f"{note}{len(qs)} valid question(s): {sum(q['answerable'] for q in qs)} answerable, "
                  f"{sum(not q['answerable'] for q in qs)} unanswerable")
            print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(cats.items())))
        elif args.cmd == "score":
            qs, note = _load_for_scoring(args)
            rep = score(qs, load_runs(args.run), _ks(args.k), args.min_overlap, args.primary_k)
            print(note + format_score(rep, title=f"{args.run}: "))
            if args.out:
                Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.cmd == "compare":
            qs, note = _load_for_scoring(args)
            named = {Path(r).stem: score(qs, load_runs(r), _ks(args.k), args.min_overlap, args.primary_k) for r in args.runs}
            print(note + format_compare(named))
            if args.out:
                Path(args.out).write_text(json.dumps(named, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.cmd == "cer":
            rep = cer_report([r for r in read_jsonl(args.samples) if not r.get("example")])
            print(format_cer(rep))
            if args.out:
                Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.cmd == "glossary":
            r = glossary_recovery(args.auto, args.curated)
            print(f"curated rows {r['curated_rows']}, automatic rows {r['auto_rows']}")
            print(f"  curated terms recovered:    {_pct(r['term_recall']).strip()}")
            print(f"  curated entities recovered: {_pct(r['entity_recall']).strip()}")
            print(f"  rows only in the automatic table: {r['auto_only_rows']}")
            print("  missing terms (sample): " + ", ".join(r["missing_terms_sample"]))
        elif args.cmd == "suggest":
            files = sorted(Path(args.dir).glob("*_asr.json")) + sorted(Path(args.dir).glob("*_spans.json"))
            if not files:
                raise EvalError(f"No *_asr.json or *_spans.json files in {args.dir}.")
            hits = suggest(args.query, files, args.limit)
            for h in hits:
                print(f"{h['video_id']}  {h['when']}  {h['link']}\n    ...{h['snippet']}...")
            if not hits:
                print("no span contains every query word; try fewer or shorter words")
    except EvalError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
