"""
Measure the verifier on its own: how often it accepts a sentence its passages do NOT support
(false accept, the dangerous error) and how often it rejects one they do (false reject, which
causes false abstains). The answer-level eval is too coarse for this: 36 questions, and a wrong
sentence can still cite the right place.

    python verify_bench.py export --out eval/verifier_items.jsonl          # from data/traces
    # label each item: "label": "y" (the passages support the statement) or "n" (they do not)
    python verify_bench.py run --items eval/verifier_items.jsonl --mode locate_judge [--thinking]
    python verify_bench.py run --items eval/verifier_items.jsonl --mode single

Items come from the statements the generator actually wrote in earlier runs (traces), with the
passages each one cited, so they carry the real difficulty: noisy ASR, other-language passages,
plausible but wrong claims. "partial" counts as accepted (the answer is shown, with a caution).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys


def export(trace_dir: str, out_path: str) -> int:
    import answer
    seen, rows = set(), []
    for p in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        if os.path.basename(p).startswith("_"):
            continue
        try:
            t = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hits = t.get("evidence") or []
        for a in t.get("attempts", []):
            try:
                ok, sents = answer.parse_generation(a.get("generation", ""), len(hits))
            except ValueError:
                continue
            for j, s in enumerate(sents if ok else []):
                if not s["cites"]:
                    continue
                spans = [hits[c - 1]["span_id"] for c in s["cites"]]
                key = hashlib.sha256(json.dumps([t["question"], s["text"], spans], ensure_ascii=False)
                                     .encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                old = (a.get("verdicts") or [None] * (j + 1))
                rows.append({"id": key, "question": t["question"], "statement": s["text"], "span_ids": spans,
                             "passages": [hits[c - 1]["text"] for c in s["cites"]],
                             "old_verdict": old[j] if j < len(old) else None, "label": ""})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _ci(xs: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def score(results: list[dict]) -> dict:
    neg = [int(r["accepted"]) for r in results if r["label"] == "n"]
    pos = [int(not r["accepted"]) for r in results if r["label"] == "y"]
    err = sum(1 for r in results if r.get("error"))
    return {"n": len(results), "unsupported_items": len(neg), "supported_items": len(pos), "errors": err,
            "false_accept": sum(neg) / len(neg) if neg else None, "false_accept_ci": _ci(neg),
            "false_reject": sum(pos) / len(pos) if pos else None, "false_reject_ci": _ci(pos)}


def run(items: list[dict], gen, mode: str, require_quote: bool = True) -> list[dict]:
    import verify
    out = []
    for it in items:
        lab = str(it.get("label", "")).strip().lower()[:1]
        if lab not in ("y", "n"):
            continue
        hits = [{"text": p} for p in it["passages"]]
        sents = [{"text": it["statement"], "cites": list(range(1, len(hits) + 1))}]
        r = {"id": it["id"], "label": lab, "statement": it["statement"]}
        try:
            v, dbg = verify.verify(it["question"], sents, hits, gen, mode=mode, require_quote=require_quote)
            r.update(verdict=v[0], accepted=v[0] in ("supported", "partial"),
                     says=(dbg.get("says") or {}).get(0, ""), quote_failed=bool(dbg.get("quote_failures")))
        except Exception as e:                   # an error abstains in the real system: not accepted
            r.update(verdict="error", accepted=False, error=f"{type(e).__name__}: {str(e)[:120]}")
        out.append(r)
    return out


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--traces", default=None)
    e.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", required=True)
    r.add_argument("--mode", default="locate_judge", choices=["locate_judge", "single"])
    r.add_argument("--thinking", action="store_true")
    r.add_argument("--out", help="per-item results (default: <items>.<mode>[.think].results.jsonl)")
    a = ap.parse_args(argv)
    import config
    if a.cmd == "export":
        n = export(a.traces or getattr(config, "TRACE_DIR", "data/traces"), a.out)
        print(f"{n} statements -> {a.out}\nWrite \"y\" or \"n\" in each item's \"label\", then run: verify_bench.py run")
        return 0
    if a.thinking:
        config.LLM_VERIFIER_THINKING = True
    import answer
    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    res = run(items, answer.make_gen(), a.mode)
    out = a.out or f"{a.items}.{a.mode}{'.think' if a.thinking else ''}.results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    s = score(res)
    fa, fr = s["false_accept"], s["false_reject"]
    print(f"mode={a.mode} thinking={a.thinking}: {s['n']} labelled items "
          f"({s['unsupported_items']} unsupported, {s['supported_items']} supported), {s['errors']} errors")
    if fa is not None:
        print(f"  false accept  {fa:6.1%}   95% CI {s['false_accept_ci'][0]:.1%}-{s['false_accept_ci'][1]:.1%}"
              "   (wrong sentences let through)")
    if fr is not None:
        print(f"  false reject  {fr:6.1%}   95% CI {s['false_reject_ci'][0]:.1%}-{s['false_reject_ci'][1]:.1%}"
              "   (good sentences dropped)")
    print(f"  per item: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))