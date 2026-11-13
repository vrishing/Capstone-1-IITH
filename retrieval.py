"""
Retrieval over the span index: dense + sparse, fused, reranked, de-duplicated.

    python retrieval.py "who killed Drona?" [--k 8] [--playlist mahabharata] [--json] [--no-rerank]

    hits = retrieve(question, client, embedder, reranker, glossary, playlist_id, k=8)

Steps
  1. Entities the question mentions (entities.query_entities: any script or English letters).
  2. Dense search (BGE-M3) and sparse search (words, phonetic keys, entity tags), both limited to
     this playlist; fused with reciprocal rank fusion in Python (RRF, k=60).
  3. The top RERANK_CANDIDATES go through the cross-encoder (bge-reranker-v2-m3). If it cannot be
     loaded the fused order is kept and every hit says reranked=False.
  4. Near-duplicate spans (same cluster_id) collapse to the best one; the top k remain.
Each hit: span_id, video_id, start, end, text, entities, cluster_id, rrf, score, reranked, url.
Retrieval never decides what is true; the verifier checks the cited text.
"""
from __future__ import annotations

import json
import sys

import config
import index_store as ix
from entities import query_entities

RRF_K = 60


def rrf_fuse(*ranked: list[str], weights: list[float] | None = None) -> dict[str, float]:
    """ranked lists of ids, best first -> id -> fused score. weights: one per list (default all 1)."""
    out: dict[str, float] = {}
    for n, lst in enumerate(ranked):
        w = weights[n] if weights else 1.0
        for r, i in enumerate(lst):
            out[i] = out.get(i, 0.0) + w / (RRF_K + r + 1)
    return out


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None, max_len: int = 1024):
        import torch
        from sentence_transformers import CrossEncoder
        dev = device or getattr(config, "EMBED_DEVICE", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name or config.RERANK_MODEL_NAME, device=dev, max_length=max_len)
        self.device = dev

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.model.predict([(query, t) for t in texts], batch_size=8,
                                                     show_progress_bar=False)]


def _search(client, name: str, using: str, query, flt, limit: int):
    return client.query_points(collection_name=name, query=query, using=using, query_filter=flt,
                               limit=limit, with_payload=True).points


def retrieve(question: str, client, embedder, reranker, gl, playlist_id: str, k: int | None = None,
             n_candidates: int | None = None, per_list: int | None = None, mode: str = "hybrid",
             use_entities: bool = True, rewrite_fn=None, dense_weight: float = 1.0,
             sparse_weight: float | None = None) -> dict:
    """mode: hybrid | dense | sparse (the last two exist for ablations). use_entities=False drops the
    entity tokens from the sparse query. rewrite_fn(question) -> the question in the playlist's language
    ("" = none): the dense search then runs for both versions, the sparse query and the reranker use the
    rewritten one. dense_weight / sparse_weight scale each list in the fusion."""
    k = k or config.TOP_K
    sparse_weight = getattr(config, "RETRIEVE_SPARSE_WEIGHT", 0.3) if sparse_weight is None else sparse_weight
    n_candidates = n_candidates or getattr(config, "RERANK_CANDIDATES", 30)
    per_list = per_list or getattr(config, "RETRIEVE_PER_LIST", 40)
    m = ix._models()
    name = ix.collection_name()
    flt = m.Filter(must=[m.FieldCondition(key="playlist_id", match=m.MatchValue(value=playlist_id))])
    if mode not in ("hybrid", "dense", "sparse"):
        raise ValueError("mode must be hybrid, dense or sparse")
    rewritten = (rewrite_fn(question) if rewrite_fn else "") or ""
    both = f"{question} {rewritten}".strip()
    ents = query_entities(both, gl) if use_entities else []
    dense_lists, sparse = [], []
    if mode in ("hybrid", "dense"):
        for q in [question] + ([rewritten] if rewritten else []):
            dense_lists.append(_search(client, name, ix.DENSE, embedder.encode([q])[0], flt, per_list))
    if mode in ("hybrid", "sparse"):
        si, sv = ix.sparse_query(both, ents)
        sparse = _search(client, name, ix.SPARSE, m.SparseVector(indices=si, values=sv), flt, per_list)
    by_id = {str(p.id): p for p in [x for lst in dense_lists for x in lst] + list(sparse)}
    fused = rrf_fuse(*[[str(p.id) for p in lst] for lst in dense_lists], [str(p.id) for p in sparse],
                     weights=[dense_weight] * len(dense_lists) + [sparse_weight])
    cand = sorted(fused, key=lambda i: -fused[i])[:n_candidates]
    scores = {i: fused[i] for i in cand}
    reranked = False
    if reranker is not None and cand:
        rs = reranker.score(rewritten or question, [by_id[i].payload["text"] for i in cand])
        scores = dict(zip(cand, rs))
        reranked = True
    cand.sort(key=lambda i: -scores[i])
    hits, seen = [], set()
    for i in cand:
        p = by_id[i].payload
        if p["cluster_id"] in seen:
            continue
        seen.add(p["cluster_id"])
        hits.append({"point_id": i, "span_id": p["span_id"], "video_id": p["video_id"],
                     "start": p["start"], "end": p["end"], "text": p["text"],
                     "entities": p.get("entities", []), "cluster_id": p["cluster_id"],
                     "rrf": round(fused[i], 5), "score": round(scores[i], 4), "reranked": reranked,
                     "url": f"https://youtu.be/{p['video_id']}?t={int(p['start'])}"})
        if len(hits) == k:
            break
    return {"question": question, "rewritten": rewritten, "query_entities": ents, "n_candidates": len(cand),
            "hits": hits}


def run_row(res: dict) -> dict:
    """One line of an eval run file from a retrieve() result."""
    return {"retrieved": [{"video_id": h["video_id"], "start": h["start"], "end": h["end"],
                           "span_id": h["span_id"]} for h in res["hits"]],
            "answer": "", "abstained": False, "citations": []}


def write_run(questions_path: str, out_path: str, stack, mode: str = "hybrid", k: int = 10, **kw) -> int:
    """Retrieval-only run over a question file (eval_harness format): the baseline and the ablations."""
    from eval_harness import collect_run, load_questions
    client, emb, rr, gl = stack
    qs = load_questions(questions_path)
    rows = collect_run(qs, lambda q: run_row(retrieve(q["question"], client, emb, rr, gl, config.PLAYLIST_ID,
                                                       k=k, mode=mode, **kw)), out_path)
    bad = sum(1 for r in rows if r.get("error"))
    print(f"{len(rows)} questions -> {out_path}" + (f" ({bad} errors)" if bad else ""))
    return 0 if not bad else 2


def _glossary():
    import correction
    return correction.get_glossary(getattr(config, "ALIAS_TABLE_PATH", None),
                                   getattr(config, "PROTECTED_WORDS_PATH", None),
                                   getattr(config, "DOMAIN_PACK_PATH", None))


def load_stack(rerank: bool = True):
    """(client, embedder, reranker-or-None, glossary) for the active playlist."""
    client = ix.open_client()
    emb = ix.Embedder()
    rr = None
    if rerank:
        try:
            rr = Reranker()
        except Exception as e:                       # offline, missing model, out of memory
            print(f"reranker unavailable ({type(e).__name__}: {str(e)[:150]}); using fused order",
                  file=sys.stderr)
    return client, emb, rr, _glossary()


def _hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--eval", metavar="QUESTIONS", help="write a retrieval run file for this question file")
    ap.add_argument("--out", help="with --eval: run file to write")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--playlist")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--rewrite", action="store_true", help="also search with the question rewritten into the playlist language (uses the LLM)")
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--sparse-weight", type=float, default=None,
                    help="weight of the sparse list in the fusion (default: RETRIEVE_SPARSE_WEIGHT, 0.3)")
    ap.add_argument("--no-entities", action="store_true", help="leave entity tags out of the sparse query")
    a = ap.parse_args(argv)
    if a.playlist:
        config.select_playlist(a.playlist)
    stack = load_stack(not a.no_rerank)
    opts = {"dense_weight": a.dense_weight, "sparse_weight": a.sparse_weight, "use_entities": not a.no_entities}
    if a.rewrite or getattr(config, "RETRIEVE_REWRITE", False):
        from query_rewrite import make_rewrite_fn
        opts["rewrite_fn"] = make_rewrite_fn()
    if a.eval:
        if not a.out:
            print("--eval needs --out", file=sys.stderr)
            return 2
        return write_run(a.eval, a.out, stack, a.mode, a.k, **opts)
    if not a.question:
        print("give a question, or --eval with --out", file=sys.stderr)
        return 2
    client, emb, rr, gl = stack
    res = retrieve(a.question, client, emb, rr, gl, config.PLAYLIST_ID, k=a.k, mode=a.mode, **opts)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    if res.get("rewritten"):
        print(f"searched also as: {res['rewritten']}")
    print(f"entities in the question: {res['query_entities'] or 'none'}")
    for i, h in enumerate(res["hits"], 1):
        print(f"\n[{i}] {h['video_id']} {_hms(h['start'])}-{_hms(h['end'])}  score {h['score']}  {h['url']}")
        print("    " + h["text"][:300].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))