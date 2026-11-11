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


def rrf_fuse(*ranked: list[str]) -> dict[str, float]:
    """ranked lists of ids, best first -> id -> fused score."""
    out: dict[str, float] = {}
    for lst in ranked:
        for r, i in enumerate(lst):
            out[i] = out.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
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
             n_candidates: int | None = None, per_list: int | None = None) -> dict:
    k = k or config.TOP_K
    n_candidates = n_candidates or getattr(config, "RERANK_CANDIDATES", 30)
    per_list = per_list or getattr(config, "RETRIEVE_PER_LIST", 40)
    m = ix._models()
    name = ix.collection_name()
    flt = m.Filter(must=[m.FieldCondition(key="playlist_id", match=m.MatchValue(value=playlist_id))])
    ents = query_entities(question, gl)
    dense_q = embedder.encode([question])[0]
    si, sv = ix.sparse_query(question, ents)
    dense = _search(client, name, ix.DENSE, dense_q, flt, per_list)
    sparse = _search(client, name, ix.SPARSE, m.SparseVector(indices=si, values=sv), flt, per_list)
    by_id = {str(p.id): p for p in list(dense) + list(sparse)}
    fused = rrf_fuse([str(p.id) for p in dense], [str(p.id) for p in sparse])
    cand = sorted(fused, key=lambda i: -fused[i])[:n_candidates]
    scores = {i: fused[i] for i in cand}
    reranked = False
    if reranker is not None and cand:
        rs = reranker.score(question, [by_id[i].payload["text"] for i in cand])
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
    return {"question": question, "query_entities": ents, "n_candidates": len(cand), "hits": hits}


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
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--playlist")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args(argv)
    if a.playlist:
        config.select_playlist(a.playlist)
    client, emb, rr, gl = load_stack(not a.no_rerank)
    res = retrieve(a.question, client, emb, rr, gl, config.PLAYLIST_ID, k=a.k)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    print(f"entities in the question: {res['query_entities'] or 'none'}")
    for i, h in enumerate(res["hits"], 1):
        print(f"\n[{i}] {h['video_id']} {_hms(h['start'])}-{_hms(h['end'])}  score {h['score']}  {h['url']}")
        print("    " + h["text"][:300].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))