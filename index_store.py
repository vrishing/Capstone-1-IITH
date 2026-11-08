"""
Vector index for evidence spans: local on-disk Qdrant, dense (BGE-M3) + sparse (BM25-style).

    client = open_client()                       # QdrantClient(path=config.QDRANT_PATH), no server
    ensure_collection(client)
    n = upsert_video(client, spans, dense_vectors, playlist_id)
    dense = Embedder().encode(texts)

One collection (config.SPANS_COLLECTION, default "spans_v1") holds every playlist; each point
carries playlist_id, so a query always filters on it. Point IDs are spans.point_id(span_id), so
indexing the same spans twice leaves the point count unchanged, and re-indexing a video first
deletes that video's old points (a re-chunked video leaves no stale spans behind).

Sparse side: BGE-M3's own sparse head needs FlagEmbedding, which is not a dependency. Instead each
span gets a BM25 term-frequency vector over three kinds of token, and Qdrant applies the IDF
(Modifier.IDF) itself, so nothing here depends on corpus statistics:
    word        the word as written, lowercased, punctuation stripped
    "k:" + key  its script-independent phonetic key (romanize.phonetic_key): an ASR spelling
                variant or the same name in another Brahmic script lands on the same token
    "e:" + id   each glossary entity tagged on the span (a question that names an entity gets the
                same tokens through entities.query_entities)
Token -> index is crc32, stable across runs and machines. Stdlib at import time; qdrant_client,
sentence_transformers and torch load only when used.
"""
from __future__ import annotations

import zlib
from collections import Counter

import config
from entities import _tokens
from romanize import phonetic_key

INDEX_ALGO = "index_v1"
K1 = 1.2
DENSE, SPARSE = "dense", "sparse"
BATCH = 128


def collection_name() -> str:
    return getattr(config, "SPANS_COLLECTION", "spans_v1")


def _models():
    from qdrant_client import models
    return models


# ---------------------------------------------------------------------------------------
# tokens and sparse vectors
# ---------------------------------------------------------------------------------------
def tokens_for(text: str, entity_ids=()) -> list[str]:
    out: list[str] = []
    for w in _tokens(text.lower()):
        out.append(w)
        k = phonetic_key(w)
        if len(k) >= 2:
            out.append("k:" + k)
    out.extend("e:" + e for e in entity_ids)
    return out


def _tid(tok: str) -> int:
    return zlib.crc32(tok.encode("utf-8")) & 0x7FFFFFFF


def sparse_doc(text: str, entity_ids=()) -> tuple[list[int], list[float]]:
    """BM25 term-frequency weights (no length normalisation: spans are all 45-90 s)."""
    acc: dict[int, float] = {}
    for tok, tf in Counter(tokens_for(text, entity_ids)).items():
        i = _tid(tok)
        acc[i] = acc.get(i, 0.0) + tf * (K1 + 1) / (tf + K1)
    idx = sorted(acc)
    return idx, [round(acc[i], 4) for i in idx]


def sparse_query(text: str, entity_ids=()) -> tuple[list[int], list[float]]:
    """Each distinct query token once, weight 1; Qdrant multiplies by IDF."""
    idx = sorted({_tid(t) for t in tokens_for(text, entity_ids)})
    return idx, [1.0] * len(idx)


# ---------------------------------------------------------------------------------------
# dense embedder
# ---------------------------------------------------------------------------------------
class Embedder:
    """BGE-M3 dense vectors, L2-normalised. Device from EMBED_DEVICE: auto | cpu | cuda | cuda:N."""

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 max_len: int | None = None, batch: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        dev = device or getattr(config, "EMBED_DEVICE", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name or config.EMBED_MODEL_NAME, device=dev)
        self.model.max_seq_length = max_len or getattr(config, "EMBED_MAX_LEN", 1024)
        if dev.startswith("cuda"):
            self.model.half()
        self.batch = batch or getattr(config, "EMBED_BATCH", 16)
        self.device = dev

    def encode(self, texts: list[str]) -> list[list[float]]:
        # long spans first: less padding, and an out-of-memory error shows up immediately
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        vecs = self.model.encode([texts[i] for i in order], batch_size=self.batch,
                                 normalize_embeddings=True, show_progress_bar=False)
        out: list = [None] * len(texts)
        for pos, i in enumerate(order):
            out[i] = [float(x) for x in vecs[pos]]
        return out


# ---------------------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------------------
def open_client():
    from qdrant_client import QdrantClient
    return QdrantClient(path=config.QDRANT_PATH)


def ensure_collection(client, name: str | None = None, dim: int | None = None) -> None:
    name = name or collection_name()
    if client.collection_exists(name):
        return
    m = _models()
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: m.VectorParams(size=dim or config.EMBED_DIM, distance=m.Distance.COSINE)},
        sparse_vectors_config={SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)},
    )


def payload_of(span: dict, playlist_id: str, cluster_id: str | None = None) -> dict:
    p = {"playlist_id": playlist_id, "span_id": span["span_id"], "video_id": span["video_id"],
         "start": span["start"], "end": span["end"], "text": span["text"],
         "language": span.get("language", ""), "entities": list(span.get("entities", [])),
         "cluster_id": cluster_id or span.get("cluster_id") or span["span_id"]}
    if "raw_text" in span:
        p["raw_text"] = span["raw_text"]
    return p


def upsert_video(client, spans: list[dict], dense: list[list[float]], playlist_id: str,
                 clusters: dict[str, str] | None = None, name: str | None = None) -> int:
    """Replace every point of this video with these spans. Returns the number of points."""
    if len(spans) != len(dense):
        raise ValueError("one dense vector per span")
    name = name or collection_name()
    m = _models()
    vids = {s["video_id"] for s in spans}
    for v in vids:
        client.delete(collection_name=name, points_selector=m.FilterSelector(filter=m.Filter(must=[
            m.FieldCondition(key="video_id", match=m.MatchValue(value=v)),
            m.FieldCondition(key="playlist_id", match=m.MatchValue(value=playlist_id))])))
    from spans import point_id
    pts = []
    for s, d in zip(spans, dense):
        idx, val = sparse_doc(s["text"], s.get("entities", ()))
        pts.append(m.PointStruct(
            id=point_id(s["span_id"]),
            vector={DENSE: d, SPARSE: m.SparseVector(indices=idx, values=val)},
            payload=payload_of(s, playlist_id, (clusters or {}).get(s["span_id"]))))
    for a in range(0, len(pts), BATCH):
        client.upsert(collection_name=name, points=pts[a:a + BATCH])
    return len(pts)


def count(client, playlist_id: str | None = None, name: str | None = None) -> int:
    name = name or collection_name()
    if not client.collection_exists(name):
        return 0
    flt = None
    if playlist_id:
        m = _models()
        flt = m.Filter(must=[m.FieldCondition(key="playlist_id", match=m.MatchValue(value=playlist_id))])
    return client.count(collection_name=name, count_filter=flt, exact=True).count


if __name__ == "__main__":
    c = open_client()
    print(f"collection {collection_name()}: {count(c)} points; this playlist ({config.PLAYLIST_ID}): "
          f"{count(c, config.PLAYLIST_ID)}")