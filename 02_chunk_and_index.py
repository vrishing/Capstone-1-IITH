"""
Step 2 — turn raw caption lines into timestamped evidence chunks, embed
them, and store them in a local Qdrant collection.

This is the "hybrid index" stage from the architecture, simplified to
dense-only (will add BM25 / entity indexing once this works).

Run: python 02_chunk_and_index.py
Needs internet access the FIRST time only, to download the BGE-M3 weights.
"""
from __future__ import annotations
import glob
import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

import config


def merge_into_chunks(entries: list[dict], chunk_seconds: int, overlap_seconds: int):
    """
    Merge consecutive caption lines into ~chunk_seconds windows.

    Captions arrive as many short lines (e.g. 2-5 seconds each). We slide
    a window over them so each stored chunk has enough context to stand
    alone as evidence, while keeping the exact start/end timestamp so we
    can still produce a precise YouTube deep link and evidence clip later.
    """
    if not entries:
        return []

    chunks = []
    window_start_idx = 0
    n = len(entries)

    while window_start_idx < n:
        window_start_time = entries[window_start_idx]["start"]
        text_parts = []
        end_idx = window_start_idx
        end_time = window_start_time

        while end_idx < n and (entries[end_idx]["start"] - window_start_time) < chunk_seconds:
            text_parts.append(entries[end_idx]["text"])
            end_time = entries[end_idx]["start"] + entries[end_idx]["duration"]
            end_idx += 1

        chunks.append({
            "start": round(window_start_time, 2),
            "end": round(end_time, 2),
            "text": " ".join(text_parts).strip(),
        })

        # slide the window forward, leaving `overlap_seconds` of overlap
        next_start_time = window_start_time + (chunk_seconds - overlap_seconds)
        next_idx = window_start_idx
        while next_idx < n and entries[next_idx]["start"] < next_start_time:
            next_idx += 1
        if next_idx <= window_start_idx:
            next_idx = window_start_idx + 1  # guarantee forward progress
        window_start_idx = next_idx

    return [c for c in chunks if c["text"]]


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"[ok] created collection '{config.COLLECTION_NAME}'")
    else:
        print(f"[ok] collection '{config.COLLECTION_NAME}' already exists")


def main():
    print(f"Loading embedding model {config.EMBED_MODEL_NAME} "
          f"(first run downloads ~2GB, be patient)...")
    model = SentenceTransformer(config.EMBED_MODEL_NAME)

    client = QdrantClient(path=config.QDRANT_PATH)
    ensure_collection(client)

    point_id = 0
    points = []

    for path in sorted(glob.glob("data/transcripts/*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        video_id = data["video_id"]
        chunks = merge_into_chunks(
            data["entries"], config.CHUNK_SECONDS, config.CHUNK_OVERLAP_SECONDS
        )
        print(f"[chunk] {video_id}: {len(data['entries'])} caption lines "
              f"-> {len(chunks)} evidence chunks")

        texts = [c["text"] for c in chunks]
        if not texts:
            continue
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "playlist_id": config.PLAYLIST_ID,
                    "video_id": video_id,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                },
            ))
            point_id += 1

    if points:
        client.upsert(collection_name=config.COLLECTION_NAME, points=points)
        print(f"[ok] indexed {len(points)} chunks into Qdrant at {config.QDRANT_PATH}")
    else:
        print("[FAIL] no chunks to index — did step 1 (fetch transcripts) succeed?")


if __name__ == "__main__":
    main()
