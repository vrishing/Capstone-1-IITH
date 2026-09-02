"""
Step 7 — Re-index the full corpus using ASR-derived chunks.

Reads all *_chunks.json files from config.ASR_OUTPUT_DIR, embeds each chunk
with BGE-M3, and stores them in Qdrant tagged with config.PLAYLIST_ID.

Run after 06_asr_to_chunks.py has created the chunk files.
"""
from __future__ import annotations
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

import config


def load_chunks_from_asr():
    """Load all ASR-derived chunks from data/asr_output/ and attach video_id."""
    all_chunks = []
    for chunks_path in sorted(Path(config.ASR_OUTPUT_DIR).glob("*_chunks.json")):
        with open(chunks_path, encoding='utf-8') as f:
            data = json.load(f)
        video_id = data["video_id"]
        for chunk in data["chunks"]:
            # Add video_id to the chunk dict
            chunk["video_id"] = video_id
            all_chunks.append(chunk)
    return all_chunks


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

    # Load all ASR chunks
    all_chunks = load_chunks_from_asr()
    if not all_chunks:
        print("[FAIL] No ASR chunk files found. Run 06_asr_to_chunks.py first.")
        return

    print(f"Found {len(all_chunks)} chunks total. Embedding...")
    texts = [c["text"] for c in all_chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "playlist_id": config.PLAYLIST_ID,
                "video_id": chunk["video_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            }
        ))

    if points:
        # Upsert in batches to be safe (1000 points per batch)
        batch_size = 1000
        for start in range(0, len(points), batch_size):
            batch = points[start:start+batch_size]
            client.upsert(collection_name=config.COLLECTION_NAME, points=batch)
            print(f"  indexed {min(start + batch_size, len(points))} / {len(points)}")
        print(f"[ok] indexed {len(points)} ASR-derived chunks into Qdrant at {config.QDRANT_PATH}")
    else:
        print("[FAIL] no chunks to index — check that 06_asr_to_chunks.py produced files.")


if __name__ == "__main__":
    main()