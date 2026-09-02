"""
Step 3 — ask a question, retrieve evidence, generate a grounded answer.

This is the query-time half of the architecture: closed-corpus retrieval
+ citation-constrained generation. No sentence-level verifier yet (that's
MVP-2) — this version relies on the prompt constraint alone, which is
exactly the "single point of failure" the verifier exists to catch later.
Treat this script as the thing that proves you NEED a verifier, not as
a finished trustworthy system.

Run: python 03_ask.py
Needs GEMINI_API_KEY set (see .env) and internet access.
"""
import os

from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

import config


SYSTEM_PROMPT = """You are answering questions using ONLY the evidence 
chunks provided below. Each chunk is numbered and comes from a specific 
video and timestamp in a closed video corpus.

Rules:
1. Every factual sentence in your answer MUST be supported by at least 
one of the numbered evidence chunks. Cite the chunk number in square 
brackets after each factual sentence, like this: [chunk 2].
2. Do NOT use any knowledge you have from outside these chunks, even if 
you are confident it is correct. If the chunks don't fully answer the 
question, say so explicitly rather than filling the gap from memory.
3. If the evidence chunks do not contain enough information to answer 
the question at all, respond with exactly: "I don't have enough 
information in this corpus to answer that." Do not guess.
"""


def retrieve(client, model, question: str, top_k: int):
    query_vector = model.encode([question], normalize_embeddings=True)[0].tolist()
    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="playlist_id", match=MatchValue(value=config.PLAYLIST_ID))]
        ),
        limit=top_k,
    ).points
    return results


def format_evidence(results) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        p = r.payload
        yt_link = f"https://youtu.be/{p['video_id']}?t={int(p['start'])}"
        lines.append(
            f"[chunk {i}] (video={p['video_id']}, {p['start']:.0f}s-{p['end']:.0f}s, "
            f"{yt_link})\n{p['text']}"
        )
    return "\n\n".join(lines)


def main():
    api_key = config.GEMINI_API_KEY
    if not api_key:
        print("[FAIL] set GEMINI_API_KEY in a .env file")
        return

    print("Loading embedding model and connecting to Qdrant...")
    model_name = getattr(config, "EMBED_MODEL_NAME", getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
    gemini_model = getattr(config, "GEMINI_MODEL", getattr(config, "GEMINI_MODEL_NAME", "gemini-2.5-flash"))

    model = SentenceTransformer(model_name)
    qdrant = QdrantClient(path=config.QDRANT_PATH)
    gemini_client = genai.Client(api_key=api_key)

    print(f"\nReady. Corpus = '{config.PLAYLIST_ID}'. Type a question (or 'quit').\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        results = retrieve(qdrant, model, question, config.TOP_K)
        if not results:
            print("\nNo evidence retrieved at all — corpus may be empty or "
                  "playlist_id filter mismatched.\n")
            continue

        evidence = format_evidence(results)

        try:
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=f"Evidence chunks:\n\n{evidence}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=1000,
                ),
            )
            answer_text = response.text

            print(f"\n{answer_text}\n")
            print("Sources retrieved this turn:")
            for i, r in enumerate(results, start=1):
                p = r.payload
                print(f"  [chunk {i}] {p['video_id']} @ {p['start']:.0f}s "
                      f"(score={r.score:.3f}) -> https://youtu.be/{p['video_id']}?t={int(p['start'])}")
            print()

        except Exception as e:
            print(f"\n[FAIL] Error querying Gemini: {e}\n")


if __name__ == "__main__":
    main()