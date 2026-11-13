"""
Query rewriting for cross-language questions.

A question in English is searched against transcripts in Telugu or Hindi. Dense search copes, but
word-level (sparse) search and the reranker do not: no English word ever appears in the transcript.
rewrite() asks the LLM once for the question in the playlist's language, names written in that
script the way a speaker would say them. Retrieval then searches with both versions.

    rewrite(question, language, gen, topic="", cache_path=...) -> str   ("" = no rewrite)

A question already in a Brahmic script is left alone. Answers are cached on disk, so a question is
rewritten once. Anything unusable from the model (no target script, too long, an error) gives "":
retrieval falls back to the original question, never fails because of the rewrite.
"""
from __future__ import annotations

import hashlib
import json
import os

from romanize import script_of

LANG_NAMES = {"te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam",
              "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia",
              "as": "Assamese", "ne": "Nepali", "sa": "Sanskrit"}
REWRITE_ALGO = "rewrite_v1"


def needs_rewrite(question: str) -> bool:
    return not script_of(question)                    # Latin (or no script at all) needs it


def _prompt(question: str, lang_name: str, topic: str) -> str:
    return (f"Translate this question into {lang_name} for searching speech transcripts of lectures about "
            f"{topic or 'a video playlist'}. Write names of people, places and terms in {lang_name} script, "
            f"the way a speaker would say them. Keep it short, one sentence, no explanation.\n"
            f"Question: {question}\n"
            'Reply with JSON only: {"query": "..."}')


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def rewrite(question: str, language: str, gen, topic: str = "", cache_path: str | None = None) -> str:
    if not needs_rewrite(question):
        return ""
    lang_name = LANG_NAMES.get(language)
    if not lang_name:
        return ""
    key = hashlib.sha256(f"{REWRITE_ALGO}|{language}|{topic}|{question}".encode("utf-8")).hexdigest()[:24]
    cache = _load(cache_path) if cache_path else {}
    if key in cache:
        return cache[key]
    try:
        raw = gen(_prompt(question, lang_name, topic))
        try:
            d = json.loads(raw)
        except ValueError:
            from llm_client import extract_json
            d = json.loads(extract_json(raw))
        out = " ".join(str(d.get("query", "")).split())
    except Exception:
        return ""                                      # not cached: try again next time
    if not (3 <= len(out) <= 300 and script_of(out)):
        return ""
    if cache_path:
        cache[key] = out
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)
    return out


def make_rewrite_fn():
    """question -> rewritten question, for the active playlist and the configured LLM."""
    import config
    import llm_client
    from run_pipeline import playlist_topic
    lang = getattr(config, "TARGET_LANGUAGE", "te")
    topic = playlist_topic()
    path = getattr(config, "QUERY_REWRITE_CACHE", os.path.join("data", "cache", "query_rewrite.json"))

    def gen(prompt):
        return llm_client.generate_default(prompt, role="answer", json_output=True, max_output_tokens=1500)
    return lambda q: rewrite(q, lang, gen, topic, path)