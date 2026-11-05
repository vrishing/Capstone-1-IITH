"""
Single entry point for every LLM call. Two backends behind one function:

  gemini   Google's API (google-genai, or the older google-generativeai as a fallback)
  openai   any server that speaks the OpenAI chat API: vLLM, Ollama, llama.cpp server, SGLang.
           This is the self-hosted route. Stdlib only, so no extra packages.

Scripts call `generate(...)` or, better, `generate_default(prompt, role=...)`, which reads the
backend and model from config.py (LLM_BACKEND, LLM_MODEL, ...). Neither SDK is imported unless
its backend is used, so a GPU server without the Gemini SDK works.

Local models differ from Gemini in ways this module absorbs:
  - they may wrap JSON in ```json fences or add a sentence around it, so json_output=True
    returns a clean JSON string or raises LLMError;
  - reasoning models print <think>...</think> first, which is removed;
  - a shared GPU can be slow or busy, so there are timeouts and retries with backoff.
Every failure raises LLMError, which the verifier treats as "abstain" (fail closed).

Check a server and model before relying on them:
  python llm_client.py --check [--role verifier]
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

BACKEND: str | None = None   # which Gemini SDK is installed (None if neither); kept for older scripts

try:
    from google import genai as _new_genai
    from google.genai import types as _new_types
    BACKEND = "google-genai"
except ImportError:
    _new_genai = None
    _new_types = None

if BACKEND is None:
    try:
        import google.generativeai as _old_genai
        BACKEND = "google-generativeai"
    except ImportError:
        _old_genai = None
else:
    _old_genai = None

_client_cache: dict = {}
_sleep = time.sleep   # tests replace this
LOCAL_BACKENDS = {"openai", "local", "vllm", "ollama"}


class LLMError(RuntimeError):
    """Any failure to get a usable answer from the model."""


# ======================================================================================
# Cleaning model output
# ======================================================================================
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.S)


def clean_text(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json(text: str) -> str:
    """The first valid JSON value in the text, as a normalised string. Raises LLMError if none."""
    t = clean_text(text)
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    try:
        json.loads(t)
        return t
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(t[i:])
                return json.dumps(obj, ensure_ascii=False)
            except ValueError:
                continue
    raise LLMError(f"no valid JSON in the model output: {t[:200]!r}")


# ======================================================================================
# OpenAI-compatible backend
# ======================================================================================
def _chat_openai(prompt: str, *, model: str, base_url: str, api_key: str | None,
                 system_instruction: str | None, temperature: float, max_output_tokens: int,
                 json_output: bool, timeout: float, retries: int = 3) -> str:
    if not model:
        raise LLMError("No model name for the local backend. Set LLM_MODEL in .env (the name the server serves).")
    url = base_url.rstrip("/") + "/chat/completions"
    messages = ([{"role": "system", "content": system_instruction}] if system_instruction else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_output_tokens}
    if json_output:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last = "no attempt made"
    attempt = 0
    while attempt < retries:
        attempt += 1
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last = f"HTTP {e.code}: {detail}"
            if e.code == 400 and "response_format" in body and re.search(r"response_format|json", detail, re.I):
                del body["response_format"]          # server has no JSON mode; extract_json copes
                attempt -= 1
                continue
            if e.code in (429, 500, 502, 503, 504):
                _sleep(min(2.0 * 2 ** (attempt - 1), 30.0))
                continue
            raise LLMError(f"model server refused the request ({last})") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            _sleep(min(2.0 * 2 ** (attempt - 1), 30.0))
        except ValueError as e:                      # reply was not JSON at all
            raise LLMError(f"model server sent an unreadable reply: {e}") from None
    else:
        raise LLMError(f"model server unavailable after {retries} attempts ({last}). Is it running at {base_url}?")

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"unexpected reply shape from the model server: {str(data)[:200]}") from None
    if not (text or "").strip():
        raise LLMError("the model returned an empty answer")
    return text


# ======================================================================================
# Public API
# ======================================================================================
def generate(
    prompt: str,
    *,
    model: str,
    api_key: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1000,
    json_output: bool = False,
    backend: str | None = None,
    base_url: str | None = None,
    timeout: float = 180.0,
) -> str:
    """Return the model's text response.

    json_output=True returns a JSON string you can pass to json.loads (or raises LLMError).
    backend: "gemini" or "openai" (default: the LLM_BACKEND environment variable, else gemini).
    """
    backend = (backend or os.environ.get("LLM_BACKEND") or "gemini").lower()

    if backend in LOCAL_BACKENDS:
        text = _chat_openai(prompt, model=model, base_url=base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                            api_key=api_key, system_instruction=system_instruction, temperature=temperature,
                            max_output_tokens=max_output_tokens, json_output=json_output, timeout=timeout)
        return extract_json(text) if json_output else clean_text(text)

    if backend != "gemini":
        raise ValueError(f"Unknown LLM backend {backend!r}. Use 'gemini' or 'openai'.")
    if BACKEND is None:
        raise ImportError(
            "No Gemini SDK found. Install one of: `pip install google-genai` (preferred) or "
            "`pip install google-generativeai`, or set LLM_BACKEND=openai to use a local model."
        )
    if BACKEND == "google-genai":
        client = _client_cache.get(api_key)
        if client is None:
            client = _new_genai.Client(api_key=api_key)
            _client_cache[api_key] = client
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_new_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **({"response_mime_type": "application/json"} if json_output else {}),
            ),
        )
        return response.text

    # Old SDK path (needs google-generativeai >= 0.5 for system_instruction)
    _old_genai.configure(api_key=api_key)
    gm = _old_genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
    response = gm.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            **({"response_mime_type": "application/json"} if json_output else {}),
        },
    )
    return response.text


def settings(role: str = "answer") -> dict:
    """Backend, model and credentials for a role ("answer" or "verifier"), from config.py."""
    import config
    backend = str(getattr(config, "LLM_BACKEND", "gemini")).lower()
    if backend == "gemini":
        model = getattr(config, "VERIFIER_MODEL", None) if role == "verifier" else None
        model = model or getattr(config, "GEMINI_MODEL_NAME")
        return {"backend": "gemini", "model": model, "api_key": getattr(config, "GEMINI_API_KEY", None)}
    if backend not in LOCAL_BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND {backend!r}. Use 'gemini' or 'openai'.")
    model = (getattr(config, "LLM_VERIFIER_MODEL", "") if role == "verifier" else "") or getattr(config, "LLM_MODEL", "")
    if not model:
        raise LLMError("LLM_BACKEND=openai needs LLM_MODEL: the model name your server is serving.")
    return {"backend": "openai", "model": model, "base_url": getattr(config, "LLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key": getattr(config, "LLM_API_KEY", None), "timeout": float(getattr(config, "LLM_TIMEOUT_S", 180.0))}


def generate_default(prompt: str, *, role: str = "answer", **kw) -> str:
    """generate() with the backend and model chosen in config.py. New code calls this."""
    return generate(prompt, **{**settings(role), **kw})


# ======================================================================================
# Is this model usable? (python llm_client.py --check)
# ======================================================================================
_VERIFIER_SYSTEM = ('You check whether a statement is supported by a passage. Reply with JSON only: '
                    '{"verdict": "supported"} or {"verdict": "unsupported"}.')
# (passage, statement, expected). A smoke test, not a benchmark: six cases cannot prove quality,
# but a model that misses more than one of them should not be your verifier.
VERIFIER_PROBES = [
    ("అర్జునుడు ద్రోణాచార్యుని దగ్గర విలువిద్య నేర్చుకున్నాడు.", "అర్జునుడు విలువిద్య నేర్చుకున్నాడు.", "supported"),
    ("అర్జునుడు ద్రోణాచార్యుని దగ్గర విలువిద్య నేర్చుకున్నాడు.", "అర్జునుడు పరశురాముని దగ్గర విలువిద్య నేర్చుకున్నాడు.", "unsupported"),
    ("భీష్ముడు గంగ కుమారుడు.", "భీష్ముని తల్లి గంగ.", "supported"),
    ("భీష్ముడు గంగ కుమారుడు.", "భీష్ముడు కుంతి కుమారుడు.", "unsupported"),
    ("कर्ण सूर्य और कुंती के पुत्र थे।", "कर्ण कुंती के पुत्र थे।", "supported"),
    ("कर्ण सूर्य और कुंती के पुत्र थे।", "कर्ण गांधारी के पुत्र थे।", "unsupported"),
]


def run_check(role: str = "answer", gen=None) -> dict:
    """Connectivity, plain text, Telugu round trip, JSON mode, and the six verifier probes."""
    gen = gen or generate_default
    out = {"plain_ok": False, "telugu_ok": False, "json_ok": False, "verifier_correct": 0,
           "verifier_total": len(VERIFIER_PROBES), "errors": [], "avg_latency_s": None}
    lat = []

    def call(prompt, **kw):
        t0 = time.time()
        try:
            r = gen(prompt, role=role, **kw)
            lat.append(time.time() - t0)
            return r
        except Exception as e:  # noqa: BLE001 - reported, not hidden
            out["errors"].append(f"{type(e).__name__}: {e}")
            return None

    r = call("Reply with the single word: ready", max_output_tokens=200)
    out["plain_ok"] = bool(r and "ready" in r.lower())
    if out["errors"]:                       # the very first call failed: server or settings problem
        out["usable"] = False
        return out
    r = call("Repeat exactly this word and nothing else: కర్ణుడు", max_output_tokens=200)
    out["telugu_ok"] = bool(r and "కర్ణుడు" in r)
    r = call('Reply with JSON only: {"ok": true}', json_output=True, max_output_tokens=200)
    try:
        out["json_ok"] = bool(r) and json.loads(r).get("ok") is True
    except (ValueError, AttributeError):
        out["json_ok"] = False
    for passage, statement, expected in VERIFIER_PROBES:
        r = call(f"Passage:\n{passage}\n\nStatement:\n{statement}", system_instruction=_VERIFIER_SYSTEM,
                 json_output=True, max_output_tokens=200)
        try:
            if r and json.loads(r).get("verdict") == expected:
                out["verifier_correct"] += 1
        except (ValueError, AttributeError):
            pass
    out["avg_latency_s"] = round(sum(lat) / len(lat), 2) if lat else None
    out["usable"] = (out["plain_ok"] and out["telugu_ok"] and out["json_ok"]
                     and out["verifier_correct"] >= out["verifier_total"] - 1)
    return out


def _main(argv: list[str]) -> int:
    if "--check" not in argv:
        print(f"Gemini SDK installed: {BACKEND or 'none'}. Use --check [--role verifier] to test the configured model.")
        return 0
    role = argv[argv.index("--role") + 1] if "--role" in argv else "answer"
    try:
        s = settings(role)
    except (LLMError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"backend {s['backend']}, model {s['model']}" + (f", server {s['base_url']}" if "base_url" in s else ""))
    r = run_check(role)
    mark = lambda ok: "ok  " if ok else "FAIL"
    print(f"  {mark(r['plain_ok'])} plain answer\n  {mark(r['telugu_ok'])} Telugu round trip\n  {mark(r['json_ok'])} JSON output")
    print(f"  {mark(r['verifier_correct'] >= r['verifier_total'] - 1)} verifier probes {r['verifier_correct']}/{r['verifier_total']}")
    print(f"  average latency per call: {r['avg_latency_s']} s")
    for e in r["errors"][:3]:
        print(f"  error: {e}")
    print("usable as a verifier" if r["usable"] else "NOT ready: fix the failures above before using it as the verifier")
    return 0 if r["usable"] else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
