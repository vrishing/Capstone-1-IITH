"""
A small OpenAI-compatible chat server for a local Hugging Face model. No packages beyond torch and
transformers (already in the venv), no pip install, nothing opened to other machines.

    CUDA_VISIBLE_DEVICES=7 python local_llm_server.py --model Qwen/Qwen3-8B --port 8011

then in .env:
    LLM_BACKEND=openai
    LLM_BASE_URL=http://127.0.0.1:8011/v1
    LLM_MODEL=Qwen/Qwen3-8B
and check with:  python llm_client.py --check

Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health. One request runs at a time
(a lock): enough for our batch jobs, and it keeps GPU memory predictable on a shared machine.
Binds to 127.0.0.1 by default. Thinking mode of Qwen3-style models is switched off (short, direct
replies; llm_client strips any <think> block anyway). response_format is accepted and ignored:
llm_client extracts the JSON from the text.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_server(generate, model_name: str, host: str = "127.0.0.1", port: int = 8011) -> ThreadingHTTPServer:
    """generate(messages, max_tokens, temperature) -> str. Returns the (not yet started) server."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            elif self.path.rstrip("/") in ("/health", ""):
                self._send(200, {"status": "ok", "model": model_name})
            else:
                self._send(404, {"detail": "Not Found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"detail": "Not Found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                msgs = req["messages"]
                if not isinstance(msgs, list) or not msgs:
                    raise ValueError("messages must be a non-empty list")
                max_tokens = int(req.get("max_tokens") or 1024)
                temperature = float(req.get("temperature") or 0.0)
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": {"message": f"bad request: {e}"}})
            try:
                with lock:
                    text = generate(msgs, max_tokens, temperature)
            except Exception as e:                       # out of memory, template error, ...
                return self._send(500, {"error": {"message": f"{type(e).__name__}: {str(e)[:300]}"}})
            self._send(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
                "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    return ThreadingHTTPServer((host, port), Handler)


def load_generate(model_id: str, dtype: str = "bfloat16", repetition_penalty: float = 1.1):
    """Load the model on the visible GPU and return generate(messages, max_tokens, temperature)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def generate(messages, max_tokens, temperature):
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:                                 # template without that option
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        kw = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
        kw["repetition_penalty"] = repetition_penalty      # small models loop in Telugu without it
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=min(max_tokens, 6000), **kw)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--name", help="name to serve it under (default: the model id)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--repetition-penalty", type=float, default=1.1, help="1.0 = off (default 1.1)")
    a = ap.parse_args(argv)
    print(f"loading {a.model} ...", flush=True)
    gen = load_generate(a.model, a.dtype, a.repetition_penalty)
    srv = make_server(gen, a.name or a.model, a.host, a.port)
    print(f"serving {a.name or a.model} at http://{a.host}:{a.port}/v1  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())