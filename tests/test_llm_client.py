"""LLM client: local OpenAI-compatible backend against a real in-process HTTP server (offline)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import llm_client as lc


class Server:
    def __init__(self):
        self.requests, self.queue = [], []        # queue items: (status, body_dict_or_str)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "body": json.loads(self.rfile.read(n) or b"{}")})
                status, body = outer.queue.pop(0) if outer.queue else (200, reply("ok"))
                data = body if isinstance(body, str) else json.dumps(body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data.encode("utf-8"))

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def reply(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


@pytest.fixture
def srv(monkeypatch):
    s = Server()
    monkeypatch.setattr(lc, "_sleep", lambda *_: None)
    yield s
    s.httpd.shutdown()


def gen(srv, prompt="hi", **kw):
    return lc.generate(prompt, model="m", backend="openai", base_url=srv.url, timeout=5, **kw)


# ---------- request and reply ----------

def test_sends_system_and_user_messages_and_returns_the_text(srv):
    srv.queue = [(200, reply("  hello  "))]
    assert gen(srv, "question", system_instruction="be brief", temperature=0.2, max_output_tokens=50, api_key="k") == "hello"
    r = srv.requests[0]
    assert r["path"] == "/v1/chat/completions" and r["auth"] == "Bearer k"
    assert r["body"]["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "question"}]
    assert r["body"]["temperature"] == 0.2 and r["body"]["max_tokens"] == 50 and r["body"]["model"] == "m"
    assert "response_format" not in r["body"]


def test_no_system_message_and_no_auth_header_when_not_given(srv):
    gen(srv, "q")
    assert srv.requests[0]["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert srv.requests[0]["auth"] is None


def test_telugu_survives_the_round_trip(srv):
    srv.queue = [(200, reply("కర్ణుడు"))]
    assert gen(srv, "కర్ణుడు") == "కర్ణుడు"
    assert srv.requests[0]["body"]["messages"][0]["content"] == "కర్ణుడు"


def test_think_blocks_are_removed(srv):
    srv.queue = [(200, reply("<think>let me see...\nmore</think>\nThe answer."))]
    assert gen(srv) == "The answer."


# ---------- JSON output ----------

@pytest.mark.parametrize("raw", [
    '{"verdict": "supported"}',
    '```json\n{"verdict": "supported"}\n```',
    'Sure! Here is the result:\n{"verdict": "supported"}\nHope that helps.',
    '<think>hmm {not json}</think>{"verdict": "supported"}',
])
def test_json_output_is_cleaned_to_valid_json(srv, raw):
    srv.queue = [(200, reply(raw))]
    out = gen(srv, json_output=True)
    assert json.loads(out) == {"verdict": "supported"}
    assert srv.requests[0]["body"]["response_format"] == {"type": "json_object"}


def test_json_output_with_no_json_raises(srv):
    srv.queue = [(200, reply("I cannot answer that."))]
    with pytest.raises(lc.LLMError, match="no valid JSON"):
        gen(srv, json_output=True)


def test_extract_json_arrays_and_unicode():
    assert json.loads(lc.extract_json('text [1, 2, {"a": "కర్ణుడు"}] tail'))[2]["a"] == "కర్ణుడు"
    with pytest.raises(lc.LLMError):
        lc.extract_json("{broken")


def test_server_without_json_mode_is_retried_without_response_format(srv):
    srv.queue = [(400, {"error": {"message": "Unknown parameter: response_format"}}), (200, reply('{"a": 1}'))]
    assert json.loads(gen(srv, json_output=True)) == {"a": 1}
    assert "response_format" in srv.requests[0]["body"] and "response_format" not in srv.requests[1]["body"]


# ---------- failures ----------

def test_other_400_is_not_retried(srv):
    srv.queue = [(400, {"error": {"message": "context length exceeded"}})]
    with pytest.raises(lc.LLMError, match="refused"):
        gen(srv)
    assert len(srv.requests) == 1


def test_transient_server_error_is_retried_then_succeeds(srv):
    srv.queue = [(503, "busy"), (500, "oops"), (200, reply("finally"))]
    assert gen(srv) == "finally" and len(srv.requests) == 3


def test_persistent_server_error_gives_up_with_a_clear_message(srv):
    srv.queue = [(503, "busy")] * 5
    with pytest.raises(lc.LLMError, match="unavailable after 3 attempts"):
        gen(srv)
    assert len(srv.requests) == 3


def test_server_not_running(monkeypatch):
    monkeypatch.setattr(lc, "_sleep", lambda *_: None)
    with pytest.raises(lc.LLMError, match="Is it running"):
        lc.generate("hi", model="m", backend="openai", base_url="http://127.0.0.1:9/v1", timeout=1)


def test_empty_answer_and_odd_shapes_raise(srv):
    for body in (reply(""), {"choices": []}, {"nope": 1}, "not json at all"):
        srv.queue = [(200, body)]
        with pytest.raises(lc.LLMError):
            gen(srv)


def test_missing_model_name_is_explained():
    with pytest.raises(lc.LLMError, match="LLM_MODEL"):
        lc.generate("hi", model="", backend="openai", base_url="http://x/v1")


def test_unknown_backend_and_missing_gemini_sdk(monkeypatch):
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        lc.generate("hi", model="m", backend="nope")
    monkeypatch.setattr(lc, "BACKEND", None)
    with pytest.raises(ImportError, match="LLM_BACKEND=openai"):
        lc.generate("hi", model="m", backend="gemini", api_key="k")


# ---------- settings from config ----------

@pytest.fixture
def cfg(monkeypatch):
    import config
    for k, v in {"LLM_BACKEND": "openai", "LLM_BASE_URL": "http://h:1/v1", "LLM_MODEL": "main-model",
                 "LLM_VERIFIER_MODEL": "", "LLM_API_KEY": None, "LLM_TIMEOUT_S": 7,
                 "GEMINI_MODEL_NAME": "gem", "VERIFIER_MODEL": "gem-v", "GEMINI_API_KEY": "gk"}.items():
        monkeypatch.setattr(config, k, v, raising=False)
    return config


def test_settings_pick_model_by_role_and_backend(cfg, monkeypatch):
    assert lc.settings("answer")["model"] == "main-model" and lc.settings("verifier")["model"] == "main-model"
    monkeypatch.setattr(cfg, "LLM_VERIFIER_MODEL", "small-verifier")
    assert lc.settings("verifier")["model"] == "small-verifier" and lc.settings("answer")["model"] == "main-model"
    assert lc.settings()["base_url"] == "http://h:1/v1" and lc.settings()["timeout"] == 7.0
    monkeypatch.setattr(cfg, "LLM_BACKEND", "gemini")
    assert lc.settings("verifier") == {"backend": "gemini", "model": "gem-v", "api_key": "gk"}
    assert lc.settings("answer")["model"] == "gem"


def test_settings_errors(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "LLM_MODEL", "")
    with pytest.raises(lc.LLMError, match="LLM_MODEL"):
        lc.settings()
    monkeypatch.setattr(cfg, "LLM_BACKEND", "banana")
    with pytest.raises(ValueError):
        lc.settings()


def test_generate_default_uses_the_configured_local_server(cfg, srv, monkeypatch):
    monkeypatch.setattr(cfg, "LLM_BASE_URL", srv.url)
    srv.queue = [(200, reply("from the local model"))]
    assert lc.generate_default("q") == "from the local model"
    assert srv.requests[0]["body"]["model"] == "main-model"


# ---------- model check ----------

def _fake_model(correct_verdicts=6, telugu=True, json_ok=True):
    state = {"i": 0}

    def gen_(prompt, **kw):
        if prompt.startswith("Reply with the single word"):
            return "ready"
        if prompt.startswith("Repeat exactly"):
            return "కర్ణుడు" if telugu else "?"
        if kw.get("json_output") and "system_instruction" not in kw:
            return '{"ok": true}' if json_ok else "yes"
        expected = lc.VERIFIER_PROBES[state["i"]][2]
        wrong = "unsupported" if expected == "supported" else "supported"
        verdict = expected if state["i"] < correct_verdicts else wrong
        state["i"] += 1
        return json.dumps({"verdict": verdict})
    return gen_


def test_check_passes_a_good_model_and_rejects_bad_ones():
    ok = lc.run_check(gen=_fake_model())
    assert ok["usable"] and ok["verifier_correct"] == 6 and not ok["errors"]
    assert lc.run_check(gen=_fake_model(correct_verdicts=5))["usable"]          # one miss is tolerated
    assert not lc.run_check(gen=_fake_model(correct_verdicts=4))["usable"]
    assert not lc.run_check(gen=_fake_model(telugu=False))["usable"]
    assert not lc.run_check(gen=_fake_model(json_ok=False))["usable"]


def test_check_reports_a_dead_server_instead_of_crashing():
    calls = []

    def dead(prompt, **kw):
        calls.append(prompt)
        raise lc.LLMError("model server unavailable")
    r = lc.run_check(gen=dead)
    assert not r["usable"] and len(r["errors"]) == 1 and r["verifier_correct"] == 0 and r["avg_latency_s"] is None
    assert len(calls) == 1                  # stopped after the first failure, did not wait through nine retries
