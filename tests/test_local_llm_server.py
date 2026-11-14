import json
import threading
import urllib.error
import urllib.request

import llm_client
import local_llm_server as srv


def start(generate):
    s = srv.make_server(generate, "tiny", "127.0.0.1", 0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}/v1"


def post(url, body):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def test_chat_roundtrip_models_and_unknown_path():
    seen = {}

    def gen(messages, max_tokens, temperature):
        seen.update(m=messages, t=max_tokens, temp=temperature)
        return "<think>x</think> నమస్కారం"
    s, url = start(gen)
    try:
        r = post(url, {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50})
        assert r["choices"][0]["message"]["content"].endswith("నమస్కారం")
        assert seen["t"] == 50 and seen["temp"] == 0.0 and seen["m"][0]["content"] == "hi"
        assert json.loads(urllib.request.urlopen(url + "/models").read())["data"][0]["id"] == "tiny"
        try:
            urllib.request.urlopen(url.replace("/v1", "") + "/nothing")
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        s.shutdown()


def test_bad_request_is_400_and_generation_error_is_500():
    def boom(m, t, temp):
        raise RuntimeError("out of memory")
    s, url = start(boom)
    try:
        for body, code in (({"model": "tiny"}, 400), ({"messages": [{"role": "user", "content": "x"}]}, 500)):
            try:
                post(url, body)
                assert False
            except urllib.error.HTTPError as e:
                assert e.code == code
    finally:
        s.shutdown()


def test_llm_client_talks_to_it_and_strips_thinking():
    s, url = start(lambda m, t, temp: '<think>hmm</think>```json\n{"ok": true}\n```')
    try:
        out = llm_client.generate("say json", backend="openai", model="tiny", base_url=url, json_output=True)
        assert json.loads(out) == {"ok": True}
    finally:
        s.shutdown()


class ResourceExhausted(Exception):
    pass


def test_rate_limit_retry_waits_the_time_the_server_asks(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_client, "_sleep", waits.append)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ResourceExhausted("429 quota PerMinute ... Please retry in 32.1s.")
        return "ok"
    assert llm_client.with_rate_limit_retry(flaky) == "ok"
    assert len(calls) == 3 and waits == [33.1, 33.1]


def test_daily_quota_and_other_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(llm_client, "_sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    for exc in (ResourceExhausted("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier"), ValueError("bad")):
        def boom(e=exc):
            raise e
        try:
            llm_client.with_rate_limit_retry(boom)
            assert False
        except type(exc):
            pass


def test_rate_limit_retry_gives_up_after_max_waits(monkeypatch):
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    n = []

    def always():
        n.append(1)
        raise ResourceExhausted("429 PerMinute retry in 1s")
    try:
        llm_client.with_rate_limit_retry(always, max_waits=2)
        assert False
    except ResourceExhausted:
        assert len(n) == 3


def test_thinking_flag_reaches_generate_and_client_sends_it():
    seen = {}

    def gen(m, t, temp, thinking=False):
        seen["thinking"], seen["t"] = thinking, t
        return '{"ok": true}'
    s, url = start(gen)
    try:
        post(url, {"model": "tiny", "messages": [{"role": "user", "content": "x"}], "max_tokens": 50})
        assert seen["thinking"] is False
        out = llm_client.generate("x", backend="openai", model="tiny", base_url=url, json_output=True,
                                  max_output_tokens=100, thinking=True)
        assert json.loads(out) == {"ok": True} and seen["thinking"] is True and seen["t"] >= 8000
    finally:
        s.shutdown()


def test_cut_off_reasoning_is_not_mistaken_for_an_answer():
    assert llm_client.clean_text('<think>maybe {"verdict": "supported"} ...') == ""