import json

import answer

HITS = [{"span_id": "v1_000000010", "video_id": "v1", "start": 10.0, "end": 70.0, "text": "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు",
         "url": "https://youtu.be/v1?t=10"},
        {"span_id": "v2_000000100", "video_id": "v2", "start": 100.0, "end": 150.0, "text": "other text",
         "url": "https://youtu.be/v2?t=100"}]
TITLES = {"v1": "Part 1", "v2": "Part 2"}


def ret(q):
    return {"hits": HITS, "query_entities": []}


class Gen:
    """Scripted LLM: answers are consumed in order per role."""
    def __init__(self, answers=None, verdicts=None, vfail=False):
        self.a, self.v, self.vfail, self.calls = list(answers or []), list(verdicts or []), vfail, []

    def __call__(self, prompt, role="answer", system=None, json_output=True, max_output_tokens=1000):
        self.calls.append(role)
        if role == "verifier":
            if self.vfail:
                raise RuntimeError("quota")
            return json.dumps({"verdicts": [{"i": i, "verdict": x} for i, x in enumerate(self.v.pop(0))]})
        return self.a.pop(0)


def gen_json(*sents, sufficient=True):
    return json.dumps({"sufficient": sufficient, "sentences": [{"text": t, "cites": c} for t, c in sents]})


def run(g, **kw):
    return answer.answer_question("Who killed Drona?", ret, g, TITLES, **kw)


def test_supported_answer_has_citations():
    g = Gen([gen_json(("Dhrishtadyumna killed Drona.", [1]))], [["supported"]])
    r = run(g)
    assert r["outcome"] == "answered" and r["attempts"] == 1
    assert "[Video Part 1, 00:00:10–00:01:10]" in r["answer"]
    assert r["sentences"][0]["cites"][0]["span_id"] == "v1_000000010"


def test_partial_gives_caution():
    r = run(Gen([gen_json(("A.", [1]), ("B.", [2]))], [["supported", "partial"]]))
    assert r["outcome"] == "answered_with_caution" and r["warnings"]


def test_unsupported_regenerates_then_succeeds():
    g = Gen([gen_json(("Wrong.", [2])), gen_json(("Right.", [1]))], [["unsupported"], ["supported"]])
    r = run(g)
    assert r["outcome"] == "answered" and r["attempts"] == 2 and "Right." in r["answer"]


def test_uncited_or_bad_cite_counts_as_unsupported_without_asking_verifier():
    g = Gen([gen_json(("No cite.", [])), gen_json(("Bad cite.", [9])), gen_json(("Still none.", []))])
    r = run(g)
    assert r["outcome"] == "abstained" and r["answer"] == answer.ABSTAIN_TEXT
    assert "verifier" not in g.calls


def test_strict_abstains_after_retries_nonstrict_keeps_verified():
    answers = lambda: [gen_json(("Good.", [1]), ("Bad.", [2]))] * 3
    verd = lambda: [["supported", "unsupported"]] * 3
    r = run(Gen(answers(), verd()))
    assert r["outcome"] == "abstained" and r["attempts"] == 3
    r2 = run(Gen(answers(), verd()), strict=False)
    assert r2["outcome"] == "answered" and "Bad." not in r2["answer"] and r2["warnings"]


def test_verifier_error_fails_closed_and_is_marked_error():
    r = run(Gen([gen_json(("X.", [1]))], vfail=True))
    assert r["outcome"] == "abstained" and r["error"] and "verifier failed" in r["reason"]


def test_insufficient_abstains_without_verifying():
    g = Gen([gen_json(sufficient=False)])
    r = run(g)
    assert r["outcome"] == "abstained" and not r["error"] and g.calls == ["answer"]


def test_no_hits_abstains():
    r = answer.answer_question("q", lambda q: {"hits": []}, Gen())
    assert r["outcome"] == "abstained" and r["reason"] == "nothing retrieved"


def test_malformed_generation_is_retried():
    g = Gen(["not json at all", gen_json(("OK.", [1]))], [["supported"]])
    r = run(g)
    assert r["outcome"] == "answered" and r["attempts"] == 2


def test_cache_roundtrip_skips_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(answer.config, "ANSWER_CACHE_DIR", str(tmp_path), raising=False)
    k = answer.cache_key("Who?", "PL", 8, "m", "10")
    assert answer.read_cache(k) is None
    answer.write_cache(k, {"error": True, "answer": "x"})
    assert answer.read_cache(k) is None
    answer.write_cache(k, {"error": False, "answer": "x"})
    assert answer.read_cache(k)["answer"] == "x"
    assert answer.cache_key(" who? ", "PL", 8, "m", "10") == k            # whitespace and case ignored
    assert answer.cache_key("Who?", "PL", 8, "m", "11") != k              # index changed


def test_language_instruction():
    assert answer.question_language("అర్జునుడు ఎవరు") == "the same language as the question"
    assert answer.question_language("Who is Arjuna") == "English"


def test_parse_verdicts_accepts_common_shapes_and_rejects_the_rest():
    pv = answer.parse_verdicts
    assert pv('{"verdicts": [{"i": 0, "verdict": "supported"}, {"i": 1, "verdict": "Partial"}]}', 2) == ["supported", "partial"]
    assert pv('[{"i": 0, "verdict": "unsupported"}]', 1) == ["unsupported"]          # bare list of objects
    assert pv('["supported", "unsupported"]', 2) == ["supported", "unsupported"]     # plain strings, in order
    for bad in ('{"verdicts": []}', '"yes"', '[{"i": 0, "verdict": "maybe"}]', '{"verdicts": ["supported"]}'):
        try:
            pv(bad, 2)
            assert False, bad
        except ValueError:
            pass


class QGen(Gen):
    """Verifier replies carry a quote per statement."""
    def __init__(self, answers, rows):
        super().__init__(answers)
        self.rows = list(rows)

    def __call__(self, prompt, role="answer", system=None, json_output=True, max_output_tokens=1000):
        if role == "verifier":
            self.calls.append(role)
            return json.dumps({"verdicts": [{"i": i, "verdict": v, "quote": q}
                                            for i, (v, q) in enumerate(self.rows.pop(0))]})
        return super().__call__(prompt, role, system, json_output, max_output_tokens)


def test_quote_in_passage_tolerates_a_slip_but_not_invention():
    p = "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు అని చెప్పారు పెద్దలు"
    assert answer.quote_in_passage("ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు", p)
    assert answer.quote_in_passage("ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపేడు అని చెప్పారు", p)     # one word wrong
    assert not answer.quote_in_passage("భీముడు కర్ణుడిని మొదటి రోజున చంపాడు", p)
    assert not answer.quote_in_passage("చంపాడు", p)                                          # too short to count


def test_verified_answer_needs_a_quote_found_in_its_passage():
    good = [[("supported", "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు")]]
    r = run(QGen([gen_json(("Dhrishtadyumna killed Drona.", [1]))], good), require_quote=True)
    assert r["outcome"] == "answered"
    bad = [[("supported", "భీముడు కర్ణుడిని మొదటి రోజున చంపాడు")]] * 3
    g = QGen([gen_json(("Dhrishtadyumna killed Drona.", [1]))] * 3, bad)
    r = run(g, require_quote=True)
    assert r["outcome"] == "abstained"                       # a quote not in the passage fails closed


def test_missing_quote_fails_only_when_required():
    rows = [[("supported", "")]]
    r = run(QGen([gen_json(("Dhrishtadyumna killed Drona.", [1]))], rows), require_quote=False)
    assert r["outcome"] == "answered"
    rows3 = [[("supported", "")]] * 3
    r = run(QGen([gen_json(("Dhrishtadyumna killed Drona.", [1]))] * 3, rows3), require_quote=True)
    assert r["outcome"] == "abstained"