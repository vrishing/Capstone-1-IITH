import json

import answer
import verify
import verify_bench

PASSAGE = ("మాతలి అన్న సారథు ఉన్నాడు ఆయన రథం పట్టుకొస్తాడు నేను వెళ్ళిన వెంటనే రథం మీద రా అని చెప్పడు "
           "అని చెప్పి ఇంద్రుడు అంతర్ధానం అయ్యాడు అర్జునుడు ఆనందం చూడాలంటే త్రినయ")
HITS = [{"span_id": "v1_000000010", "video_id": "v1", "start": 10.0, "end": 70.0, "url": "u1",
         "text": "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు అని చెప్పారు పెద్దలు"},
        {"span_id": "v2_000000100", "video_id": "v2", "start": 100.0, "end": 150.0, "url": "u2", "text": PASSAGE}]


def test_locate_tolerates_an_inflected_first_word_and_rejects_invention():
    # the exact failure seen on m24: the model wrote అర్జునుడి where the passage has అర్జునుడు
    assert verify.locate("అర్జునుడి ఆనందం చూడాలంటే త్రినయ", PASSAGE)
    assert verify.locate("మాతలి అన్న సారథి ఉన్నాడు", PASSAGE)              # one vowel sign changed
    assert verify.locate("భీముడు కర్ణుడిని మొదటి రోజున చంపాడు", PASSAGE) is None
    assert verify.locate("మాతలి", PASSAGE) is None                          # too short to count


def test_window_is_real_text_around_the_match():
    span = verify.locate("ఇంద్రుడు అంతర్ధానం అయ్యాడు", PASSAGE)
    w = verify.window(PASSAGE, span, pad=2)
    assert "ఇంద్రుడు అంతర్ధానం అయ్యాడు" in w and w.startswith("... ") and w.endswith(" ...")


class Scripted:
    """locate replies, then judge replies, in order; records prompts per system."""
    def __init__(self, locate, judge):
        self.loc, self.jud, self.prompts = list(locate), list(judge), []

    def __call__(self, prompt, role="verifier", system=None, json_output=True, max_output_tokens=1000):
        self.prompts.append((system, prompt))
        if system == verify.LOCATE_SYSTEM:
            return json.dumps({"quotes": self.loc.pop(0)})
        return json.dumps({"items": self.jud.pop(0)})


def test_locate_judge_only_judges_located_sentences_on_a_window():
    sents = [{"text": "Dhrishtadyumna killed Drona.", "cites": [1]},
             {"text": "Bhima killed Karna on the first day.", "cites": [2]}]
    g = Scripted([[{"i": 0, "passage": 1, "quote": "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు"},
                   {"i": 1, "passage": 2, "quote": "భీముడు కర్ణుడిని మొదటి రోజున చంపాడు"}]],     # invented
                 [[{"i": 0, "says": "Dhrishtadyumna killed Drona.", "verdict": "supported"}]])
    v, dbg = verify.verify("Who killed Drona?", sents, HITS, g)
    assert v == ["supported", "unsupported"]
    assert [f["i"] for f in dbg["quote_failures"]] == [1]
    judge_prompt = g.prompts[1][1]
    assert "Question: Who killed Drona?" in judge_prompt and "Bhima" not in judge_prompt
    assert dbg["says"][0].startswith("Dhrishtadyumna")


def test_judge_can_reject_a_located_but_wrong_claim():
    sents = [{"text": "Arjuna's charioteer was Matali.", "cites": [2]}]
    g = Scripted([[{"i": 0, "passage": 2, "quote": "మాతలి అన్న సారథు ఉన్నాడు"}]],
                 [[{"i": 0, "says": "Indra says his charioteer Matali will bring a chariot.", "verdict": "partial"}]])
    assert verify.verify("Who was Arjuna's charioteer?", sents, HITS, g)[0] == ["partial"]
    g2 = Scripted([[{"i": 0, "passage": 2, "quote": "మాతలి అన్న సారథు ఉన్నాడు"}]],
                  [[{"i": 0, "says": "...", "verdict": "unsupported"}]])
    assert verify.verify("q", sents, HITS, g2)[0] == ["unsupported"]


def test_no_quote_at_all_means_no_judge_call():
    g = Scripted([[]], [])
    v, _ = verify.verify("q", [{"text": "x y z", "cites": [1]}], HITS, g)
    assert v == ["unsupported"] and len(g.prompts) == 1


def test_judge_missing_an_item_raises_so_the_answer_abstains():
    g = Scripted([[{"i": 0, "passage": 1, "quote": "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు"}]], [[]])
    try:
        verify.verify("q", [{"text": "x", "cites": [1]}], HITS, g)
        assert False
    except ValueError:
        pass


def test_answer_question_with_locate_judge_and_partial_answer_note():
    gen_reply = json.dumps({"sufficient": True, "not_covered": ["what Karna says"],
                            "sentences": [{"text": "Dhrishtadyumna killed Drona.", "cites": [1]}]})

    class G(Scripted):
        def __call__(self, prompt, role="answer", system=None, json_output=True, max_output_tokens=1000):
            if role == "answer":
                return gen_reply
            return super().__call__(prompt, role, system, json_output, max_output_tokens)
    g = G([[{"i": 0, "passage": 1, "quote": "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు"}]],
          [[{"i": 0, "says": "Dhrishtadyumna killed Drona.", "verdict": "supported"}]])
    r = answer.answer_question("Who killed Drona?", lambda q: {"hits": HITS, "query_entities": []}, g,
                               {"v1": "Part 1"}, verify_mode="locate_judge")
    assert r["outcome"] == "answered" and r["not_covered"] == ["what Karna says"]
    assert "Not covered by the passages found: what Karna says" in answer.render(r)


def test_bench_export_and_score(tmp_path):
    tr = {"question": "Who killed Drona?", "evidence": HITS,
          "attempts": [{"generation": json.dumps({"sufficient": True, "sentences": [
              {"text": "Dhrishtadyumna killed Drona.", "cites": [1]},
              {"text": "Bhima killed Drona.", "cites": [1]}]}), "verdicts": ["supported", "supported"]}]}
    d = tmp_path / "traces"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(tr), encoding="utf-8")
    (d / "_llm_cache.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "items.jsonl"
    assert verify_bench.export(str(d), str(out)) == 2
    assert verify_bench.export(str(d), str(out)) == 2              # deterministic, no duplicates
    items = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    items[0]["label"], items[1]["label"] = "y", "n"

    def gen(prompt, role="verifier", system=None, json_output=True, max_output_tokens=1000):
        if system == verify.LOCATE_SYSTEM:
            return json.dumps({"quotes": [{"i": 0, "passage": 1, "quote": "ద్రోణుడిని ధృష్టద్యుమ్నుడు చంపాడు"}]})
        return json.dumps({"items": [{"i": 0, "says": "", "verdict": "supported"}]})   # accepts everything
    s = verify_bench.score(verify_bench.run(items, gen, "locate_judge"))
    assert (s["false_accept"], s["false_reject"]) == (1.0, 0.0)