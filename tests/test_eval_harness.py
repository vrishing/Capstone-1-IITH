"""Evaluation harness: metrics checked against hand-computed values (offline)."""
import json

import pytest

import eval_harness as eh

V1, V2 = "aaaaaaaaaaa", "bbbbbbbbbbb"


def span(v, s, e):
    return {"video_id": v, "start": s, "end": e, "span_id": f"{v}_{s}"}


def q(qid, gold=None, answerable=True, category=None, must=None):
    r = {"id": qid, "question": f"question {qid}", "answerable": answerable, "category": category,
         "gold": [{"video_id": v, "start": s, "end": e} for v, s, e in (gold or [])],
         "must_include": must or []}
    return r


@pytest.fixture
def qs(tmp_path):
    rows = [
        q("q1", [(V1, 100, 160)], category="factoid", must=["Drona"]),
        q("q2", [(V1, 300, 360)], category="factoid"),
        q("q3", [(V2, 50, 110), (V2, 500, 560)], category="multi"),
        q("q4", [(V2, 900, 960)], category="factoid"),
        q("q5", answerable=False),
        q("q6", answerable=False),
    ]
    p = tmp_path / "q.jsonl"
    eh.write_jsonl(p, rows)
    return eh.load_questions(p)


def run(qid, retrieved=(), answer="", abstained=False, citations=()):
    return {"id": qid, "retrieved": list(retrieved), "answer": answer, "abstained": abstained,
            "citations": list(citations)}


@pytest.fixture
def runs():
    return {r["id"]: r for r in [
        # q1: hit at rank 2, cited correctly, mentions Drona
        run("q1", [span(V2, 0, 60), span(V1, 90, 150)], "Drona taught him.", citations=[span(V1, 90, 150)]),
        # q2: right video but 200 s away: miss
        run("q2", [span(V1, 500, 560)], "x", citations=[span(V1, 500, 560)]),
        # q3: first gold at rank 1, second gold at rank 3
        run("q3", [span(V2, 40, 100), span(V1, 0, 60), span(V2, 495, 555)], "ok", citations=[span(V2, 40, 100), span(V1, 0, 60)]),
        # q4: nothing retrieved, abstained
        run("q4", [], abstained=True),
        # unanswerable: q5 abstained correctly, q6 answered
        run("q5", [span(V1, 0, 60)], abstained=True),
        run("q6", [span(V1, 0, 60)], "made up", citations=[span(V1, 0, 60)]),
    ]}


# ---------- metrics ----------

def test_retrieval_metrics_match_hand_computation(qs, runs):
    m = eh.score(qs, runs, ks=(1, 3, 5), primary_k=5)["metrics"]
    # first-hit ranks: q1=2, q2=none, q3=1, q4=none
    assert m["recall@1"] == pytest.approx(1 / 4)
    assert m["recall@3"] == pytest.approx(2 / 4)
    assert m["recall@5"] == pytest.approx(2 / 4)
    assert m["mrr"] == pytest.approx((1 / 2 + 0 + 1 + 0) / 4)
    assert m["timestamp_hit@1"] == pytest.approx(1 / 4)
    # q3 has two gold ranges, hit at ranks 1 and 3 -> evidence recall@1 is 1/2, @3 is 1
    r1 = eh.score(qs, runs, ks=(1,), primary_k=1)["metrics"]["evidence_recall@1"]
    assert r1 == pytest.approx((0 + 0 + 0.5 + 0) / 4)
    r3 = eh.score(qs, runs, ks=(3,), primary_k=3)["metrics"]["evidence_recall@3"]
    assert r3 == pytest.approx((1 + 0 + 1 + 0) / 4)


def test_answer_metrics(qs, runs):
    rep = eh.score(qs, runs, primary_k=5)
    m = rep["metrics"]
    assert m["citation_hit"] == pytest.approx(2 / 4)            # q1 and q3 cite a gold span
    assert m["citation_precision"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)   # q1, q2, q3 cite
    assert m["end_to_end"] == pytest.approx(2 / 4)              # q1 has Drona, q3 has no must_include
    assert m["false_abstain"] == pytest.approx(1 / 4)           # q4
    assert m["abstain_rate"] == pytest.approx(1 / 2)            # q5 yes, q6 no
    assert rep["failed_ids"] == ["q2", "q4"]


def test_end_to_end_needs_the_required_terms(qs, runs):
    runs["q1"]["answer"] = "Someone else taught him."
    assert eh.score(qs, runs)["metrics"]["end_to_end"] == pytest.approx(1 / 4)


def test_abstained_answer_cannot_get_citation_credit(qs, runs):
    runs["q1"]["abstained"] = True
    m = eh.score(qs, runs)["metrics"]
    assert m["citation_hit"] == pytest.approx(1 / 4) and m["false_abstain"] == pytest.approx(2 / 4)


def test_overlap_rule_is_half_the_gold_or_min_overlap():
    g = {"video_id": V1, "start": 100.0, "end": 160.0}
    assert eh._hit(span(V1, 150, 400), g, 10.0)         # exactly 10 s of overlap
    assert not eh._hit(span(V1, 152, 400), g, 10.0)     # 8 s
    assert eh._hit(span(V1, 0, 400), g, 10.0)           # covers it
    assert not eh._hit(span(V2, 100, 160), g, 10.0)     # wrong video
    short = {"video_id": V1, "start": 100.0, "end": 106.0}
    assert eh._hit(span(V1, 103, 200), short, 10.0)     # half of a 6 s range is enough
    assert not eh._hit(span(V1, 106, 200), short, 10.0)  # touching only


def test_missing_and_errored_runs_count_as_failures(qs, runs):
    del runs["q2"]
    runs["q3"]["error"] = "ValueError: boom"
    rep = eh.score(qs, runs)
    assert rep["missing_runs"] == ["q2"] and rep["errored"] == ["q3"]
    assert "WARNING" in eh.format_score(rep)


def test_per_category_breakdown(qs, runs):
    by = eh.score(qs, runs, primary_k=5)["by_category"]
    assert by["factoid"]["n"] == 3 and by["factoid"]["recall@5"] == pytest.approx(1 / 3)
    assert by["multi"]["recall@5"] == 1.0
    assert by["unanswerable"]["abstain_rate"] == pytest.approx(1 / 2)


def test_bootstrap_ci_is_deterministic_and_contains_the_mean():
    v = [1.0] * 6 + [0.0] * 4
    a, b = eh.boot_ci(v), eh.boot_ci(v)
    assert a == b and a[0] <= 0.6 <= a[1] and a[0] < a[1]
    assert eh.boot_ci([1.0]) is None


def test_time_parsing():
    assert eh.parse_time("1:02:03") == 3723 and eh.parse_time("12:30") == 750 and eh.parse_time(90) == 90
    assert eh.fmt_time(3723) == "01:02:03"
    with pytest.raises(eh.EvalError):
        eh.parse_time("soon")


# ---------- question file checks ----------

@pytest.mark.parametrize("row,msg", [
    ({"id": "a", "question": "q", "answerable": True}, "at least one gold"),
    ({"id": "a", "question": "q", "answerable": False, "gold": [{"video_id": V1, "start": 0, "end": 5}]}, "must not have gold"),
    ({"id": "a", "question": "q", "gold": [{"video_id": V1, "start": 9, "end": 5}]}, "ends before"),
    ({"id": "a", "question": " ", "gold": [{"video_id": V1, "start": 0, "end": 5}]}, "empty question"),
    ({"question": "q"}, "missing id"),
])
def test_bad_question_rows_are_rejected_with_the_reason(tmp_path, row, msg):
    p = tmp_path / "q.jsonl"
    eh.write_jsonl(p, [row])
    with pytest.raises(eh.EvalError, match=msg):
        eh.load_questions(p)


def test_duplicate_ids_rejected_and_examples_ignored(tmp_path):
    p = tmp_path / "q.jsonl"
    good = {"id": "a", "question": "q", "gold": [{"video_id": V1, "start": 0, "end": 5}]}
    eh.write_jsonl(p, [good, dict(good)])
    with pytest.raises(eh.EvalError, match="duplicate"):
        eh.load_questions(p)
    eh.write_jsonl(p, [dict(good, example=True), {"id": "b", "question": "q", "answerable": False}])
    assert [x["id"] for x in eh.load_questions(p)] == ["b"]


def test_leakage_drops_questions_with_evidence_in_the_sample(qs):
    keep, dropped = eh.drop_leaked(qs, {V1})
    assert dropped == ["q1", "q2"] and [x["id"] for x in keep] == ["q3", "q4", "q5", "q6"]


def test_collect_run_records_errors_and_keeps_going(tmp_path, qs):
    def fn(question):
        if question["id"] == "q2":
            raise RuntimeError("boom")
        return {"retrieved": [], "answer": "a", "abstained": False, "citations": []}
    rows = eh.collect_run(qs, fn, tmp_path / "run.jsonl")
    assert len(rows) == 6 and rows[1]["error"].startswith("RuntimeError")
    assert set(eh.load_runs(tmp_path / "run.jsonl")) == {x["id"] for x in qs}


# ---------- error rates ----------

def test_edit_distance_and_known_rates():
    assert eh.edit_distance(list("kitten"), list("sitting")) == 3
    r = eh.error_rates("the cat sat", "the cat sit")
    assert r["wer"] == pytest.approx(1 / 3) and r["cer"] == pytest.approx(1 / 11)
    assert eh.error_rates("same text", "same text")["cer"] == 0.0


def test_normalisation_ignores_punctuation_case_and_joiners():
    assert eh.error_rates("Hello, World!", "hello world")["cer"] == 0.0
    assert eh.error_rates("क्षत्रिय।", "क्षत्रिय")["cer"] == 0.0


def test_aksharas_group_conjuncts_and_vowel_signs():
    assert eh.aksharas("కృష్ణ") == ["కృ", "ష్ణ"]
    assert eh.aksharas("క్షత్రియ") == ["క్ష", "త్రి", "య"]
    assert eh.aksharas("కర్ణుడు") == ["క", "ర్ణు", "డు"]


def test_one_wrong_vowel_sign_is_one_akshara_error():
    r = eh.error_rates("కర్ణుడు", "కర్ణడు")      # ు dropped from the middle akshara
    assert r["aer_edits"] == 1 and r["aer_len"] == 3
    assert r["cer_edits"] == 1


def test_cer_report_micro_average_and_regression_flag():
    samples = [
        {"id": "a", "reference": "కర్ణుడు వచ్చాడు", "raw": "కరణుడు వచ్చాడు", "corrected": "కర్ణుడు వచ్చాడు"},
        {"id": "b", "reference": "భీష్ముడు అన్నాడు", "raw": "భీష్ముడు అన్నాడు", "corrected": "భీష్మడు అన్నాడు"},
    ]
    rep = eh.cer_report(samples)
    assert rep["worse_after_correction"] == ["b"]
    assert rep["per_sample"][0]["corrected"]["cer"] == 0.0 and rep["per_sample"][0]["raw"]["cer"] > 0
    assert rep["micro"]["cer"]["raw"] > 0 and "WARNING" in eh.format_cer(rep)
    with pytest.raises(eh.EvalError, match="missing 'corrected'"):
        eh.cer_report([{"id": "z", "reference": "a", "raw": "a", "corrected": ""}])


# ---------- glossary recovery ----------

def test_glossary_recovery(tmp_path):
    hdr = "term,entity_id,roman,type,variants,needs_context,source,notes\n"
    (tmp_path / "cur.csv").write_text(hdr + "కృష్ణుడు,Q1,,person,,0,,\nఅర్జునుడు,Q2,,person,,0,,\nభీష్ముడు,Q3,,person,,0,,\nభీముడు,Q4,,person,,0,,\n", encoding="utf-8")
    (tmp_path / "auto.csv").write_text(hdr + "కృష్ణుడు,Q1,,person,,0,,\nఅర్జున,Q2,,person,,0,,\nకౌరవులు,Q9,,person,,0,,\n", encoding="utf-8")
    r = eh.glossary_recovery(tmp_path / "auto.csv", tmp_path / "cur.csv")
    assert r["term_recall"] == pytest.approx(1 / 4)          # only కృష్ణుడు matches exactly
    assert r["entity_recall"] == pytest.approx(2 / 4)        # Q1 and Q2 are there under some spelling
    assert r["auto_only_rows"] == 2 and "భీముడు" in r["missing_terms_sample"]


# ---------- suggest and CLI ----------

def test_suggest_finds_spans_by_word_or_phonetic_key(tmp_path):
    spans = [{"video_id": V1, "start": 100.0, "end": 160.0, "text": "అప్పుడు కర్ణుడు యుద్ధానికి వచ్చాడు"},
             {"video_id": V1, "start": 160.0, "end": 220.0, "text": "భీష్ముడు మాట్లాడాడు"}]
    p = tmp_path / f"{V1}_spans.json"
    p.write_text(json.dumps({"spans": spans}, ensure_ascii=False), encoding="utf-8")
    hits = eh.suggest("కర్ణుడు", [p])
    assert [h["when"] for h in hits] == ["00:01:40-00:02:40"] and hits[0]["link"].endswith("?t=100")
    assert [h["when"] for h in eh.suggest("Karna", [p])] == ["00:01:40-00:02:40"]    # English query, Telugu text
    assert [h["when"] for h in eh.suggest("Bhishma", [p])] == ["00:02:40-00:03:40"]
    assert eh.suggest("కర్ణుడు యుద్ధానికి", [p]) and not eh.suggest("కర్ణుడు భీష్ముడు", [p])


def test_cli_end_to_end(tmp_path, capsys, qs, runs):
    qp, rp = tmp_path / "q.jsonl", tmp_path / "run.jsonl"
    eh.write_jsonl(qp, [q("q1", [(V1, 100, 160)]), q("q2", answerable=False)])
    eh.write_jsonl(rp, [run("q1", [span(V1, 100, 160)], "a", citations=[span(V1, 100, 160)]), run("q2", abstained=True)])
    assert eh.main(["init", "--dir", str(tmp_path / "e")]) == 0
    assert eh.main(["validate", "--questions", str(qp)]) == 0
    assert eh.main(["score", "--questions", str(qp), "--run", str(rp), "--out", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out and "100.0%" in out
    assert json.loads((tmp_path / "r.json").read_text())["metrics"]["mrr"] == 1.0
    assert eh.main(["score", "--questions", str(tmp_path / "none.jsonl"), "--run", str(rp)]) == 1
    sample = tmp_path / "sample.txt"
    sample.write_text(V1 + "\n")
    assert eh.main(["validate", "--questions", str(qp), "--exclude-ids", str(sample)]) == 0
    assert "excluded 1" in capsys.readouterr().out
