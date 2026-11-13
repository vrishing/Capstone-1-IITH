import json

import eval_harness
import gold_pool as gp


def hit(sid, vid, start, text="t"):
    return {"span_id": sid, "video_id": vid, "start": start, "end": start + 50, "text": text, "url": "u"}


def fake_retrieve(text, mode, k):
    data = {"hybrid": [hit("a_1", "a", 10), hit("b_1", "b", 100)], "dense": [hit("a_1", "a", 10)],
            "sparse": [hit("a_2", "a", 60)]}
    return data[mode][:k]


def test_pool_unions_retrievers_and_records_who_found_it():
    pooled = gp.pool_for({"question": "q", "hops": 1}, fake_retrieve)
    by = {p["span_id"]: p for p in pooled}
    assert set(by) == {"a_1", "b_1", "a_2"}
    assert by["a_1"]["found_by"] == ["hybrid", "dense"] and by["a_2"]["found_by"] == ["sparse"]


def test_build_apply_roundtrip_and_eval_harness_accepts_it(tmp_path):
    draft = [{"id": "m1", "question": "q1", "answerable": True, "category": "simple", "hops": 1, "language": "en"},
             {"id": "m2", "question": "q2", "answerable": True, "category": "multi_video", "hops": 2, "language": "en"},
             {"id": "m3", "question": "q3", "answerable": False, "category": "unanswerable", "hops": 0},
             {"id": "m4", "question": "q4", "answerable": True, "category": "simple", "hops": 1}]
    csvp = str(tmp_path / "pool.csv")
    assert gp.build(draft, fake_retrieve, csvp) == 3 * 4
    rows = gp.read_pool(csvp)
    for r in rows:
        if r["qid"] in ("m1", "m2") and r["span_id"] in ("a_1", "a_2"):
            r["label"] = "y"
        if r["qid"] == "m2" and r["span_id"] == "b_1":
            r["llm"] = "y"
        if r["qid"] == "m3" and r["span_id"] == "b_1":
            r["label"] = "y"
    out, issues = gp.apply(draft, rows)
    ids = [q["id"] for q in out]
    assert ids == ["m1", "m2", "m3"]                       # m4 has no gold: not written
    assert out[0]["gold"] == [{"video_id": "a", "start": 10.0, "end": 110.0}]   # a_1 (10-60) + a_2 (60-110) merged
    assert issues["no_gold"] == ["m4"] and issues["unanswerable_has_evidence"] == ["m3"]
    assert issues["too_few_videos"] == ["m2"]              # needs 2 videos, only one labelled by a human
    out2, issues2 = gp.apply(draft, rows, accept_llm=True)
    assert issues2["too_few_videos"] == [] and len(out2[1]["gold"]) == 2
    p = tmp_path / "q.jsonl"
    gp.write_questions(str(p), [q for q in out if q["id"] != "m3"])
    assert len(eval_harness.load_questions(p)) == 2        # the real harness reads it


def test_merge_ranges_keeps_videos_apart():
    r = gp.merge_ranges([("a", 0, 50), ("a", 52, 100), ("a", 200, 250), ("b", 0, 50)])
    assert [(g["video_id"], g["start"], g["end"]) for g in r] == [("a", 0, 100), ("a", 200, 250), ("b", 0, 50)]


def test_prelabel_is_resumable_and_respects_max_calls(tmp_path):
    csvp = str(tmp_path / "pool.csv")
    draft = [{"id": f"m{i}", "question": f"q{i}", "hops": 1} for i in range(3)]
    gp.build(draft, fake_retrieve, csvp)
    calls = []

    def gen(prompt, role, system, json_output, max_output_tokens):
        calls.append(1)
        return json.dumps({"relevant": [1]})
    r = gp.prelabel(csvp, gen, max_calls=2)
    assert (r["asked"], r["left"]) == (2, 1)
    r = gp.prelabel(csvp, gen, max_calls=2)
    assert (r["asked"], r["cached"], r["left"]) == (1, 2, 0) and len(calls) == 3
    labels = [x["llm"] for x in gp.read_pool(csvp) if x["qid"] == "m0"]
    assert labels.count("y") == 1


def test_prelabel_accepts_a_bare_list_reply(tmp_path):
    csvp = str(tmp_path / "pool.csv")
    gp.build([{"id": "m0", "question": "q", "hops": 1}], fake_retrieve, csvp)
    gen = lambda prompt, role, system, json_output, max_output_tokens: "[]"
    assert gp.prelabel(csvp, gen)["left"] == 0
    assert {r["llm"] for r in gp.read_pool(csvp)} == {"n"}
    gen2 = lambda prompt, role, system, json_output, max_output_tokens: "[2]"
    gp2 = str(tmp_path / "pool2.csv")
    gp.build([{"id": "m0", "question": "q", "hops": 1}], fake_retrieve, gp2)
    gp.prelabel(gp2, gen2)
    assert [r["llm"] for r in gp.read_pool(gp2)].count("y") == 1


def test_prelabel_asks_in_chunks_and_maps_numbers_back(tmp_path):
    csvp = str(tmp_path / "pool.csv")
    rows = [{"qid": "m0", "question": "q", "rank": i, "found_by": "hybrid", "span_id": f"s{i}",
             "video_id": "v", "start": i * 100, "end": i * 100 + 50, "link": "", "text": f"t{i}",
             "llm": "", "label": "", "note": ""} for i in range(25)]
    gp.write_pool(csvp, rows)
    calls = []

    def gen(prompt, role, system, json_output, max_output_tokens):
        calls.append(1)
        return json.dumps({"relevant": [1]})        # the first passage of every chunk
    assert gp.prelabel(csvp, gen)["left"] == 0
    assert len(calls) == 3
    ys = [i for i, r in enumerate(gp.read_pool(csvp)) if r["llm"] == "y"]
    assert ys == [0, 10, 20]