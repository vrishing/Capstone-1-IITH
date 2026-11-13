import json

import query_rewrite as qr


def test_only_latin_questions_are_rewritten_and_answers_are_cached(tmp_path):
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return json.dumps({"query": "ద్రోణుడిని ఎవరు చంపారు?"})
    cache = str(tmp_path / "c" / "rw.json")
    assert qr.rewrite("Who killed Drona?", "te", gen, "Mahabharata", cache) == "ద్రోణుడిని ఎవరు చంపారు?"
    assert qr.rewrite("Who killed Drona?", "te", gen, "Mahabharata", cache) == "ద్రోణుడిని ఎవరు చంపారు?"
    assert len(calls) == 1 and "Telugu" in calls[0] and "Mahabharata" in calls[0]
    assert qr.rewrite("ద్రోణుడిని ఎవరు చంపారు?", "te", gen, "", cache) == ""        # already in script
    assert len(calls) == 1


def test_bad_replies_give_no_rewrite_and_are_not_cached(tmp_path):
    cache = str(tmp_path / "rw.json")
    for bad in ('{"query": "still english"}', "not json", '{"query": ""}'):
        assert qr.rewrite("Who killed Drona?", "te", lambda p, b=bad: b, "", cache) == ""
    def boom(p):
        raise RuntimeError("quota")
    assert qr.rewrite("Who killed Drona?", "te", boom, "", cache) == ""
    assert qr.rewrite("Who killed Drona?", "xx", lambda p: "{}", "", cache) == ""    # unknown language
    assert not (tmp_path / "rw.json").exists()
    assert qr.rewrite("Who killed Drona?", "te", lambda p: '```json\n{"query": "ద్రోణుడు ఎవరు"}\n```', "", cache) == "ద్రోణుడు ఎవరు"