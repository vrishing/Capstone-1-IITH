import csv

import correction
import entities
from spans import build_spans


def make_gl(tmp_path, rows):
    p = tmp_path / "aliases.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "entity_id", "roman", "type", "variants",
                                          "needs_context", "source", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({"roman": "", "type": "person", "variants": "", "needs_context": "",
                        "source": "", "notes": "", **r})
    return correction.load_glossary(str(p))


def test_exact_inflected_variant_and_phrase(tmp_path):
    gl = make_gl(tmp_path, [
        {"term": "అర్జునుడు", "entity_id": "Q1", "variants": "అర్జునడు"},
        {"term": "ధర్మరాజు", "entity_id": "Q2"},
        {"term": "కురుక్షేత్ర యుద్ధం", "entity_id": "Q3", "type": "event"},
    ])
    assert entities.text_entities("అర్జునుడు వచ్చాడు", gl) == ["Q1"]
    assert entities.text_entities("అర్జునుని చూసి", gl) == ["Q1"]        # inflected: stem + ending
    assert entities.text_entities("అర్జునడు అన్నాడు", gl) == ["Q1"]      # known variant
    assert entities.text_entities("ధర్మరాజు, కురుక్షేత్ర యుద్ధం జరిగింది.", gl) == ["Q2", "Q3"]
    assert entities.text_entities("ఇంటికి వెళ్ళాడు", gl) == []


def test_empty_glossary_tags_nothing(tmp_path):
    gl = correction.load_glossary(str(tmp_path / "missing.csv"))
    assert entities.text_entities("anything at all", gl) == []


def test_tag_spans_fills_entities(tmp_path):
    gl = make_gl(tmp_path, [{"term": "అర్జునుడు", "entity_id": "Q1"}])
    segs = [{"text": "అర్జునుడు వచ్చాడు", "start": 0.0, "duration": 50.0}]
    sp = entities.tag_spans(build_spans("v", segs, "te"), gl)
    assert sp[0]["entities"] == ["Q1"]


def test_clusters_group_near_duplicates_across_videos():
    base = " ".join(f"w{i}" for i in range(40))
    mk = lambda v, t, st: {"span_id": f"{v}_{st:09d}", "video_id": v, "start": st, "text": t}
    spans = [mk("b", base + " extra", 0), mk("a", base, 0), mk("a", " ".join(f"z{i}" for i in range(40)), 100000)]
    entities.cluster_spans(spans)
    assert spans[0]["cluster_id"] == spans[1]["cluster_id"] == "a_000000000"   # earliest member
    assert spans[2]["cluster_id"] == spans[2]["span_id"]                      # singleton