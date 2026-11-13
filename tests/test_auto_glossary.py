import json

import glossary_builder as gb
from tests.test_glossary_builder import wiki, make_fetcher  # noqa: F401  (fixture reuse)
from tests.test_entities import make_gl
import entities


def test_infer_category_type():
    f = gb.infer_category_type
    assert f("Category:Characters in the Mahabharata") == "person"
    assert f("Category:Kingdoms in the Mahabharata") == "place"
    assert f("Category:Parvas in the Mahabharata") == "text"
    assert f("Category:Machine learning algorithms") == "concept"


def test_auto_select_skips_junk_tiny_and_over_budget():
    pack = {"categories": [
        {"title": "Category:Machine learning", "_pages": 40, "_subcats": 5},
        {"title": "Category:Machine learning stubs", "_pages": 90, "_subcats": 0},
        {"title": "Category:Tiny", "_pages": 1, "_subcats": 0},
        {"title": "Category:Machine learning researchers", "_pages": 30, "_subcats": 40},
        {"title": "Category:Huge", "_pages": 9000, "_subcats": 0}]}
    out = gb.auto_select(pack)
    use = {c["title"]: c["use"] for c in out["categories"]}
    assert use == {"Category:Machine learning": True, "Category:Machine learning stubs": False,
                   "Category:Tiny": False, "Category:Machine learning researchers": True, "Category:Huge": False}
    assert out["categories"][3]["type"] == "person" and out["categories"][3]["depth"] == 0
    assert out["categories"][0]["type"] == "concept" and out["auto"] is True


def test_clean_topic_and_infer_topic():
    assert gb.clean_topic("NPTEL Machine Learning Lecture 1 - Hindi") == "Machine Learning"
    assert gb.infer_topic("x", ["a"], lambda p: json.dumps({"topic": "Machine learning"})) == "Machine learning"
    assert gb.infer_topic("NPTEL Deep Learning", [], lambda p: "garbage") == "Deep Learning"

    def boom(p):
        raise RuntimeError("quota")
    assert gb.infer_topic("NPTEL Deep Learning", [], boom) == "Deep Learning"


def test_auto_end_to_end_and_never_overwrites_a_pack(tmp_path, wiki):
    wiki.search_hits = ["Category:People", "Category:Places", "Category:Wikipedia stubs"]
    wiki.cat_info = {"Category:People": {"pages": 4, "subcats": 1}, "Category:Places": {"pages": 3, "subcats": 0},
                     "Category:Wikipedia stubs": {"pages": 50, "subcats": 0}}
    pack, out, aliases = tmp_path / "g" / "domain_pack.json", tmp_path / "g", tmp_path / "g" / "aliases.csv"
    logs = []
    r = gb.auto(make_fetcher(tmp_path, wiki), "Mahabharata", pack, out, aliases, log=logs.append)
    assert r["promoted"] and aliases.exists()
    chosen = [c["title"] for c in json.loads(pack.read_text(encoding="utf-8"))["categories"] if c["use"]]
    assert chosen == ["Category:People", "Category:Places"]
    rows = aliases.read_text(encoding="utf-8").splitlines()
    assert len(rows) > 1                                    # header + promoted label rows
    before = pack.read_bytes()
    gb.auto(make_fetcher(tmp_path, wiki), "Something else", pack, out, aliases, log=logs.append)
    assert pack.read_bytes() == before                      # an existing pack is used, not rewritten
    assert any("existing domain pack" in m for m in logs)


def test_text_entities_matches_an_asr_spelling_variant_by_phonetic_key(tmp_path):
    gl = make_gl(tmp_path, [{"term": "శంతనుడు", "entity_id": "Q1", "roman": "Shantanu"}])
    assert entities.text_entities("శాంతనుడు వచ్చాడు", gl) == ["Q1"]            # long vowel: same key
    assert entities.text_entities("శాంతనుడు వచ్చాడు", gl, key_match=False) == []
    assert entities.text_entities("కథ వచ్చాడు", gl) == []                       # short keys never match this way