import types

import correction
import entities
import index_store as ix
import retrieval
from tests.test_index_store import fake_models
from tests.test_entities import make_gl


def pt(i, vid, text, cluster=None, ents=()):
    return types.SimpleNamespace(id=i, payload={"span_id": f"{vid}_{i}", "video_id": vid, "start": 10.0,
                                                "end": 60.0, "text": text, "entities": list(ents),
                                                "cluster_id": cluster or f"{vid}_{i}"})


class FakeClient:
    def __init__(self, dense, sparse):
        self.lists = {"dense": dense, "sparse": sparse}
        self.calls = []

    def query_points(self, collection_name, query, using, query_filter, limit, with_payload):
        self.calls.append((using, query_filter))
        return types.SimpleNamespace(points=self.lists[using][:limit])


class Emb:
    def encode(self, texts):
        return [[0.0, 1.0] for _ in texts]


class Rer:
    def score(self, q, texts):
        return [float(len(t)) for t in texts]     # longer text = better


def test_fusion_rerank_dedup_and_filter(monkeypatch):
    monkeypatch.setattr(ix, "_models", fake_models)
    a, b, c = pt(1, "v", "short"), pt(2, "v", "a longer span text"), pt(3, "w", "longest span text of all", cluster="v_2")
    cl = FakeClient([a, b], [b, c])
    out = retrieval.retrieve("q", cl, Emb(), Rer(), correction.Glossary(), "PL", k=5)
    ids = [h["point_id"] for h in out["hits"]]
    assert ids == ["3", "1"]                     # reranked by length; 2 collapsed into its cluster (3 won)
    assert all(h["reranked"] for h in out["hits"])
    assert {u for u, f in cl.calls} == {"dense", "sparse"}
    assert all(f.must[0].key == "playlist_id" and f.must[0].match.value == "PL" for _, f in cl.calls)


def test_no_reranker_keeps_fused_order(monkeypatch):
    monkeypatch.setattr(ix, "_models", fake_models)
    a, b = pt(1, "v", "x"), pt(2, "v", "y")
    out = retrieval.retrieve("q", FakeClient([a, b], [b, a]), Emb(), None, correction.Glossary(), "PL", k=2)
    assert [h["point_id"] for h in out["hits"]] == ["1", "2"]       # tie broken by first list order
    assert not out["hits"][0]["reranked"]


def test_rrf_prefers_items_in_both_lists():
    f = retrieval.rrf_fuse(["a", "b", "c"], ["c", "b"])
    assert f["b"] > f["a"] and f["c"] > f["a"]


def test_query_entities_by_roman_name_and_script(tmp_path):
    gl = make_gl(tmp_path, [{"term": "అర్జునుడు", "entity_id": "Q1", "roman": "Arjuna"},
                            {"term": "ధర్మరాజు", "entity_id": "Q2", "roman": "Dharmaraja"}])
    assert entities.query_entities("Who is Arjuna?", gl) == ["Q1"]
    assert entities.query_entities("అర్జునుడు ఎవరు", gl) == ["Q1"]
    assert entities.query_entities("what is dharma raja", gl) == ["Q2"]
    assert entities.query_entities("hello there", gl) == []