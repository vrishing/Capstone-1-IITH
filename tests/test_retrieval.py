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


def test_modes_skip_the_unused_side(monkeypatch):
    monkeypatch.setattr(ix, "_models", fake_models)
    a, b = pt(1, "v", "x"), pt(2, "v", "y")
    cl = FakeClient([a], [b])
    out = retrieval.retrieve("q", cl, Emb(), None, correction.Glossary(), "PL", k=5, mode="dense")
    assert [h["point_id"] for h in out["hits"]] == ["1"] and [u for u, _ in cl.calls] == ["dense"]
    cl = FakeClient([a], [b])
    out = retrieval.retrieve("q", cl, Emb(), None, correction.Glossary(), "PL", k=5, mode="sparse", use_entities=False)
    assert [h["point_id"] for h in out["hits"]] == ["2"] and [u for u, _ in cl.calls] == ["sparse"]
    import pytest
    with pytest.raises(ValueError):
        retrieval.retrieve("q", cl, Emb(), None, correction.Glossary(), "PL", mode="nope")


def test_weighted_rrf_and_rewrite_runs_dense_twice_and_reranks_with_rewrite(monkeypatch):
    f = retrieval.rrf_fuse(["a", "b"], ["b", "a"], weights=[3.0, 1.0])
    assert f["a"] > f["b"]
    monkeypatch.setattr(ix, "_models", fake_models)
    a, b = pt(1, "v", "short"), pt(2, "v", "a longer span text")
    cl = FakeClient([a, b], [b])
    seen = []

    class Rer2:
        def score(self, q, texts):
            seen.append(q)
            return [float(len(t)) for t in texts]
    out = retrieval.retrieve("who?", cl, Emb(), Rer2(), correction.Glossary(), "PL", k=5,
                             rewrite_fn=lambda q: "ఎవరు")
    assert [u for u, _ in cl.calls].count("dense") == 2 and out["rewritten"] == "ఎవరు"
    assert seen == ["ఎవరు"]
    cl2 = FakeClient([a, b], [b])
    out2 = retrieval.retrieve("ఎవరు", cl2, Emb(), None, correction.Glossary(), "PL", rewrite_fn=lambda q: "")
    assert [u for u, _ in cl2.calls].count("dense") == 1 and out2["rewritten"] == ""


def test_sparse_weight_changes_the_fused_order(monkeypatch):
    monkeypatch.setattr(ix, "_models", fake_models)
    a, b = pt(1, "v", "x"), pt(2, "v", "y")
    cl = lambda: FakeClient([a, b], [b, a])          # dense says a first, sparse says b first
    top = lambda sw: retrieval.retrieve("q", cl(), Emb(), None, correction.Glossary(), "PL", k=1,
                                        sparse_weight=sw)["hits"][0]["point_id"]
    assert top(0.2) == "1" and top(5.0) == "2"
    f = retrieval.rrf_fuse(["a"], ["a"], weights=[1.0, 0.5])
    assert abs(f["a"] - 1.5 / (retrieval.RRF_K + 1)) < 1e-9