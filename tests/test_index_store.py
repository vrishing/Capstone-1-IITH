import types

import index_store as ix
from spans import point_id


def test_tokens_have_word_key_and_entity():
    t = ix.tokens_for("అర్జునుడు వచ్చాడు.", ["Q1"])
    assert "అర్జునుడు" in t and "e:Q1" in t
    assert any(x.startswith("k:") for x in t)
    assert not any("." in x for x in t)


def test_phonetic_token_matches_across_spellings():
    a = set(ix.tokens_for("अर्जुन"))
    b = set(ix.tokens_for("అర్జున"))
    assert {x for x in a if x.startswith("k:")} & {x for x in b if x.startswith("k:")}


def test_sparse_doc_and_query_are_stable_and_sorted():
    i1, v1 = ix.sparse_doc("a a b", ["Q1"])
    i2, v2 = ix.sparse_doc("a a b", ["Q1"])
    assert (i1, v1) == (i2, v2) and i1 == sorted(i1) and len(set(i1)) == len(i1)
    assert max(v1) <= 2.2 + 1e-6                       # tf saturates below K1 + 1
    qi, qv = ix.sparse_query("a b", [])
    assert set(qi) <= set(i1) and all(w == 1.0 for w in qv)
    assert ix.sparse_query("a a a", []) == ix.sparse_query("a", [])


def fake_models():
    mk = lambda name: (lambda **kw: types.SimpleNamespace(_name=name, **kw))
    ns = types.SimpleNamespace(**{n: mk(n) for n in
        ["VectorParams", "SparseVectorParams", "SparseVector", "PointStruct", "Filter",
         "FieldCondition", "MatchValue", "FilterSelector"]})
    ns.Distance = types.SimpleNamespace(COSINE="cosine")
    ns.Modifier = types.SimpleNamespace(IDF="idf")
    return ns


class FakeClient:
    def __init__(self):
        self.points, self.created = {}, []

    def collection_exists(self, name):
        return bool(self.created)

    def create_collection(self, **kw):
        self.created.append(kw)

    def delete(self, collection_name, points_selector):
        vid = [c.match.value for c in points_selector.filter.must if c.key == "video_id"][0]
        self.points = {k: p for k, p in self.points.items() if p.payload["video_id"] != vid}

    def upsert(self, collection_name, points):
        for p in points:
            self.points[p.id] = p


def spans(vid, n):
    return [{"span_id": f"{vid}_{i:09d}", "video_id": vid, "start": i * 60.0, "end": i * 60.0 + 50,
             "text": f"word{i} alpha", "language": "te", "entities": ["Q1"]} for i in range(n)]


def test_upsert_is_idempotent_and_replaces_a_video(monkeypatch):
    monkeypatch.setattr(ix, "_models", fake_models)
    c = FakeClient()
    ix.ensure_collection(c)
    ix.ensure_collection(c)
    assert len(c.created) == 1
    assert set(c.created[0]["vectors_config"]) == {"dense"} and set(c.created[0]["sparse_vectors_config"]) == {"sparse"}
    s = spans("v1", 3)
    d = [[0.1] * 4 for _ in s]
    assert ix.upsert_video(c, s, d, "PL", {s[1]["span_id"]: "v1_000000000"}) == 3
    assert c.points[point_id(s[1]["span_id"])].payload["cluster_id"] == "v1_000000000"
    assert c.points[point_id(s[2]["span_id"])].payload["cluster_id"] == s[2]["span_id"]
    ix.upsert_video(c, s, d, "PL")
    assert len(c.points) == 3                                   # same IDs: nothing duplicated
    assert point_id("v1_000000000") in c.points
    ix.upsert_video(c, spans("v1", 2), d[:2], "PL")             # re-chunked into fewer spans
    assert len(c.points) == 2
    ix.upsert_video(c, spans("v2", 1), d[:1], "PL")             # another video is untouched
    assert len(c.points) == 3