"""
Tests for glossary_builder.py. No internet needed.

FakeWiki stands in for Wikipedia and Wikidata. It mimics the real API's habits that matter:
paging with 'continue', langlinks arriving in a second reply, empty maps sent as [], and
a 50-title limit per request. The Telugu strings are test fixtures, not claims about Wikipedia.
"""
import csv
import http.server
import json
import threading
import urllib.parse

import pytest

import glossary_builder as gb

GOOD_UA = "TestSuite/0.1 (tester@college.edu)"


# --------------------------------------------------------------------------------------
# Fake Wikipedia / Wikidata
# --------------------------------------------------------------------------------------
class FakeWiki:
    PAGE_SIZE = 2  # category listings come back this many at a time, to force paging

    def __init__(self):
        self.calls = []
        self.fail_plan = []          # statuses to return before behaving, e.g. [429, 503]
        self.members = {}            # category title -> list of {"ns", "title"}
        self.qid = {}                # en page title -> Q id
        self.te_title = {}           # en page title -> te page title
        self.entities = {}           # Q id -> wbgetentities dict
        self.redirects = {}          # te title -> [redirect titles]
        self.search_hits = []
        self.cat_info = {}
        self.max_titles_seen = 0

    def __call__(self, url, params, headers):
        self.calls.append((url, dict(params)))
        assert headers["User-Agent"] == GOOD_UA
        if self.fail_plan:
            return self.fail_plan.pop(0), "", {"retry-after": "1"}
        host = urllib.parse.urlparse(url).netloc
        if host == "www.wikidata.org":
            body = self._wikidata(params)
        elif params.get("list") == "search":
            body = {"query": {"search": [{"title": t} for t in self.search_hits]}}
        elif params.get("list") == "categorymembers":
            body = self._members(params)
        elif params.get("prop") == "categoryinfo":
            body = {"query": {"pages": [{"title": t, "categoryinfo": self.cat_info.get(t, {})}
                                        for t in params["titles"].split("|")]}}
        elif params.get("prop") == "redirects":
            body = {"query": {"pages": [{"title": t, "redirects": [{"title": r} for r in self.redirects.get(t, [])]}
                                        for t in params["titles"].split("|")]}}
        elif "pageprops" in params.get("prop", ""):
            body = self._pageprops(params)
        else:
            raise AssertionError(f"unexpected call {params}")
        return 200, json.dumps(body, ensure_ascii=False), {}

    def _members(self, p):
        items = self.members.get(p["cmtitle"], [])
        start = int(p.get("cmcontinue", 0))
        chunk = items[start : start + self.PAGE_SIZE]
        body = {"query": {"categorymembers": chunk}}
        if start + self.PAGE_SIZE < len(items):
            body["continue"] = {"cmcontinue": str(start + self.PAGE_SIZE), "continue": "-||"}
        return body

    def _pageprops(self, p):
        titles = p["titles"].split("|")
        self.max_titles_seen = max(self.max_titles_seen, len(titles))
        assert len(titles) <= 50
        wants_links = "langlinks" in p["prop"]
        if wants_links and "llcontinue" in p:  # second reply carries the langlinks
            pages = [{"title": t, **({"langlinks": [{"lang": "te", "title": self.te_title[t]}]} if t in self.te_title else {})}
                     for t in titles]
            return {"query": {"pages": pages}}
        pages = [{"title": t, **({"pageprops": {"wikibase_item": self.qid[t]}} if t in self.qid else {})} for t in titles]
        body = {"query": {"pages": pages}}
        if wants_links:
            body["continue"] = {"llcontinue": "x", "continue": "||"}
        return body

    def _wikidata(self, p):
        ents = {}
        for q in p["ids"].split("|"):
            ents[q] = self.entities.get(q, {"id": q, "missing": ""})
        return {"entities": ents}


def ent(qid, en, te=None, aliases_te=(), desc=None, tewiki=None):
    labels = {"en": {"language": "en", "value": en}}
    if te:
        labels["te"] = {"language": "te", "value": te}
    return {
        "id": qid,
        "labels": labels,
        "aliases": {"te": [{"language": "te", "value": a} for a in aliases_te]} if aliases_te else [],
        "descriptions": {"en": {"language": "en", "value": desc}} if desc else [],
        "sitelinks": {"tewiki": {"site": "tewiki", "title": tewiki or te}} if (tewiki or te) else {},
    }


BHISHMA, KRISHNA, DRAUPADI, HASTI, DHARMA = "భీష్ముడు", "కృష్ణుడు", "ద్రౌపది", "హస్తినాపురం", "ధర్మం"
KRISHNA_SHORT = "కృష్ణ"  # alias of both Krishna and Draupadi in the fixture: an ambiguous spelling


@pytest.fixture
def wiki():
    w = FakeWiki()
    m = lambda *titles: [{"ns": 0, "title": t} for t in titles]
    sub = lambda t: {"ns": 14, "title": t}
    w.members = {
        "Category:People": m("Bhishma", "Krishna", "Arjuna") + [sub("Category:Warriors")],
        "Category:Warriors": m("Karna", "Obscure page") + [sub("Category:Deep"), sub("Category:People")],  # cycle back
        "Category:Deep": m("Too deep"),
        "Category:Places": m("Hastinapura"),
        "Category:Ideas": m("Dharma"),
    }
    w.qid = {"Bhishma": "Q1", "Krishna": "Q2", "Arjuna": "Q3", "Karna": "Q4", "Hastinapura": "Q6",
             "Dharma": "Q8", "Too deep": "Q9"}
    w.te_title = {"Bhishma": BHISHMA + " (మహాభారతం)", "Krishna": KRISHNA, "Hastinapura": HASTI, "Dharma": DHARMA}
    w.entities = {
        "Q1": ent("Q1", "Bhishma", BHISHMA, ["భీష్మ", "గాంగేయుడు"], "Kuru elder", tewiki=BHISHMA + " (మహాభారతం)"),
        "Q2": ent("Q2", "Krishna", KRISHNA, ["కేశవుడు", KRISHNA_SHORT, "Krishna"]),   # last alias is Latin
        "Q3": ent("Q3", "Arjuna"),                                                    # nothing in Telugu
        "Q4": ent("Q4", "Karna", "కర్ణుడు"),
        "Q5": ent("Q5", "Draupadi", DRAUPADI, [KRISHNA_SHORT]),
        "Q6": ent("Q6", "Hastinapura", HASTI),
        "Q8": ent("Q8", "Dharma", DHARMA),
    }
    w.redirects = {BHISHMA + " (మహాభారతం)": ["భీష్మాచార్యుడు", BHISHMA]}
    return w


def make_fetcher(tmp_path, wiki, **kw):
    kw.setdefault("min_interval_s", 0)
    kw.setdefault("sleep", lambda s: None)
    return gb.Fetcher(GOOD_UA, tmp_path / "cache", transport=wiki, **kw)


def pack_for_build(**over):
    p = {
        "topic": "Test", "language": "te",
        "categories": [
            {"wiki": "en", "title": "Category:People", "type": "person", "depth": 1, "use": True},
            {"wiki": "en", "title": "Category:Places", "type": "place", "depth": 0, "use": True},
            {"wiki": "en", "title": "Category:Ideas", "type": "concept", "depth": 0, "use": True},
            {"wiki": "en", "title": "Category:Skipped", "type": "person", "depth": 0, "use": False},
        ],
        "wikidata_ids": [{"id": "Q5", "type": "person"}],
    }
    p.update(over)
    return p


def write_pack(tmp_path, pack):
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
    return path


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# User agent
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("ua", [
    "", "Pravachanam/0.1", "Pravachanam/0.1 (CHANGE_ME@example.com)", "x (you@example.org)",
    "x (https://example.com/me)", "x (put-a-real-email)",
])
def test_placeholder_or_missing_contact_is_refused(ua):
    with pytest.raises(gb.FetchError, match="GLOSSARY_USER_AGENT"):
        gb.check_user_agent(ua)


def test_real_contact_is_accepted():
    assert gb.check_user_agent(GOOD_UA) == GOOD_UA
    assert gb.check_user_agent("Bot/1 (https://github.com/vrishing/Capstone-1-IITH)")


def test_fetcher_refuses_to_start_with_placeholder(tmp_path):
    with pytest.raises(gb.FetchError):
        gb.Fetcher("Bot/1 (CHANGE_ME@example.com)", tmp_path)


# --------------------------------------------------------------------------------------
# Fetcher: cache and retries
# --------------------------------------------------------------------------------------
def test_second_identical_call_comes_from_cache(tmp_path, wiki):
    wiki.search_hits = ["Category:A"]
    f = make_fetcher(tmp_path, wiki)
    p = {"action": "query", "list": "search", "srsearch": "x"}
    f.get_json("https://en.wikipedia.org/w/api.php", p)
    f.get_json("https://en.wikipedia.org/w/api.php", p)
    assert len(wiki.calls) == 1
    assert f.stats == {"network": 1, "cache": 1, "retries": 0}
    # a fresh Fetcher on the same folder also hits the cache (survives restarts)
    f2 = make_fetcher(tmp_path, wiki)
    f2.get_json("https://en.wikipedia.org/w/api.php", p)
    assert len(wiki.calls) == 1


def test_different_params_are_cached_separately(tmp_path, wiki):
    f = make_fetcher(tmp_path, wiki)
    f.get_json("https://en.wikipedia.org/w/api.php", {"list": "search", "srsearch": "a"})
    f.get_json("https://en.wikipedia.org/w/api.php", {"list": "search", "srsearch": "b"})
    assert len(wiki.calls) == 2


def test_corrupt_cache_file_is_refetched(tmp_path, wiki):
    f = make_fetcher(tmp_path, wiki)
    p = {"list": "search", "srsearch": "a"}
    f.get_json("https://en.wikipedia.org/w/api.php", p)
    next((tmp_path / "cache").rglob("*.json")).write_text("{not json", encoding="utf-8")
    assert f.get_json("https://en.wikipedia.org/w/api.php", p)["query"]["search"] == []
    assert len(wiki.calls) == 2


def test_retries_429_and_5xx_then_succeeds_and_honours_retry_after(tmp_path, wiki):
    wiki.fail_plan = [429, 503]
    sleeps = []
    f = make_fetcher(tmp_path, wiki, sleep=sleeps.append)
    f.get_json("https://en.wikipedia.org/w/api.php", {"list": "search", "srsearch": "a"})
    assert f.stats["retries"] == 2
    assert sleeps == [1.0, 1.0]  # Retry-After header was 1


def test_backoff_doubles_without_retry_after(tmp_path):
    calls = {"n": 0}

    def transport(url, params, headers):
        calls["n"] += 1
        return (500, "", {}) if calls["n"] < 4 else (200, "{}", {})

    sleeps = []
    f = gb.Fetcher(GOOD_UA, tmp_path, transport=transport, sleep=sleeps.append, min_interval_s=0, backoff_s=2)
    f.get_json("https://x/api", {"a": "1"})
    assert sleeps == [2.0, 4.0, 8.0]


def test_gives_up_after_max_retries(tmp_path, wiki):
    wiki.fail_plan = [500] * 20
    f = make_fetcher(tmp_path, wiki, max_retries=2)
    with pytest.raises(gb.FetchError, match="Gave up after 3 attempts"):
        f.get_json("https://en.wikipedia.org/w/api.php", {"a": "1"})
    assert len(wiki.calls) == 3


def test_client_error_is_not_retried(tmp_path, wiki):
    wiki.fail_plan = [404]
    f = make_fetcher(tmp_path, wiki)
    with pytest.raises(gb.FetchError, match="HTTP 404"):
        f.get_json("https://en.wikipedia.org/w/api.php", {"a": "1"})
    assert len(wiki.calls) == 1


def test_api_level_error_fails_but_maxlag_is_retried(tmp_path):
    replies = [{"error": {"code": "maxlag"}}, {"error": {"code": "badvalue", "info": "no"}}]
    seen = []

    def transport(url, params, headers):
        seen.append(1)
        return 200, json.dumps(replies.pop(0)), {}

    f = gb.Fetcher(GOOD_UA, tmp_path, transport=transport, sleep=lambda s: None, min_interval_s=0)
    with pytest.raises(gb.FetchError, match="badvalue"):
        f.get_json("https://x/api", {"a": "1"})
    assert len(seen) == 2  # maxlag retried once, badvalue stopped it


def test_throttle_spaces_requests(tmp_path):
    t = {"now": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(round(s, 3))
        t["now"] += s

    f = gb.Fetcher(GOOD_UA, tmp_path, transport=lambda *a: (200, "{}", {}), sleep=sleep,
                   clock=lambda: t["now"], min_interval_s=0.5)
    f.get_json("https://x/api", {"a": "1"})
    f.get_json("https://x/api", {"a": "2"})
    assert sleeps == [0.5]  # first call free, second waited


class _Handler(http.server.BaseHTTPRequestHandler):
    log = []

    def do_GET(self):
        _Handler.log.append((self.path, self.headers.get("User-Agent")))
        if len([p for p, _ in _Handler.log]) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "3")
            self.end_headers()
            return
        body = json.dumps({"query": {"echo": "ప్రశ్న"}}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_real_urllib_transport_against_local_server(tmp_path):
    _Handler.log = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        sleeps = []
        f = gb.Fetcher(GOOD_UA, tmp_path, sleep=sleeps.append, min_interval_s=0)  # default transport
        data = f.get_json(f"http://127.0.0.1:{srv.server_port}/w/api.php", {"titles": "భీష్ముడు|x", "a": "1"})
    finally:
        srv.shutdown()
    assert data["query"]["echo"] == "ప్రశ్న"
    assert sleeps == [3.0]
    assert all(ua == GOOD_UA for _, ua in _Handler.log)
    assert urllib.parse.parse_qs(urllib.parse.urlparse(_Handler.log[0][0]).query)["titles"] == ["భీష్ముడు|x"]


def test_unreachable_server_is_retried_then_reported(tmp_path):
    f = gb.Fetcher(GOOD_UA, tmp_path, sleep=lambda s: None, min_interval_s=0, max_retries=1)
    with pytest.raises(gb.FetchError, match="Gave up"):
        f.get_json("http://127.0.0.1:9/w/api.php", {"a": "1"})


# --------------------------------------------------------------------------------------
# Surface cleaning
# --------------------------------------------------------------------------------------
def test_clean_surface():
    assert gb.clean_surface("భీష్ముడు (మహాభారతం)") == "భీష్ముడు"
    assert gb.clean_surface("భీష్ముడు (a) (b)") == "భీష్ముడు"
    assert gb.clean_surface("  భీష్ముడు_  ప్రభువు ") == "భీష్ముడు ప్రభువు"
    assert gb.clean_surface("(only qualifier)") == ""
    assert gb.clean_surface(None) == ""
    assert gb.clean_surface("\u0c15\u0c46\u0c56") == "\u0c15\u0c48"  # two-part vowel sign becomes the single NFC one


# --------------------------------------------------------------------------------------
# Category walking, batching
# --------------------------------------------------------------------------------------
def test_walk_follows_paging_depth_and_survives_cycles(tmp_path, wiki):
    f = make_fetcher(tmp_path, wiki)
    pages, info = gb.walk_category(f, "en", "Category:People", depth=1, limit=100)
    assert set(pages) == {"Bhishma", "Krishna", "Arjuna", "Karna", "Obscure page"}  # paging worked, 'Too deep' not reached
    assert pages["Karna"] == "Category:Warriors"
    assert info == {"categories_walked": 2, "truncated": False}


def test_walk_depth_zero_ignores_subcategories(tmp_path, wiki):
    pages, info = gb.walk_category(make_fetcher(tmp_path, wiki), "en", "Category:People", 0, 100)
    assert set(pages) == {"Bhishma", "Krishna", "Arjuna"} and info["categories_walked"] == 1


def test_walk_deeper_reaches_third_level(tmp_path, wiki):
    pages, _ = gb.walk_category(make_fetcher(tmp_path, wiki), "en", "Category:People", 2, 100)
    assert "Too deep" in pages


def test_walk_never_revisits_a_category_even_when_depth_allows_it(tmp_path, wiki):
    # People -> Warriors -> People is a loop; People -> Warriors -> Deep is the only other branch
    f = make_fetcher(tmp_path, wiki)
    pages, info = gb.walk_category(f, "en", "Category:People", depth=10, limit=100)
    assert info["categories_walked"] == 3
    listing_calls = [c for c in wiki.calls if c[1].get("list") == "categorymembers" and "cmcontinue" not in c[1]]
    assert sorted(c[1]["cmtitle"] for c in listing_calls) == ["Category:Deep", "Category:People", "Category:Warriors"]


def test_walk_stops_at_limit_and_says_so(tmp_path, wiki):
    pages, info = gb.walk_category(make_fetcher(tmp_path, wiki), "en", "Category:People", 1, 4)
    assert len(pages) == 4 and info["truncated"] is True


def test_walk_ignores_non_article_namespaces(tmp_path, wiki):
    wiki.members["Category:X"] = [{"ns": 0, "title": "A"}, {"ns": 10, "title": "Template:T"}, {"ns": 6, "title": "File:F"}]
    pages, _ = gb.walk_category(make_fetcher(tmp_path, wiki), "en", "Category:X", 0, 10)
    assert list(pages) == ["A"]


def test_lookup_batches_titles_by_50_and_merges_langlinks(tmp_path, wiki):
    titles = [f"Page {i}" for i in range(120)]
    for i, t in enumerate(titles):
        wiki.qid[t] = f"Q{1000 + i}"
        if i % 2 == 0:
            wiki.te_title[t] = f"తె {i}"
    out = gb.lookup_pages(make_fetcher(tmp_path, wiki), "en", titles)
    assert wiki.max_titles_seen == 50
    assert all(out[t]["qid"] for t in titles)
    assert out["Page 0"]["te_title"] == "తె 0" and out["Page 1"]["te_title"] is None


def test_lookup_on_telugu_wiki_uses_title_itself(tmp_path, wiki):
    wiki.qid["భీష్ముడు"] = "Q1"
    out = gb.lookup_pages(make_fetcher(tmp_path, wiki), "te", ["భీష్ముడు"])
    assert out["భీష్ముడు"] == {"qid": "Q1", "te_title": "భీష్ముడు"}
    assert all("langlinks" not in c[1].get("prop", "") for c in wiki.calls)


def test_entities_handle_missing_empty_maps_and_merged_items(tmp_path, wiki):
    wiki.entities["Q7"] = {"id": "Q10", "redirects": {"from": "Q7", "to": "Q10"},
                           "labels": [], "aliases": [], "descriptions": []}
    out = gb.fetch_entities(make_fetcher(tmp_path, wiki), ["Q1", "Q7", "Q99"])
    assert out["Q99"] is None
    assert out["Q7"]["id"] == "Q10" and out["Q7"]["te_label"] is None and out["Q7"]["te_aliases"] == []
    assert out["Q1"]["te_label"] == BHISHMA and out["Q1"]["en_description"] == "Kuru elder"


# --------------------------------------------------------------------------------------
# Pack
# --------------------------------------------------------------------------------------
def test_pack_loading_rules(tmp_path):
    p = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    assert [c["title"] for c in p["categories"]] == ["Category:People", "Category:Places", "Category:Ideas"]  # use:false dropped
    assert p["no_fuzzy_types"] == ["concept", "other"] and p["max_pages"] == 2000

    with pytest.raises(gb.PackError, match="use:true but has no 'type'"):
        gb.load_pack(write_pack(tmp_path, pack_for_build(categories=[{"wiki": "en", "title": "C", "type": "", "use": True}])))
    with pytest.raises(gb.PackError, match="Nothing to harvest"):
        gb.load_pack(write_pack(tmp_path, pack_for_build(categories=[], wikidata_ids=[])))
    with pytest.raises(gb.PackError, match="active playlist is 'te'"):
        gb.load_pack(write_pack(tmp_path, pack_for_build(language="hi")))
    with pytest.raises(gb.PackError, match="wikidata_ids"):
        gb.load_pack(write_pack(tmp_path, pack_for_build(wikidata_ids=[{"id": "Krishna", "type": "person"}])))
    with pytest.raises(gb.PackError, match="No domain pack"):
        gb.load_pack(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(gb.PackError, match="not valid JSON"):
        gb.load_pack(tmp_path / "bad.json")


def test_propose_writes_unused_categories_and_will_not_overwrite(tmp_path, wiki):
    wiki.search_hits = ["Category:Mahabharata", "Category:Characters in the Mahabharata"]
    wiki.cat_info = {"Category:Mahabharata": {"pages": 12, "subcats": 3}}
    path = tmp_path / "glossary" / "domain_pack.json"
    gb.propose(make_fetcher(tmp_path, wiki), "Mahabharata", path, playlist_id="PL123")
    pack = json.loads(path.read_text(encoding="utf-8"))
    assert pack["playlist_id"] == "PL123" and pack["wikidata_ids"] == []
    assert [c["use"] for c in pack["categories"]] == [False, False]
    assert pack["categories"][0]["_pages"] == 12 and pack["categories"][1]["_pages"] == 0
    with pytest.raises(gb.PackError, match="already exists"):
        gb.propose(make_fetcher(tmp_path, wiki), "Mahabharata", path)
    gb.propose(make_fetcher(tmp_path, wiki), "Mahabharata", path, force=True)
    # a freshly proposed pack is not buildable until a human opts categories in
    with pytest.raises(gb.PackError, match="Nothing to harvest"):
        gb.load_pack(path)


# --------------------------------------------------------------------------------------
# Build, end to end
# --------------------------------------------------------------------------------------
@pytest.fixture
def built(tmp_path, wiki):
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    out = tmp_path / "glossary"
    report = gb.build(pack, make_fetcher(tmp_path, wiki), out, log=lambda m: None)
    return out, report, wiki


def test_build_writes_the_four_files(built):
    out, report, _ = built
    for name in ("aliases_candidates.csv", "missing_telugu.csv", "ambiguous.csv", "build_report.json"):
        assert (out / name).exists()
    assert not (out / "aliases.csv").exists()  # build never touches the live table
    assert (out / "aliases_candidates.csv").read_bytes().startswith(b"\xef\xbb\xbf")  # opens correctly in Excel


def test_candidate_rows_have_origin_source_and_cleaned_titles(built):
    out, _, _ = built
    rows = read_rows(out / "aliases_candidates.csv")
    assert list(rows[0].keys()) == gb.CANDIDATE_COLUMNS
    bh = {r["term"]: r for r in rows if r["entity_id"] == "Q1"}
    assert set(bh) == {BHISHMA, "భీష్మ", "గాంగేయుడు", "భీష్మాచార్యుడు"}  # '(మహాభారతం)' stripped, duplicate merged
    main = bh[BHISHMA]
    assert main["origin"] == "wikidata_label_te" and main["roman"] == "Bhishma" and main["type"] == "person"
    assert main["source"] == "wikidata:Q1/label;tewiki:" + BHISHMA + " (మహాభారతం);tewiki-redirect:" + BHISHMA
    assert "also: wikipedia_title_te,wikipedia_redirect_te" in main["notes"] and "Kuru elder" in main["notes"]
    assert bh["భీష్మ"]["origin"] == "wikidata_alias_te" and bh["భీష్మ"]["source"] == "wikidata:Q1/alias"
    assert bh["భీష్మాచార్యుడు"]["origin"] == "wikipedia_redirect_te"
    assert all(r["decision"] == "" and r["variants"] == "" for r in rows)


def test_latin_aliases_are_dropped_and_counted(built):
    out, report, _ = built
    assert "Krishna" not in {r["term"] for r in read_rows(out / "aliases_candidates.csv")}
    assert report["counts"]["surfaces_dropped"]["no_telugu_script"] == 1


def test_missing_telugu_lists_entities_without_guessing(built):
    out, report, _ = built
    miss = read_rows(out / "missing_telugu.csv")
    assert [(r["entity_id"], r["en_label"], r["type"]) for r in miss] == [("Q3", "Arjuna", "person")]
    assert "en:Category:People" in miss[0]["categories"]
    assert "Q3" not in {r["entity_id"] for r in read_rows(out / "aliases_candidates.csv")}
    assert report["counts"]["entities_missing_telugu"] == 1


def test_ambiguous_spelling_is_flagged_everywhere(built):
    out, report, _ = built
    amb = read_rows(out / "ambiguous.csv")
    assert len(amb) == 1 and amb[0]["term"] == KRISHNA_SHORT
    assert amb[0]["entity_ids"] == "Q2 Q5" and amb[0]["en_labels"] == "Krishna | Draupadi"
    rows = [r for r in read_rows(out / "aliases_candidates.csv") if r["term"] == KRISHNA_SHORT]
    assert {r["entity_id"] for r in rows} == {"Q2", "Q5"}
    assert all(r["needs_context"] == "1" and "ambiguous with" in r["notes"] for r in rows)


def test_needs_context_follows_no_fuzzy_types(built):
    out, _, _ = built
    by = {r["term"]: r for r in read_rows(out / "aliases_candidates.csv")}
    assert by[DHARMA]["needs_context"] == "1"      # concept: exact match only
    assert by[HASTI]["needs_context"] == "0"       # place
    assert by[BHISHMA]["needs_context"] == "0"     # person


def test_direct_wikidata_id_and_subcategory_pages_are_included(built):
    out, _, _ = built
    ids = {r["entity_id"] for r in read_rows(out / "aliases_candidates.csv")}
    assert "Q5" in ids   # came only from wikidata_ids
    assert "Q4" in ids   # came from a subcategory
    assert "Q9" not in ids  # depth 1 does not reach 'Too deep'
    assert {r["term"] for r in read_rows(out / "aliases_candidates.csv") if r["entity_id"] == "Q5"} == {DRAUPADI, KRISHNA_SHORT}


def test_report_contents(built):
    out, report, wiki = built
    on_disk = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    c = report["counts"]
    assert c["entities"] == 7 and c["entities_with_telugu"] == 6 and c["ambiguous_terms"] == 1
    assert c["pages_without_wikidata"] == 1 and report["pages_without_wikidata"][0]["title"] == "Obscure page"
    assert report["categories"][0]["title"] == "Category:People" and report["categories"][0]["truncated"] is False
    assert len(report["pack_sha256"]) == 64 and "CC0" in report["licences"]["wikidata"] and "CC BY-SA" in report["licences"]["wikipedia"]
    assert report["api_requests"]["network"] == len(wiki.calls)


def test_rebuild_uses_cache_only_and_gives_identical_files(tmp_path, wiki):
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    out = tmp_path / "glossary"
    f1 = make_fetcher(tmp_path, wiki)
    gb.build(pack, f1, out, log=lambda m: None)
    first = (out / "aliases_candidates.csv").read_bytes()
    n = len(wiki.calls)
    f2 = make_fetcher(tmp_path, wiki)
    gb.build(pack, f2, out, log=lambda m: None)
    assert len(wiki.calls) == n and f2.stats["network"] == 0
    assert (out / "aliases_candidates.csv").read_bytes() == first


def test_same_entity_in_two_categories_is_one_entity_with_conflict_note(tmp_path, wiki):
    wiki.members["Category:Places"].append({"ns": 0, "title": "Krishna"})
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    rows = [r for r in read_rows(tmp_path / "g" / "aliases_candidates.csv") if r["entity_id"] == "Q2"]
    assert rows and all(r["type"] == "other" and "type conflict: person/place" in r["notes"] for r in rows)


def test_merged_wikidata_items_do_not_create_duplicates(tmp_path, wiki):
    wiki.qid["Duplicate of Karna"] = "Q44"
    wiki.members["Category:Places"].append({"ns": 0, "title": "Duplicate of Karna"})
    wiki.entities["Q44"] = {**wiki.entities["Q4"]}  # same canonical id Q4 returned for Q44
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    karna = [r for r in read_rows(tmp_path / "g" / "aliases_candidates.csv") if r["term"] == "కర్ణుడు"]
    assert len(karna) == 1 and karna[0]["entity_id"] == "Q4"


def test_missing_wikidata_item_is_reported_not_crashed(tmp_path, wiki):
    wiki.qid["Ghost"] = "Q777"  # no entity record -> API says missing
    wiki.members["Category:Places"].append({"ns": 0, "title": "Ghost"})
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    report = gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    assert report["wikidata_items_not_returned"] == ["Q777"]


# --------------------------------------------------------------------------------------
# Promote
# --------------------------------------------------------------------------------------
EXISTING = (
    "term,entity_id,roman,type,variants,needs_context,source,notes\n"
    "హస్తిన,Q6,Hastinapura,place,,0,hand,my own row\n"
    "కర్ణుడు,Q999,Wrong,person,,0,hand,conflicts with Wikidata Q4\n"
)


@pytest.fixture
def cand_path(built):
    return built[0] / "aliases_candidates.csv"


def test_promote_label_mode_adds_only_label_rows_and_leaves_existing_bytes_alone(tmp_path, cand_path):
    aliases = tmp_path / "aliases.csv"
    aliases.write_bytes(EXISTING.encode("utf-8"))
    s = gb.promote(cand_path, aliases, accept="label")
    after = aliases.read_bytes()
    assert after.startswith(EXISTING.encode("utf-8"))  # existing rows byte-for-byte untouched
    rows = read_rows(aliases)
    terms = {r["term"] for r in rows}
    assert {BHISHMA, KRISHNA, DRAUPADI, HASTI, DHARMA} <= terms
    assert "భీష్మ" not in terms and "గాంగేయుడు" not in terms and "భీష్మాచార్యుడు" not in terms  # aliases/redirects not label
    assert KRISHNA_SHORT not in terms
    assert list(rows[0].keys()) == gb.ALIAS_COLUMNS
    assert s["conflicts"] == [{"line": s["conflicts"][0]["line"], "term": "కర్ణుడు", "new_entity": "Q4", "existing_entities": ["Q999"]}]
    assert "కర్ణుడు" in terms and not any(r["entity_id"] == "Q4" for r in rows)  # only the old Q999 row exists
    assert s["added"] == len(rows) - 2 and s["invalid"] == []


def test_promote_is_idempotent(tmp_path, cand_path):
    aliases = tmp_path / "aliases.csv"
    gb.promote(cand_path, aliases, accept="label")
    first = aliases.read_bytes()
    s = gb.promote(cand_path, aliases, accept="label")
    assert s["added"] == 0 and s["already_present"] > 0
    assert aliases.read_bytes() == first


def test_promote_creates_missing_table_with_header_and_no_bom(tmp_path, cand_path):
    aliases = tmp_path / "new" / "aliases.csv"
    gb.promote(cand_path, aliases, accept="label")
    raw = aliases.read_bytes()
    assert raw.startswith(b"term,entity_id,roman") and not raw.startswith(b"\xef\xbb\xbf")
    assert len(read_rows(aliases)) > 0


def test_promote_all_mode_adds_aliases_but_never_ambiguous(tmp_path, cand_path):
    aliases = tmp_path / "aliases.csv"
    s = gb.promote(cand_path, aliases, accept="all")
    terms = {r["term"] for r in read_rows(aliases)}
    assert {"భీష్మ", "గాంగేయుడు", "భీష్మాచార్యుడు"} <= terms
    assert KRISHNA_SHORT not in terms and s["skipped_ambiguous"] == 2


def _edit_candidates(path, decisions):
    rows = read_rows(path)
    for r in rows:
        key = (r["term"], r["entity_id"])
        if key in decisions:
            r["decision"] = decisions[key]
    path.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))


def test_promote_decided_mode_takes_only_accepted_rows_including_ambiguous(tmp_path, cand_path):
    _edit_candidates(cand_path, {("భీష్మ", "Q1"): "accept", (KRISHNA_SHORT, "Q5"): "accept", ("కేశవుడు", "Q2"): "reject"})
    aliases = tmp_path / "aliases.csv"
    s = gb.promote(cand_path, aliases, accept="decided")
    got = {(r["term"], r["entity_id"]) for r in read_rows(aliases)}
    assert got == {("భీష్మ", "Q1"), (KRISHNA_SHORT, "Q5")}
    assert s["added"] == 2


def test_reject_always_wins(tmp_path, cand_path):
    _edit_candidates(cand_path, {(BHISHMA, "Q1"): "reject"})
    aliases = tmp_path / "aliases.csv"
    s = gb.promote(cand_path, aliases, accept="label")
    assert BHISHMA not in {r["term"] for r in read_rows(aliases)} and s["rejected_by_you"] == 1


def test_promote_refuses_an_alias_table_with_other_columns(tmp_path, cand_path):
    aliases = tmp_path / "aliases.csv"
    aliases.write_text("term,entity\nx,y\n", encoding="utf-8")
    before = aliases.read_bytes()
    with pytest.raises(gb.PromoteError, match="Not touching it"):
        gb.promote(cand_path, aliases)
    assert aliases.read_bytes() == before


def test_promote_rejects_bad_rows_with_line_numbers(tmp_path):
    cands = tmp_path / "aliases_candidates.csv"
    base = dict(roman="r", type="person", variants="", needs_context="0", source="s", notes="", origin="wikidata_label_te", decision="")
    rows = [
        {**base, "term": "భీష్ముడు", "entity_id": "Q1"},
        {**base, "term": "Bhishma", "entity_id": "Q1"},
        {**base, "term": "కృష్ణుడు", "entity_id": "Krishna"},
        {**base, "term": "అర్జునుడు", "entity_id": "Q3", "type": ""},
        {**base, "term": "కర్ణుడు", "entity_id": "Q4", "needs_context": "yes"},
        {**base, "term": "", "entity_id": "Q5"},
    ]
    cands.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))
    s = gb.promote(cands, tmp_path / "aliases.csv")
    assert s["added"] == 1
    assert [(i["line"], i["reason"]) for i in s["invalid"]] == [
        (3, "term has no characters in the te script"), (4, "entity_id is not like Q123"), (5, "type is empty"),
        (6, "needs_context must be 0 or 1"), (7, "empty term"),
    ]


def test_promote_needs_all_candidate_columns(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("term,entity_id\nx,y\n", encoding="utf-8")
    with pytest.raises(gb.PromoteError, match="missing columns"):
        gb.promote(p, tmp_path / "a.csv")


def test_promote_keeps_windows_line_endings_and_adds_missing_final_newline(tmp_path, cand_path):
    aliases = tmp_path / "aliases.csv"
    original = EXISTING.replace("\n", "\r\n").rstrip("\r\n").encode("utf-8")  # CRLF, no final newline
    aliases.write_bytes(original)
    gb.promote(cand_path, aliases, accept="label")
    raw = aliases.read_bytes()
    assert raw.startswith(original + b"\r\n")
    assert raw.count(b"\n") == raw.count(b"\r\n")  # no mixed endings
    assert len(read_rows(aliases)) > 2


def test_promoted_file_satisfies_the_corrector_contract(tmp_path, cand_path):
    aliases = tmp_path / "aliases.csv"
    gb.promote(cand_path, aliases)
    with open(aliases, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == gb.ALIAS_COLUMNS == ["term", "entity_id", "roman", "type", "variants", "needs_context", "source", "notes"]
        for r in reader:
            assert r["term"] and gb.TELUGU_RE.search(r["term"]) and gb.QID_RE.match(r["entity_id"])
            assert r["needs_context"] in ("0", "1") and r["type"] and r["source"]


# --------------------------------------------------------------------------------------
# Clear error when no glossary exists
# --------------------------------------------------------------------------------------
def test_require_alias_table_gives_actionable_errors(tmp_path):
    with pytest.raises(gb.GlossaryMissingError, match="glossary_builder.py propose"):
        gb.require_alias_table(tmp_path / "aliases.csv")
    empty = tmp_path / "aliases.csv"
    empty.write_text(",".join(gb.ALIAS_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(gb.GlossaryMissingError, match="no rows"):
        gb.require_alias_table(empty)
    empty.write_text(EXISTING, encoding="utf-8")
    assert gb.require_alias_table(empty) == empty


# --------------------------------------------------------------------------------------
# Command line (main) with a stand-in config module
# --------------------------------------------------------------------------------------
@pytest.fixture
def cli(tmp_path, wiki, monkeypatch):
    import types
    cfg = types.ModuleType("config")
    cfg.PLAYLIST_ID = "PLTEST"
    cfg.GLOSSARY_DIR = str(tmp_path / "glossary" / "PLTEST")
    cfg.DOMAIN_PACK_PATH = str(tmp_path / "glossary" / "PLTEST" / "domain_pack.json")
    cfg.ALIAS_TABLE_PATH = str(tmp_path / "glossary" / "PLTEST" / "aliases.csv")
    cfg.GLOSSARY_USER_AGENT = GOOD_UA
    monkeypatch.setitem(__import__("sys").modules, "config", cfg)
    monkeypatch.setattr(gb, "urllib_transport", wiki)  # Fetcher falls back to this when no transport is given
    monkeypatch.setattr(gb.time, "sleep", lambda s: None)
    return cfg


def test_cli_propose_build_promote(cli, wiki, capsys):
    wiki.search_hits = ["Category:People", "Category:Places"]
    assert gb.main(["propose", "--topic", "Test"]) == 0
    pack = json.loads(open(cli.DOMAIN_PACK_PATH, encoding="utf-8").read())
    assert pack["playlist_id"] == "PLTEST" and not any(c["use"] for c in pack["categories"])

    assert gb.main(["build"]) == 1  # nothing opted in yet
    assert "Nothing to harvest" in capsys.readouterr().err

    for c, typ in zip(pack["categories"], ("person", "place")):
        c.update(use=True, type=typ, depth=0)
    open(cli.DOMAIN_PACK_PATH, "w", encoding="utf-8").write(json.dumps(pack, ensure_ascii=False))
    assert gb.main(["build"]) == 0
    assert (gb.Path(cli.GLOSSARY_DIR) / "aliases_candidates.csv").exists()
    assert (gb.Path(cli.GLOSSARY_DIR) / "_cache").is_dir()

    assert gb.main(["promote", "--accept", "label"]) == 0
    out = capsys.readouterr().out
    assert '"added"' in out
    assert {r["term"] for r in read_rows(cli.ALIAS_TABLE_PATH)} >= {BHISHMA, KRISHNA, HASTI}


def test_cli_refuses_placeholder_user_agent(cli, capsys):
    cli.GLOSSARY_USER_AGENT = "PravachanamRAG/0.1 (CHANGE_ME@example.com)"
    assert gb.main(["propose", "--topic", "Test"]) == 1
    assert "GLOSSARY_USER_AGENT" in capsys.readouterr().err


def test_cli_explains_missing_config_setting(cli):
    del cli.ALIAS_TABLE_PATH
    with pytest.raises(SystemExit, match="config_additions.py"):
        gb.main(["build"])


# --------------------------------------------------------------------------------------
# Fixes from the first live run
# --------------------------------------------------------------------------------------
def test_conflicting_categories_give_exact_only_other_type(tmp_path, wiki):
    wiki.members["Category:Places"].append({"ns": 0, "title": "Bhishma"})  # person in People, also listed as a place
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    rows = [r for r in read_rows(tmp_path / "g" / "aliases_candidates.csv") if r["entity_id"] == "Q1"]
    assert rows and all(r["type"] == "other" and r["needs_context"] == "1" for r in rows)
    assert all("type conflict: person/place" in r["notes"] for r in rows)


def test_telugu_link_dropped_when_wikidata_says_it_is_another_item(tmp_path, wiki):
    wiki.entities["Q6"]["sitelinks"] = {"tewiki": {"site": "tewiki", "title": "వేరే పేజీ"}}
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    report = gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    rows = [r for r in read_rows(tmp_path / "g" / "aliases_candidates.csv") if r["entity_id"] == "Q6"]
    # Wikidata's own label for Q6 is still valid; only the Wikipedia-derived spelling is removed
    assert [(r["term"], r["origin"]) for r in rows] == [(HASTI, "wikidata_label_te")]
    assert report["rejected_te_links"] == [{"entity_id": "Q6", "en_title": "Hastinapura",
                                            "claimed_te_title": HASTI, "wikidata_tewiki": "వేరే పేజీ"}]


def test_telugu_link_kept_when_wikidata_agrees(tmp_path, wiki):
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    report = gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    assert report["rejected_te_links"] == []
    assert HASTI in {r["term"] for r in read_rows(tmp_path / "g" / "aliases_candidates.csv")}


def test_phrase_redirects_dropped_but_names_kept(tmp_path, wiki):
    wiki.redirects[HASTI] = ["హస్తినాపురం లోని నగరం", "హస్తినాపురం పురము"]
    pack = gb.load_pack(write_pack(tmp_path, pack_for_build()))
    report = gb.build(pack, make_fetcher(tmp_path, wiki), tmp_path / "g", log=lambda m: None)
    terms = {r["term"] for r in read_rows(tmp_path / "g" / "aliases_candidates.csv") if r["entity_id"] == "Q6"}
    assert "హస్తినాపురం లోని నగరం" not in terms
    assert "హస్తినాపురం పురము" in terms and HASTI in terms
    assert report["counts"]["surfaces_dropped"]["phrase_redirect"] == 1


def test_other_is_always_exact_only_even_if_pack_omits_it(tmp_path):
    p = gb.load_pack(write_pack(tmp_path, pack_for_build(no_fuzzy_types=["concept"])))
    assert "other" in p["no_fuzzy_types"] and "concept" in p["no_fuzzy_types"]


# --------------------------------------------------------------------------------------
# Curated promotion (the default): labels, titles, aliases; no redirects; short and common forced exact
# --------------------------------------------------------------------------------------
def _cand(term, origin, decision="", entity="Q1", needs="0"):
    return {"term": term, "entity_id": entity, "roman": "R", "type": "person", "variants": "",
            "needs_context": needs, "source": "s", "notes": "", "origin": origin, "decision": decision}


def test_curated_takes_labels_and_aliases_but_not_redirects(tmp_path):
    rows = [
        _cand("కృష్ణుడు", "wikidata_alias_te", entity="Q42891"),
        _cand("అర్జునుడు", "wikipedia_title_te", entity="Q185790"),
        _cand("పార్థుడు", "wikipedia_redirect_te", entity="Q185790"),
    ]
    p = tmp_path / "c.csv"
    p.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))
    s = gb.promote(p, tmp_path / "a.csv", accept="curated")
    got = {r["term"] for r in read_rows(tmp_path / "a.csv")}
    assert got == {"కృష్ణుడు", "అర్జునుడు"}
    assert s["not_selected"] == 1


def test_curated_redirect_can_be_added_by_hand(tmp_path):
    rows = [_cand("పార్థుడు", "wikipedia_redirect_te", decision="accept", entity="Q185790")]
    p = tmp_path / "c.csv"
    p.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))
    gb.promote(p, tmp_path / "a.csv", accept="curated")
    assert {r["term"] for r in read_rows(tmp_path / "a.csv")} == {"పార్థుడు"}


def test_curated_forces_short_and_common_spellings_to_exact_only(tmp_path):
    rows = [
        _cand("కన్నడు", "wikidata_alias_te", entity="Q42891"),              # 6 chars, common word marked below
        _cand("కన్ను", "wikidata_alias_te", entity="Q99"),                   # 5 chars, fine as-is
        _cand("కృష్ణ", "wikidata_alias_te", entity="Q2", decision="common"),  # 4 chars, marked common
        _cand("అర్జున", "wikidata_alias_te", entity="Q3"),                   # 5 chars: ok
        _cand("కర్ణ", "wikidata_alias_te", entity="Q4", needs="0"),          # 4 chars: forced
    ]
    rows[0]["decision"] = "common"
    p = tmp_path / "c.csv"
    p.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))
    s = gb.promote(p, tmp_path / "a.csv", accept="curated")
    by = {r["term"]: r["needs_context"] for r in read_rows(tmp_path / "a.csv")}
    assert by["కన్నడు"] == "1" and by["కృష్ణ"] == "1" and by["కర్ణ"] == "1"
    assert by["కన్ను"] == "0" and by["అర్జున"] == "0"
    assert s["forced_exact_only"] == 3


def test_curated_still_skips_ambiguous_spellings(tmp_path):
    rows = [_cand("కృష్ణుడు", "wikidata_alias_te", entity="Q1"), _cand("కృష్ణుడు", "wikidata_alias_te", entity="Q2")]
    p = tmp_path / "c.csv"
    p.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))
    s = gb.promote(p, tmp_path / "a.csv", accept="curated")
    assert s["skipped_ambiguous"] == 2 and s["added"] == 0


# --------------------------------------------------------------------------------------
# Other languages: the builder follows set_language() (config.TARGET_LANGUAGE)
# --------------------------------------------------------------------------------------
@pytest.fixture
def hindi():
    gb.set_language("hi")
    yield
    gb.set_language("te")


def test_hindi_surfaces_keep_devanagari_and_drop_other_scripts(hindi):
    dropped = {"no_telugu_script": 0, "empty_after_cleaning": 0, "phrase_redirect": 0}
    e = {"qid": "Q185790", "te_label": "अर्जुन", "te_title": "अर्जुन", "te_aliases": ["Arjuna", "పార్థుడు", "पार्थ"]}
    out = gb._surfaces(e, {"अर्जुन": ["धनंजय"]}, dropped)
    by = {r["surface"]: r for r in out}
    assert set(by) == {"अर्जुन", "पार्थ", "धनंजय"}
    assert by["अर्जुन"]["origins"] == ["wikidata_label_hi", "wikipedia_title_hi"]
    assert "hiwiki:अर्जुन" in by["अर्जुन"]["sources"]
    assert by["धनंजय"]["origins"] == ["wikipedia_redirect_hi"]
    assert dropped["no_telugu_script"] == 2          # the Latin and the Telugu alias


def test_hindi_curated_promotion_excludes_redirects_and_checks_script(hindi, tmp_path):
    rows = [_cand("अर्जुन", "wikidata_label_hi", entity="Q185790"),
            _cand("धनंजय", "wikipedia_redirect_hi", entity="Q185790"),
            _cand("అర్జునుడు", "wikidata_label_hi", entity="Q185791")]
    p = tmp_path / "c.csv"
    p.write_bytes(gb._csv_bytes(gb.CANDIDATE_COLUMNS, rows, bom=True))
    s = gb.promote(p, tmp_path / "a.csv", accept="curated")
    assert {r["term"] for r in read_rows(tmp_path / "a.csv")} == {"अर्जुन"}
    assert s["not_selected"] == 1 and s["invalid"][0]["reason"] == "term has no characters in the hi script"


def test_pack_language_must_match_active_playlist(hindi, tmp_path):
    with pytest.raises(gb.PackError, match="active playlist is 'hi'"):
        gb.load_pack(write_pack(tmp_path, pack_for_build(language="te")))
    assert gb.load_pack(write_pack(tmp_path, pack_for_build(language="hi")))["language"] == "hi"


def test_unknown_language_is_refused():
    with pytest.raises(gb.PackError, match="Unsupported language"):
        gb.set_language("en")
    assert gb.LANG == "te"
