"""
glossary_builder.py - build a per-playlist term glossary from Wikipedia and Wikidata.

Why: the glossary must come from trusted, citable, free sources and differ per playlist.
Nothing here is written from memory. Every row records where it came from.

Three commands (run from the project root):

  python glossary_builder.py propose --topic "Mahabharata"
      Searches Wikipedia for categories about the topic and writes a draft domain pack
      (DOMAIN_PACK_PATH). Every category starts with use:false. You edit it by hand.

  python glossary_builder.py build
      Reads the domain pack, harvests the chosen categories, and writes four files into
      GLOSSARY_DIR: aliases_candidates.csv, missing_telugu.csv, ambiguous.csv,
      build_report.json. Nothing is added to the live alias table yet.

  python glossary_builder.py promote --accept label
      Copies reviewed candidate rows into ALIAS_TABLE_PATH (aliases.csv). Existing rows are
      never changed or removed.

Needs internet and a real contact in GLOSSARY_USER_AGENT (Wikimedia policy). The builder
refuses to run while the placeholder is still there. All API replies are cached on disk,
so a rerun does not hit the network again.

Any Indic language: the target language comes from config.TARGET_LANGUAGE (the active playlist in
playlists.json) through set_language(). Names like "te_label" and "missing_telugu.csv" are kept from the
Telugu-only version; they mean "target language".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

# --- Contract with the corrector -----------------------------------------------------
ALIAS_COLUMNS = ["term", "entity_id", "roman", "type", "variants", "needs_context", "source", "notes"]
CANDIDATE_COLUMNS = ALIAS_COLUMNS + ["origin", "decision"]

SHORT_TERM_CHARS = 5  # spellings shorter than this are exact-match only in curated mode

# Unicode block(s) for each language's script. A spelling must contain at least one character
# from its language's block, which drops Latin aliases and other-script noise.
SCRIPT_RANGES = {
    "hi": "\u0900-\u097F", "mr": "\u0900-\u097F", "ne": "\u0900-\u097F", "sa": "\u0900-\u097F",
    "mai": "\u0900-\u097F", "kok": "\u0900-\u097F", "doi": "\u0900-\u097F", "brx": "\u0900-\u097F",
    "bn": "\u0980-\u09FF", "as": "\u0980-\u09FF", "pa": "\u0A00-\u0A7F", "gu": "\u0A80-\u0AFF",
    "or": "\u0B00-\u0B7F", "ta": "\u0B80-\u0BFF", "te": "\u0C00-\u0C7F", "kn": "\u0C80-\u0CFF",
    "ml": "\u0D00-\u0D7F", "ur": "\u0600-\u06FF\u0750-\u077F", "ks": "\u0600-\u06FF\u0750-\u077F",
    "sd": "\u0600-\u06FF\u0750-\u077F", "sat": "\u1C50-\u1C7F", "mni": "\uABC0-\uABFF\u0980-\u09FF",
}
# Words that mark a descriptive redirect ("X of Y") rather than a name. Optional per language;
# every language also drops redirects longer than 6 words.
PHRASE_MARKERS = {"te": ["లోని"]}


def set_language(lang: str) -> None:
    """Point every language-dependent setting at one language code (te, hi, ta, ...)."""
    global LANG, ORIGIN_PRIORITY, LABEL_ORIGINS, REDIRECT_ORIGIN, TELUGU_RE
    if lang not in SCRIPT_RANGES:
        raise PackError(f"Unsupported language {lang!r}. Known: {', '.join(sorted(SCRIPT_RANGES))}.")
    LANG = lang
    # Where a target-language spelling came from, best first. Label and title are the most trustworthy.
    ORIGIN_PRIORITY = [f"wikidata_label_{lang}", f"wikipedia_title_{lang}",
                       f"wikidata_alias_{lang}", f"wikipedia_redirect_{lang}"]
    LABEL_ORIGINS = set(ORIGIN_PRIORITY[:2])
    REDIRECT_ORIGIN = ORIGIN_PRIORITY[3]
    TELUGU_RE = re.compile(f"[{SCRIPT_RANGES[lang]}]")  # name kept; it is the target-script check


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
LANG = "te"
BATCH = 50  # Wikimedia API limit for titles= and ids= per request

LICENCES = {
    "wikidata": "CC0 1.0: labels, aliases and descriptions",
    "wikipedia": "CC BY-SA 4.0: page titles and redirects; attribution is the Wikipedia page "
    "named in each row's source column",
}

QID_RE = re.compile(r"^Q\d+$")
_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


class FetchError(RuntimeError):
    """Network or API failure that retrying will not fix."""


class PackError(ValueError):
    """The domain pack is missing or wrong."""


class PromoteError(ValueError):
    """The alias table cannot be safely extended."""


class GlossaryMissingError(RuntimeError):
    """The alias table does not exist yet (or is empty). Call require_alias_table()."""


# ======================================================================================
# Small helpers
# ======================================================================================
def _chunks(seq: Iterable, n: int) -> Iterator[list]:
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _as_dict(x) -> dict:
    """Wikibase sends an empty map as [] instead of {}."""
    return x if isinstance(x, dict) else {}


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _csv_bytes(columns: list[str], rows: list[dict], bom: bool, eol: str = "\n") -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator=eol, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8-sig" if bom else "utf-8")


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def clean_surface(s: str | None) -> str:
    """NFC, single spaces, and trailing '(...)' qualifiers removed (e.g. 'Bhishma (epic)')."""
    s = unicodedata.normalize("NFC", s or "").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    while True:
        t = _PAREN_RE.sub("", s).strip()
        if t == s:
            return s
        s = t


def check_user_agent(ua: str | None) -> str:
    ua = (ua or "").strip()
    low = ua.lower()
    has_contact = re.search(r"[^\s@()<>]+@[^\s@()<>]+\.[^\s@()<>]+", ua) or re.search(r"https?://\S+", ua)
    markers = ("example.com", "example.org", "example.net", "change_me", "changeme",
               "your-email", "your_email", "yourname", "you@", "email@here", "put-a-real")
    if not has_contact or any(m in low for m in markers):
        raise FetchError(
            "GLOSSARY_USER_AGENT needs a real contact (an email or a URL). Wikimedia asks every "
            "script to identify itself. Edit it in config.py, e.g. "
            "'PravachanamRAG/0.1 (you@yourcollege.edu)'."
        )
    return ua


def require_alias_table(path: str | Path) -> Path:
    """For the corrector: fail with a clear message when no glossary has been built yet."""
    p = Path(path)
    hint = (
        "Build the glossary first:\n"
        "  python glossary_builder.py propose --topic <your topic>\n"
        "  (edit the domain pack: set use:true and a type on the right categories)\n"
        "  python glossary_builder.py build\n"
        "  python glossary_builder.py promote --accept label"
    )
    if not p.exists():
        raise GlossaryMissingError(f"No alias table at {p}.\n{hint}")
    _, rows = _read_csv(p)
    if not rows:
        raise GlossaryMissingError(f"The alias table at {p} has a header but no rows.\n{hint}")
    return p


# ======================================================================================
# Fetcher: user agent, disk cache, retries
# ======================================================================================
Transport = Callable[[str, dict, dict], "tuple[int, str, dict]"]


def urllib_transport(url: str, params: dict, headers: dict, timeout: float = 30) -> tuple[int, str, dict]:
    """Real network call. Returns (status, body, lowercase response headers)."""
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8"), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, body, {k.lower(): v for k, v in e.headers.items()}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 599, str(e), {}  # treated like a server error: retried


class Fetcher:
    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_retries: int = 5,
        backoff_s: float = 2.0,
        min_interval_s: float = 0.15,
    ):
        self.user_agent = check_user_agent(user_agent)
        self.cache_dir = Path(cache_dir)
        self.transport = transport or urllib_transport
        self.sleep, self.clock = sleep, clock
        self.max_retries, self.backoff_s, self.min_interval_s = max_retries, backoff_s, min_interval_s
        self._last = float("-inf")
        self.stats = {"network": 0, "cache": 0, "retries": 0}

    # -- cache --
    def _cache_path(self, url: str, params: dict) -> Path:
        key = url + "?" + urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def get_json(self, url: str, params: dict) -> dict:
        path = self._cache_path(url, params)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.stats["cache"] += 1
                return data
            except json.JSONDecodeError:
                pass  # corrupt cache file: fetch again and overwrite
        data = self._fetch(url, params)
        _atomic_write(path, json.dumps(data, ensure_ascii=False).encode("utf-8"))
        return data

    # -- network --
    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self._last + self.min_interval_s - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last = self.clock()

    def _wait_time(self, attempt: int, headers: dict) -> float:
        try:
            return min(float(headers.get("retry-after", "")), 60.0)
        except ValueError:
            return min(self.backoff_s * (2**attempt), 60.0)

    def _fetch(self, url: str, params: dict) -> dict:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            status, body, resp_headers = self.transport(url, params, headers)
            self.stats["network"] += 1
            if status == 200:
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    raise FetchError(f"Reply from {url} was not JSON: {e}") from e
                err = data.get("error") if isinstance(data, dict) else None
                if not err:
                    return data
                code = err.get("code", "unknown") if isinstance(err, dict) else str(err)
                if code not in ("maxlag", "ratelimited"):
                    raise FetchError(f"API error from {url}: {err}")
                last = f"API error {code}"
            elif status == 429 or status >= 500:
                last = f"HTTP {status}"
            else:
                raise FetchError(f"HTTP {status} from {url} params={params}")
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                self.sleep(self._wait_time(attempt, resp_headers))
        raise FetchError(f"Gave up after {self.max_retries + 1} attempts ({last}): {url}")


# ======================================================================================
# Wikipedia / Wikidata calls
# ======================================================================================
def wiki_api(wiki: str) -> str:
    return f"https://{wiki}.wikipedia.org/w/api.php"


def _base(**kw) -> dict:
    return {"format": "json", "formatversion": "2", **{k: str(v) for k, v in kw.items()}}


def _follow(fetcher: Fetcher, url: str, params: dict, max_requests: int = 5000) -> Iterator[dict]:
    """Yield every reply of a query, following the API's 'continue' token."""
    params = dict(params)
    for _ in range(max_requests):
        data = fetcher.get_json(url, params)
        yield data
        cont = data.get("continue")
        if not cont:
            return
        params = {**params, **{k: str(v) for k, v in cont.items()}}
    raise FetchError(f"Too many continuation requests for {url}")


def search_categories(fetcher: Fetcher, wiki: str, topic: str, limit: int = 50) -> list[str]:
    params = _base(action="query", list="search", srsearch=topic, srnamespace=14, srlimit=limit)
    data = fetcher.get_json(wiki_api(wiki), params)
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def category_sizes(fetcher: Fetcher, wiki: str, titles: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for batch in _chunks(titles, BATCH):
        params = _base(action="query", prop="categoryinfo", titles="|".join(batch))
        for data in _follow(fetcher, wiki_api(wiki), params):
            for page in data.get("query", {}).get("pages", []):
                info = page.get("categoryinfo") or {}
                out[page["title"]] = {"pages": info.get("pages", 0), "subcats": info.get("subcats", 0)}
    return out


def walk_category(fetcher: Fetcher, wiki: str, title: str, depth: int, limit: int) -> tuple[dict, dict]:
    """
    Breadth-first walk of a category. Returns ({page_title: category_it_was_found_in}, info).
    Follows subcategories down to `depth`, never revisits a category, stops at `limit` pages.
    """
    if wiki == "en" and ":" not in title:
        title = "Category:" + title
    url = wiki_api(wiki)
    seen = {title}
    queue = deque([(title, 0)])
    pages: dict[str, str] = {}
    truncated = False
    walked = 0
    while queue and not truncated:
        cat, d = queue.popleft()
        walked += 1
        params = _base(action="query", list="categorymembers", cmtitle=cat, cmlimit=500, cmtype="page|subcat")
        for data in _follow(fetcher, url, params):
            for m in data.get("query", {}).get("categorymembers", []):
                if m.get("ns") == 14:
                    if d < depth and m["title"] not in seen:
                        seen.add(m["title"])
                        queue.append((m["title"], d + 1))
                elif m.get("ns") == 0 and m["title"] not in pages:
                    if len(pages) >= limit:
                        truncated = True
                        break
                    pages[m["title"]] = cat
            if truncated:
                break
    return pages, {"categories_walked": walked, "truncated": truncated}


def lookup_pages(fetcher: Fetcher, wiki: str, titles: list[str]) -> dict[str, dict]:
    """For each page: its Wikidata item and its Telugu Wikipedia title (when they exist)."""
    out: dict[str, dict] = {}
    for batch in _chunks(titles, BATCH):
        kw = dict(action="query", prop="pageprops", ppprop="wikibase_item", titles="|".join(batch))
        if wiki != LANG:
            kw.update(prop="pageprops|langlinks", lllang=LANG, lllimit=500)
        for data in _follow(fetcher, wiki_api(wiki), _base(**kw)):
            for page in data.get("query", {}).get("pages", []):
                rec = out.setdefault(page["title"], {"qid": None, "te_title": None})
                qid = (page.get("pageprops") or {}).get("wikibase_item")
                if qid:
                    rec["qid"] = qid
                links = page.get("langlinks") or []
                if links and links[0].get("title"):
                    rec["te_title"] = links[0]["title"]
        if wiki == LANG:
            for t, rec in out.items():
                rec["te_title"] = t
    return out


def fetch_entities(fetcher: Fetcher, qids: list[str]) -> dict[str, dict | None]:
    """
    Telugu and English labels, Telugu aliases, English description. Keyed by every id we asked
    for; merged (redirected) items resolve to the canonical id in the 'id' field. None = missing.
    """
    out: dict[str, dict | None] = {}
    for batch in _chunks(qids, BATCH):
        params = _base(action="wbgetentities", ids="|".join(batch), props="labels|aliases|descriptions|sitelinks", sitefilter=f"{LANG}wiki",
                       languages=f"{LANG}|en")
        data = fetcher.get_json(WIKIDATA_API, params)
        for key, ent in _as_dict(data.get("entities")).items():
            if "missing" in ent:
                out[key] = None
                continue
            labels, aliases, descs = _as_dict(ent.get("labels")), _as_dict(ent.get("aliases")), _as_dict(ent.get("descriptions"))
            detail = {
                "id": ent.get("id", key),
                "te_label": _as_dict(labels.get(LANG)).get("value"),
                "en_label": _as_dict(labels.get("en")).get("value"),
                "te_aliases": [a.get("value") for a in aliases.get(LANG, []) if a.get("value")],
                "en_description": _as_dict(descs.get("en")).get("value"),
                "te_sitelink": _as_dict(_as_dict(ent.get("sitelinks")).get(f"{LANG}wiki")).get("title"),
            }
            out[key] = detail
            out[detail["id"]] = detail
            red = _as_dict(ent.get("redirects"))
            if red.get("from"):
                out[red["from"]] = detail
    return out


def fetch_te_redirects(fetcher: Fetcher, te_titles: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for batch in _chunks(te_titles, BATCH):
        params = _base(action="query", prop="redirects", rdlimit=500, rdnamespace=0, titles="|".join(batch))
        for data in _follow(fetcher, wiki_api(LANG), params):
            for page in data.get("query", {}).get("pages", []):
                for r in page.get("redirects", []) or []:
                    if r.get("title") and r["title"] not in out[page["title"]]:
                        out[page["title"]].append(r["title"])
    return dict(out)


# ======================================================================================
# Domain pack
# ======================================================================================
def load_pack(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise PackError(f"No domain pack at {p}. Run 'propose' first, or create it by hand.")
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise PackError(f"Domain pack {p} is not valid JSON: {e}") from e
    if raw.get("language", LANG) != LANG:
        raise PackError(f"The pack is for language {raw.get('language')!r} but the active playlist is "
                        f"{LANG!r}. Select the right playlist (PLAYLIST=<name>) or fix the pack.")
    cats = []
    for i, c in enumerate(raw.get("categories", [])):
        if not c.get("use", False):
            continue
        for k in ("wiki", "title", "type"):
            if not str(c.get(k, "")).strip():
                raise PackError(f"categories[{i}] is marked use:true but has no '{k}'. Fill it in.")
        depth = int(c.get("depth", 1))
        if depth < 0:
            raise PackError(f"categories[{i}] has a negative depth.")
        cats.append({"wiki": c["wiki"].strip(), "title": c["title"].strip(), "type": c["type"].strip(), "depth": depth})
    ids = []
    for i, w in enumerate(raw.get("wikidata_ids", [])):
        qid, typ = str(w.get("id", "")).strip(), str(w.get("type", "")).strip()
        if not QID_RE.match(qid) or not typ:
            raise PackError(f"wikidata_ids[{i}] needs an id like Q123 and a type.")
        ids.append({"id": qid, "type": typ})
    if not cats and not ids:
        raise PackError("Nothing to harvest: no category has use:true and wikidata_ids is empty.")
    return {
        "topic": raw.get("topic", ""),
        "language": LANG,
        "categories": cats,
        "wikidata_ids": ids,
        "no_fuzzy_types": sorted(set(raw.get("no_fuzzy_types", ["concept", "other"])) | {"other"}),  # "other" is always exact-only
        "max_pages": int(raw.get("max_pages", 2000)),
    }


def propose(fetcher: Fetcher, topic: str, pack_path: str | Path, wiki: str = "en",
            playlist_id: str = "", force: bool = False) -> dict:
    """Write a draft domain pack. Refuses to overwrite one you may have edited."""
    pack_path = Path(pack_path)
    if pack_path.exists() and not force:
        raise PackError(f"{pack_path} already exists. Use --force to overwrite it (your edits would be lost).")
    titles = search_categories(fetcher, wiki, topic)
    sizes = category_sizes(fetcher, wiki, titles) if titles else {}
    pack = {
        "_help": "Set use:true and a type (person, place, text, concept, ...) on the categories that "
                 "match your playlist. depth = how many subcategory levels to follow. Delete the rest.",
        "topic": topic,
        "language": LANG,
        "playlist_id": playlist_id,
        "categories": [
            {"wiki": wiki, "title": t, "type": "", "depth": 1, "use": False,
             "_pages": sizes.get(t, {}).get("pages", 0), "_subcats": sizes.get(t, {}).get("subcats", 0)}
            for t in titles
        ],
        "wikidata_ids": [],
        "no_fuzzy_types": ["concept", "other"],
        "max_pages": 2000,
    }
    _atomic_write(pack_path, (json.dumps(pack, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return pack


# ======================================================================================
# Harvest and build
# ======================================================================================
def _add_entity(entities: dict, qid: str, cat: dict, en_page: str | None, te_title: str | None) -> None:
    e = entities.setdefault(qid, {"qid": qid, "type": cat["type"], "types_seen": [], "categories": [],
                                  "en_title": None, "te_title": None})
    if cat["type"] not in e["types_seen"]:
        e["types_seen"].append(cat["type"])
    label = f'{cat["wiki"]}:{cat["title"]}'
    if label not in e["categories"]:
        e["categories"].append(label)
    if cat["wiki"] == "en" and en_page and not e["en_title"]:
        e["en_title"] = en_page
    if te_title and not e["te_title"]:
        e["te_title"] = te_title


def harvest(fetcher: Fetcher, pack: dict, log: Callable[[str], None] = print) -> dict:
    entities: dict[str, dict] = {}
    cat_stats, no_wikidata = [], []
    for cat in pack["categories"]:
        log(f'Walking {cat["wiki"]}:{cat["title"]} (depth {cat["depth"]}) ...')
        pages, info = walk_category(fetcher, cat["wiki"], cat["title"], cat["depth"], pack["max_pages"])
        looked = lookup_pages(fetcher, cat["wiki"], list(pages))
        with_item = 0
        for title in pages:
            rec = looked.get(title) or {}
            if not rec.get("qid"):
                no_wikidata.append({"wiki": cat["wiki"], "title": title, "category": pages[title]})
                continue
            with_item += 1
            _add_entity(entities, rec["qid"], cat, title, rec.get("te_title"))
        cat_stats.append({"wiki": cat["wiki"], "title": cat["title"], "type": cat["type"], "depth": cat["depth"],
                          "pages_found": len(pages), "pages_with_wikidata": with_item,
                          "categories_walked": info["categories_walked"], "truncated": info["truncated"]})
        log(f"  {len(pages)} pages, {with_item} with a Wikidata item" + ("  (HIT max_pages)" if info["truncated"] else ""))
    for w in pack["wikidata_ids"]:
        _add_entity(entities, w["id"], {"wiki": "wikidata", "title": "wikidata_ids", "type": w["type"]}, None, None)

    log(f"Fetching {len(entities)} Wikidata entities ...")
    details = fetch_entities(fetcher, sorted(entities))

    merged: dict[str, dict] = {}
    unresolved = []
    rejected_links = []
    for qid in sorted(entities):
        d = details.get(qid)
        if d is None:
            unresolved.append(qid)
            continue
        e, cid = entities[qid], d["id"]
        m = merged.setdefault(cid, {**d, "qid": cid, "type": e["type"], "types_seen": [], "categories": [],
                                    "en_title": None, "te_title": None, "requested_ids": []})
        m["requested_ids"].append(qid)
        for t in e["types_seen"]:
            if t not in m["types_seen"]:
                m["types_seen"].append(t)
        for c in e["categories"]:
            if c not in m["categories"]:
                m["categories"].append(c)
        m["en_title"] = m["en_title"] or e["en_title"]
        m["te_title"] = m["te_title"] or e["te_title"]
        # The English page claims a Telugu page. Wikidata must agree that the Telugu page is this
        # same item; otherwise the interlanguage link is wrong (e.g. five parvas sharing one page).
        if m["te_title"] and m.get("te_sitelink") != m["te_title"]:
            rejected_links.append({"entity_id": cid, "en_title": m["en_title"],
                                   "claimed_te_title": m["te_title"], "wikidata_tewiki": m.get("te_sitelink")})
            m["te_title"] = None
        # One type per entity. If categories disagree, use exact-only "other" and keep both in notes.
        m["type"] = m["types_seen"][0] if len(m["types_seen"]) == 1 else "other"

    te_titles = sorted({m["te_title"] for m in merged.values() if m["te_title"]})
    log(f"Fetching Telugu Wikipedia redirects for {len(te_titles)} pages ...")
    redirects = fetch_te_redirects(fetcher, te_titles) if te_titles else {}
    return {"entities": merged, "redirects": redirects, "category_stats": cat_stats,
            "no_wikidata": no_wikidata, "wikidata_unresolved": unresolved, "rejected_te_links": rejected_links}


def _is_phrase(s: str) -> bool:
    """A redirect that describes an entity ('X లోని Y' = 'Y of X') rather than names it."""
    return any(m in s for m in PHRASE_MARKERS.get(LANG, [])) or len(s.split()) > 6


def _surfaces(e: dict, redirects: dict, dropped: dict) -> list[dict]:
    """All Telugu spellings of one entity, deduplicated, best origin first."""
    raw: list[tuple[str, str, str]] = []  # (text, origin, source)
    if e.get("te_label"):
        raw.append((e["te_label"], ORIGIN_PRIORITY[0], f'wikidata:{e["qid"]}/label'))
    if e.get("te_title"):
        raw.append((e["te_title"], ORIGIN_PRIORITY[1], f'{LANG}wiki:{e["te_title"]}'))
    for a in e.get("te_aliases", []):
        raw.append((a, ORIGIN_PRIORITY[2], f'wikidata:{e["qid"]}/alias'))
    for r in redirects.get(e.get("te_title") or "", []):
        if _is_phrase(r):
            dropped["phrase_redirect"] += 1
            continue
        raw.append((r, REDIRECT_ORIGIN, f"{LANG}wiki-redirect:{r}"))

    by_surface: dict[str, dict] = {}
    for text, origin, source in raw:
        s = clean_surface(text)
        if not s:
            dropped["empty_after_cleaning"] += 1
            continue
        if not TELUGU_RE.search(s):
            dropped["no_telugu_script"] += 1
            continue
        rec = by_surface.setdefault(s, {"surface": s, "origins": [], "sources": []})
        if origin not in rec["origins"]:
            rec["origins"].append(origin)
        if source not in rec["sources"]:
            rec["sources"].append(source)
    out = []
    for rec in by_surface.values():
        rec["origins"].sort(key=ORIGIN_PRIORITY.index)
        out.append(rec)
    out.sort(key=lambda r: (ORIGIN_PRIORITY.index(r["origins"][0]), r["surface"]))
    return out


def build(pack: dict, fetcher: Fetcher, out_dir: str | Path, log: Callable[[str], None] = print) -> dict:
    out_dir = Path(out_dir)
    h = harvest(fetcher, pack, log)
    dropped = {"no_telugu_script": 0, "empty_after_cleaning": 0, "phrase_redirect": 0}
    redirects = h["redirects"]

    per_entity: dict[str, list[dict]] = {}
    for qid, e in h["entities"].items():
        per_entity[qid] = _surfaces(e, redirects, dropped)

    owners: dict[str, set] = defaultdict(set)
    for qid, surfs in per_entity.items():
        for s in surfs:
            owners[s["surface"]].add(qid)
    ambiguous_terms = {t for t, q in owners.items() if len(q) > 1}

    rows, missing = [], []
    for qid in sorted(h["entities"], key=lambda q: (h["entities"][q]["type"], int(q[1:]))):
        e = h["entities"][qid]
        roman = e.get("en_label") or e.get("en_title") or ""
        surfs = per_entity[qid]
        if not surfs:
            missing.append({"entity_id": qid, "en_label": roman, "en_description": e.get("en_description") or "",
                            "type": e["type"], "categories": " | ".join(e["categories"]), "en_title": e.get("en_title") or ""})
            continue
        for s in surfs:
            term = s["surface"]
            notes = []
            if e.get("en_description"):
                notes.append(e["en_description"])
            if " " in term:
                notes.append("multiword")
            if term in ambiguous_terms:
                others = sorted(owners[term] - {qid}, key=lambda q: int(q[1:]))
                notes.append("ambiguous with " + ",".join(others))
            if len(s["origins"]) > 1:
                notes.append("also: " + ",".join(s["origins"][1:]))
            if len(e["types_seen"]) > 1:
                notes.append("type conflict: " + "/".join(e["types_seen"]))
            rows.append({
                "term": term, "entity_id": qid, "roman": roman, "type": e["type"], "variants": "",
                "needs_context": 1 if (e["type"] in pack["no_fuzzy_types"] or term in ambiguous_terms) else 0,
                "source": ";".join(s["sources"]), "notes": " | ".join(notes),
                "origin": s["origins"][0], "decision": "",
            })

    ambiguous_rows = []
    for term in sorted(ambiguous_terms):
        qids = sorted(owners[term], key=lambda q: int(q[1:]))
        ambiguous_rows.append({
            "term": term, "entity_ids": " ".join(qids),
            "en_labels": " | ".join(h["entities"][q].get("en_label") or h["entities"][q].get("en_title") or "" for q in qids),
            "types": " | ".join(h["entities"][q]["type"] for q in qids),
        })

    _atomic_write(out_dir / "aliases_candidates.csv", _csv_bytes(CANDIDATE_COLUMNS, rows, bom=True))
    _atomic_write(out_dir / "missing_telugu.csv", _csv_bytes(
        ["entity_id", "en_label", "en_description", "type", "categories", "en_title"], missing, bom=True))
    _atomic_write(out_dir / "ambiguous.csv", _csv_bytes(
        ["term", "entity_ids", "en_labels", "types"], ambiguous_rows, bom=True))

    report = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": pack["topic"],
        "pack_sha256": hashlib.sha256(json.dumps(pack, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "categories": h["category_stats"],
        "counts": {
            "entities": len(h["entities"]),
            "entities_with_telugu": len(h["entities"]) - len(missing),
            "entities_missing_telugu": len(missing),
            "candidate_rows": len(rows),
            "ambiguous_terms": len(ambiguous_rows),
            "pages_without_wikidata": len(h["no_wikidata"]),
            "wikidata_items_not_returned": len(h["wikidata_unresolved"]),
            "surfaces_dropped": dropped,
        },
        "pages_without_wikidata": h["no_wikidata"],
        "wikidata_items_not_returned": h["wikidata_unresolved"],
        "rejected_te_links": h["rejected_te_links"],
        "api_requests": dict(fetcher.stats),
        "licences": LICENCES,
        "files": ["aliases_candidates.csv", "missing_telugu.csv", "ambiguous.csv", "build_report.json"],
    }
    _atomic_write(out_dir / "build_report.json", (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


# ======================================================================================
# Promote
# ======================================================================================
def _invalid_reason(row: dict) -> str | None:
    term = row["term"]
    if not term:
        return "empty term"
    if not TELUGU_RE.search(term):
        return f"term has no characters in the {LANG} script"
    if not QID_RE.match(row["entity_id"]):
        return "entity_id is not like Q123"
    if not row["type"]:
        return "type is empty"
    if row["needs_context"] not in ("0", "1"):
        return "needs_context must be 0 or 1"
    return None


def promote(candidates_path: str | Path, aliases_path: str | Path, accept: str = "label") -> dict:
    """
    Append reviewed candidate rows to the live alias table. Existing rows are never changed.

    accept = label    only rows whose spelling is a Wikidata label or Telugu Wikipedia title
             decided  only rows whose 'decision' column says accept (you reviewed them)
             all      every row except rejected ones
    Rows with decision=reject are never promoted. Ambiguous spellings (shared by several
    entities) are promoted only in 'decided' mode. A spelling already in the table under a
    different entity is reported as a conflict and skipped.
    """
    if accept not in ("curated", "label", "decided", "all"):
        raise PromoteError("accept must be curated, label, decided or all")
    cols, cands = _read_csv(Path(candidates_path))
    missing_cols = [c for c in CANDIDATE_COLUMNS if c not in cols]
    if missing_cols:
        raise PromoteError(f"Candidates file is missing columns: {missing_cols}")

    def norm(r: dict) -> dict:
        n = {c: (r.get(c) or "").strip() for c in CANDIDATE_COLUMNS}
        n["term"] = unicodedata.normalize("NFC", n["term"])
        return n

    cands = [norm(r) for r in cands]
    owners: dict[str, set] = defaultdict(set)
    for r in cands:
        owners[r["term"]].add(r["entity_id"])
    ambiguous = {t for t, q in owners.items() if len(q) > 1}

    aliases_path = Path(aliases_path)
    original = b""
    existing_pairs: set = set()
    existing_terms: dict[str, set] = defaultdict(set)
    eol = "\n"
    if aliases_path.exists():
        original = aliases_path.read_bytes()
        eol = "\r\n" if b"\r\n" in original else "\n"
        reader = csv.DictReader(io.StringIO(original.decode("utf-8-sig"), newline=""))
        if list(reader.fieldnames or []) != ALIAS_COLUMNS:
            raise PromoteError(f"{aliases_path} has columns {reader.fieldnames}, expected {ALIAS_COLUMNS}. "
                               "Not touching it.")
        for r in reader:
            t = unicodedata.normalize("NFC", (r["term"] or "").strip())
            existing_pairs.add((t, (r["entity_id"] or "").strip()))
            existing_terms[t].add((r["entity_id"] or "").strip())

    summary = {"accept": accept, "added": 0, "already_present": 0, "rejected_by_you": 0,
               "not_selected": 0, "skipped_ambiguous": 0, "conflicts": [], "invalid": []}
    new_rows, seen_new = [], set()
    for i, r in enumerate(cands, start=2):  # start=2: line 1 is the header
        decision = r["decision"].lower()
        if decision == "reject":
            summary["rejected_by_you"] += 1
            continue
        if accept == "curated":
            # Labels, titles and aliases. Redirects are the noisiest source, so only if you accepted them.
            selected = not r["origin"].startswith("wikipedia_redirect_") or decision == "accept"
        else:
            selected = {"label": r["origin"] in LABEL_ORIGINS, "decided": decision == "accept",
                        "all": True}[accept]
        if not selected:
            summary["not_selected"] += 1
            continue
        if accept != "decided" and r["term"] in ambiguous:
            summary["skipped_ambiguous"] += 1
            continue
        if accept == "curated" and (decision == "common" or len(r["term"]) < SHORT_TERM_CHARS):
            # Common words and very short spellings are exact-match only: fuzzy matching would
            # turn ordinary words into names.
            r["needs_context"] = "1"
            summary["forced_exact_only"] = summary.get("forced_exact_only", 0) + 1
        bad = _invalid_reason(r)
        if bad:
            summary["invalid"].append({"line": i, "term": r["term"], "reason": bad})
            continue
        pair = (r["term"], r["entity_id"])
        if pair in existing_pairs or pair in seen_new:
            summary["already_present"] += 1
            continue
        others = existing_terms.get(r["term"], set()) - {r["entity_id"]}
        if others:
            summary["conflicts"].append({"line": i, "term": r["term"], "new_entity": r["entity_id"],
                                         "existing_entities": sorted(others)})
            continue
        seen_new.add(pair)
        new_rows.append({c: r[c] for c in ALIAS_COLUMNS})

    summary["added"] = len(new_rows)
    if not new_rows:
        return summary
    if original:
        tail = _csv_bytes(ALIAS_COLUMNS, new_rows, bom=False, eol=eol).split(eol.encode(), 1)[1]  # drop the header line
        glue = b"" if original.endswith(b"\n") else eol.encode()
        data = original + glue + tail
    else:
        data = _csv_bytes(ALIAS_COLUMNS, new_rows, bom=False, eol=eol)
    _atomic_write(aliases_path, data)
    return summary


set_language(LANG)  # Telugu until config says otherwise


# ======================================================================================
# Command line
# ======================================================================================
def _load_config():
    try:
        import config
        for name in ("GLOSSARY_DIR", "DOMAIN_PACK_PATH", "GLOSSARY_USER_AGENT", "ALIAS_TABLE_PATH", "PLAYLIST_ID"):
            getattr(config, name)
        set_language(getattr(config, "TARGET_LANGUAGE", "te"))
        return config
    except ImportError as e:
        raise SystemExit(f"Cannot import config.py ({e}). Run from the project root with the venv active.")
    except AttributeError as e:
        raise SystemExit(f"config.py is missing a glossary setting ({e}). Add the lines from config_additions.py.")


def _fetcher(cfg, out_dir: Path) -> Fetcher:
    return Fetcher(cfg.GLOSSARY_USER_AGENT, out_dir / "_cache")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a per-playlist glossary from Wikipedia and Wikidata.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("propose", help="search Wikipedia for categories and write a draft domain pack")
    p1.add_argument("--topic", required=True)
    p1.add_argument("--wiki", default="en", help="Wikipedia language to search (default en)")
    p1.add_argument("--pack", help="where to write the pack (default DOMAIN_PACK_PATH)")
    p1.add_argument("--force", action="store_true", help="overwrite an existing pack")
    p2 = sub.add_parser("build", help="harvest the pack's categories into candidate files")
    p2.add_argument("--pack")
    p2.add_argument("--out", help="output folder (default GLOSSARY_DIR)")
    p3 = sub.add_parser("promote", help="append reviewed candidates to the live alias table")
    p3.add_argument("--accept", choices=["curated", "label", "decided", "all"], default="curated")
    p3.add_argument("--dir", help="folder with aliases_candidates.csv (default GLOSSARY_DIR)")
    p3.add_argument("--aliases", help="alias table to extend (default ALIAS_TABLE_PATH)")
    args = ap.parse_args(argv)

    cfg = _load_config()
    out_dir = Path(getattr(args, "out", None) or getattr(args, "dir", None) or cfg.GLOSSARY_DIR)
    try:
        if args.cmd == "propose":
            pack_path = Path(args.pack or cfg.DOMAIN_PACK_PATH)
            pack = propose(_fetcher(cfg, out_dir), args.topic, pack_path, wiki=args.wiki,
                           playlist_id=cfg.PLAYLIST_ID, force=args.force)
            print(f"Wrote {pack_path} with {len(pack['categories'])} candidate categories.")
            print("Open it, set use:true and a type on the categories that fit, delete the rest, then run 'build'.")
        elif args.cmd == "build":
            pack = load_pack(args.pack or cfg.DOMAIN_PACK_PATH)
            report = build(pack, _fetcher(cfg, out_dir), out_dir)
            print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
            print(f"Files are in {out_dir}. Review aliases_candidates.csv and missing_telugu.csv, then 'promote'.")
        else:
            summary = promote(out_dir / "aliases_candidates.csv", args.aliases or cfg.ALIAS_TABLE_PATH, args.accept)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (FetchError, PackError, PromoteError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
