"""
Playlist registry: one entry per playlist, each with its own language.

playlists.json (project root) looks like:
  {"default": "mahabharata",
   "playlists": {"mahabharata": {"id": "PL...", "language": "te", "name": "Mahabharata pravachanam"},
                 "nptel-dsa":   {"id": "PL...", "language": "hi", "name": "NPTEL DSA (Hindi)"}}}

Everything that used to be Telugu-only (ASR language, caption languages, glossary folder) is read
from the selected entry, so adding a language means adding an entry, not editing code.
Stdlib only.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "playlists.json"

# Languages ai4bharat/indic-conformer-600m-multilingual accepts (its language_code argument).
ASR_LANGUAGES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili",
    "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

# Unicode script name prefix -> the one language it identifies on its own. Scripts shared by
# several languages (Devanagari, Bengali, Arabic) are deliberately absent: those need --lang.
_SCRIPT_TO_LANG = {"TELUGU": "te", "TAMIL": "ta", "KANNADA": "kn", "MALAYALAM": "ml",
                   "GUJARATI": "gu", "ORIYA": "or", "GURMUKHI": "pa", "OL CHIKI": "sat"}
_SHARED_SCRIPTS = {"DEVANAGARI", "BENGALI", "ARABIC", "MEETEI"}


class RegistryError(Exception):
    """Shown to the user as a plain message."""


def parse_playlist_id(text: str) -> str:
    """A playlist URL (any form with list=...) or a bare ID -> the ID."""
    text = text.strip()
    m = re.search(r"[?&]list=([\w-]+)", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w-]{10,}", text):
        return text
    raise RegistryError(f"Not a playlist URL or ID: {text!r}")


def load(path: Path | str | None = None) -> dict:
    path = Path(path or REGISTRY_PATH)
    if not path.exists():
        return {"default": None, "playlists": {}}
    with open(path, encoding="utf-8") as f:
        reg = json.load(f)
    reg.setdefault("playlists", {})
    reg.setdefault("default", None)
    return reg


def save(reg: dict, path: Path | str | None = None) -> None:
    path = Path(path or REGISTRY_PATH)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def check_language(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang not in ASR_LANGUAGES:
        raise RegistryError(
            f"Language {lang!r} is not supported by IndicConformer. Use one of: "
            + ", ".join(f"{k} ({v})" for k, v in sorted(ASR_LANGUAGES.items())))
    return lang


def resolve(key_or_url: str | None, path: Path | str | None = None) -> tuple[str, dict]:
    """Registry key, playlist URL or playlist ID -> (key, entry). None means the default entry."""
    reg = load(path)
    pls = reg["playlists"]
    if not key_or_url:
        key_or_url = reg.get("default")
        if not key_or_url:
            raise RegistryError("playlists.json has no default playlist. Add one with "
                                "`python run_pipeline.py --add-playlist URL --lang xx`.")
    if key_or_url in pls:
        return key_or_url, pls[key_or_url]
    try:
        pid = parse_playlist_id(key_or_url)
    except RegistryError:
        raise RegistryError(f"No playlist named {key_or_url!r} in playlists.json. "
                            f"Known: {', '.join(sorted(pls)) or 'none'}.") from None
    for k, e in pls.items():
        if e["id"] == pid:
            return k, e
    raise RegistryError(f"Playlist {pid} is not registered yet. Add it with "
                        f"`python run_pipeline.py --add-playlist <url> --lang xx`.")


def add(url: str, lang: str, key: str | None = None, name: str | None = None,
        make_default: bool = False, path: Path | str | None = None) -> tuple[str, dict]:
    pid = parse_playlist_id(url)
    lang = check_language(lang)
    reg = load(path)
    for k, e in reg["playlists"].items():
        if e["id"] == pid:
            if e["language"] != lang:
                raise RegistryError(f"{pid} is already registered as {k!r} with language "
                                    f"{e['language']!r}. Edit playlists.json to change it.")
            return k, e
    key = key or pid
    if key in reg["playlists"]:
        raise RegistryError(f"The name {key!r} is already used by another playlist. Pass --key.")
    entry = {"id": pid, "language": lang, "name": name or key}
    reg["playlists"][key] = entry
    if make_default or not reg.get("default"):
        reg["default"] = key
    save(reg, path)
    return key, entry


def guess_language(texts: list[str]) -> str | None:
    """Language from the script of titles, only when the script identifies one language."""
    counts: dict[str, int] = {}
    for t in texts:
        for ch in t or "":
            if not ch.isalpha() or ord(ch) < 0x0900:
                continue
            try:
                script = unicodedata.name(ch)
            except ValueError:
                continue
            for s in list(_SCRIPT_TO_LANG) + list(_SHARED_SCRIPTS):
                if script.startswith(s):
                    counts[s] = counts.get(s, 0) + 1
                    break
    if not counts:
        return None
    top = max(counts, key=counts.get)
    return _SCRIPT_TO_LANG.get(top)  # None for shared scripts: ambiguous


def spread_sample(ids: list[str], n: int) -> list[str]:
    """n IDs spread evenly across the list (first, last and evenly spaced in between)."""
    if n <= 0 or not ids:
        return []
    if n >= len(ids):
        return list(ids)
    if n == 1:
        return [ids[0]]
    step = (len(ids) - 1) / (n - 1)
    picked = []
    for i in range(n):
        v = ids[round(i * step)]
        if v not in picked:
            picked.append(v)
    return picked
