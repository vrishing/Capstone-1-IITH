"""
One romanization and one phonetic key for every Brahmic Indic script, plus Latin input.

    romanize("కుంతి")      -> "kunti"          readable, keeps length and retroflex (capitals)
    phonetic_key("కుంతి")  -> "kunt"           coarse key for matching
    phonetic_key("कुंती")  -> "kunt"           same key from Devanagari
    phonetic_key("Kuntī")  -> "kunt"           and from English with or without diacritics

The nine Brahmic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) are laid out in parallel: the same offset from the block start is the same
sound. So one table covers all of them. Perso-Arabic (Urdu, Sindhi, Kashmiri), Ol Chiki and
Meetei Mayek are not handled; their text passes through lowercased.

Uses: the corrector's phonetic matching, the automatic glossary ("does this name occur in the
transcripts?") and English queries against native-script names (the roman_key column).
Stdlib only.
"""
from __future__ import annotations

import re
import unicodedata

BLOCKS = {0x0900: "Devanagari", 0x0980: "Bengali", 0x0A00: "Gurmukhi", 0x0A80: "Gujarati",
          0x0B00: "Oriya", 0x0B80: "Tamil", 0x0C00: "Telugu", 0x0C80: "Kannada", 0x0D00: "Malayalam"}

# Offsets inside a block. Capitals mark retroflex (T D N S L R) for the readable form.
_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "rii", 0x61: "lii"}
_SIGNS = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "rii",
          0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
          0x4C: "au", 0x62: "li", 0x63: "lii", 0x56: "i", 0x57: "u"}
_CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "c", 0x1B: "ch",
               0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "T", 0x20: "Th", 0x21: "D", 0x22: "Dh",
               0x23: "N", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
               0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
               0x31: "R", 0x32: "l", 0x33: "L", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "S",
               0x38: "s", 0x39: "h", 0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "D",
               0x5D: "Dh", 0x5E: "f", 0x5F: "y"}
_NASAL = {0x01: "n", 0x02: "n", 0x70: "n"}       # candrabindu, anusvara, Gurmukhi tippi
_LABIALS = {0x2A, 0x2B, 0x2C, 0x2D, 0x2E}         # p ph b bh m: anusvara before them is m
_VISARGA = {0x03: "h"}
_VIRAMA = 0x4D
_SILENT = {0x3C, 0x3D, 0x4E, 0x55, 0x71, 0x51, 0x52, 0x53, 0x54}  # nukta, avagraha, marks
# Malayalam chillu letters: a consonant with no vowel.
_CHILLU = {0x0D7A: "N", 0x0D7B: "n", 0x0D7C: "R", 0x0D7D: "l", 0x0D7E: "L", 0x0D7F: "k", 0x0D54: "m"}


def _block(cp: int) -> int | None:
    base = cp & ~0x7F
    return base if base in BLOCKS else None


def romanize(text: str) -> str:
    """Readable romanization. Latin text comes back with diacritics removed, case kept."""
    text = unicodedata.normalize("NFC", text or "")
    out: list[str] = []
    pending_a = False          # the last consonant still owns its inherent 'a'
    for i, ch in enumerate(text):
        cp = ord(ch)
        base = _block(cp)
        if cp in _CHILLU:
            if pending_a:
                out.append("a")
            out.append(_CHILLU[cp])
            pending_a = False
            continue
        if base is None:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(ch)
            continue
        off = cp - base
        if off in _SILENT:
            continue                          # nukta etc.: the consonant keeps its 'a'
        if off == _VIRAMA:
            pending_a = False
            continue
        if off in _SIGNS:
            out.append(_SIGNS[off])           # replaces the inherent 'a'
            pending_a = False
            continue
        if pending_a:
            out.append("a")
            pending_a = False
        if off in _CONSONANTS:
            out.append(_CONSONANTS[off])
            pending_a = True
        elif off in _VOWELS:
            out.append(_VOWELS[off])
        elif off in _NASAL:
            # Anusvara is m at the end of a word (Telugu -ం) and before p/b/m, n elsewhere.
            nxt = text[i + 1] if i + 1 < len(text) else ""
            nb = _block(ord(nxt)) if nxt else None
            if off == 0x02 and (nb is None or ord(nxt) - nb in _LABIALS):
                out.append("m")
            else:
                out.append(_NASAL[off])
        elif off in _VISARGA:
            out.append(_VISARGA[off])
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))       # native digits
        # anything else in the block (rare signs) is dropped
    if pending_a:
        out.append("a")
    s = "".join(out)
    # Latin diacritics: ā -> a, ṣ -> s, ṛ -> r
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s


# Applied in order on lowercase text. Aspiration, vowel length, sibilants and spelling variants
# collapse, so "Bhīṣma", "Bheeshma" and భీష్మ meet. Voicing (k/g, t/d) is kept: it carries meaning.
_KEY_RULES = [
    (re.compile(r"ee"), "i"), (re.compile(r"oo"), "u"),
    (re.compile(r"([aiu])\1+"), r"\1"),     # vowel length
    (re.compile(r"ri(?=[^aeiou])"), "r"),  # vocalic r before a consonant: krishna/kṛṣṇa -> krshna
    (re.compile(r"([kgcjtdpb])h"), r"\1"), # aspirates; also ch -> c
    (re.compile(r"sh|zh"), lambda m: "s" if m.group(0) == "sh" else "l"),
    (re.compile(r"w"), "v"), (re.compile(r"f"), "p"), (re.compile(r"z"), "j"),
    (re.compile(r"q"), "k"), (re.compile(r"x"), "ks"), (re.compile(r"y(?=[^aeiou]|$)"), "i"),
    (re.compile(r"([a-z])\1+"), r"\1"),   # long vowels and doubled consonants
]
_NON_WORD = re.compile(r"[^a-z0-9 ]+")


def _word_key(w: str) -> str:
    for pat, rep in _KEY_RULES:
        w = pat.sub(rep, w)
    if len(w) > 2 and w.endswith("a"):
        w = w[:-1]                            # word-final schwa: arjuna/arjun, karna/karn
    return w


def phonetic_key(text: str) -> str:
    """Coarse, script-independent key. Words are kept apart by single spaces."""
    s = romanize(text).lower()
    s = _NON_WORD.sub(" ", s)
    return " ".join(k for k in (_word_key(w) for w in s.split()) if k)


def roman_key(text: str) -> str:
    """Lighter key for the alias table and English queries: no diacritics, lowercase, single spaces."""
    s = romanize(text).lower()
    return " ".join(_NON_WORD.sub(" ", s).split())


def script_of(text: str) -> str | None:
    """Name of the first Brahmic script found in text, or None."""
    for ch in text or "":
        b = _block(ord(ch))
        if b is not None:
            return BLOCKS[b]
    return None
