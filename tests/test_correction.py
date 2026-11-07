import csv
import json
import os
import sys
import types

import pytest

import correction as c
from correction import Corrector, correct_file, correct_segments, drop_leading_noise, load_glossary
from manifest import Manifest, Stage, atomic_write_json

COLS = ["term", "entity_id", "roman", "type", "variants", "needs_context", "source", "notes"]

ARJUNA = "అర్జునుడు"        # key arjunudu
BHISHMA = "భీష్ముడు"        # key bismudu
KARNA = "కర్ణుడు"          # key karnudu
YUDHISHTHIRA = "యుధిష్ఠిరుడు"  # key yudistirudu (11 letters: one edit scores 90.9, in the review band)


def row(term, eid, roman="", typ="person", variants="", needs_context="0"):
    return {"term": term, "entity_id": eid, "roman": roman, "type": typ, "variants": variants,
            "needs_context": needs_context, "source": "", "notes": ""}


def write_aliases(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def gl(tmp_path):
    p = tmp_path / "aliases.csv"
    write_aliases(p, [row(ARJUNA, "Q1", "Arjuna"), row(BHISHMA, "Q2", "Bhishma"), row(KARNA, "Q3", "Karna"),
                      row(YUDHISHTHIRA, "Q4", "Yudhishthira")])
    return load_glossary(str(p))


def fix(gl, text, **cfg):
    seg = [{"text": text, "start": 0.0, "duration": 5.0}]
    out, ch, rv = correct_segments(seg, Corrector(gl, cfg))
    return out[0]["text"], ch, rv


# ---------- whole words, already-correct forms, inflection ----------

def test_correct_and_inflected_forms_are_untouched(gl):
    for text in (ARJUNA, "అర్జునుడికి", "అర్జునుడితో వెళ్ళాడు", "అర్జునుని"):
        assert fix(gl, text)[0] == text


def test_misheard_name_is_fixed_when_the_phonetic_key_matches(gl):
    assert fix(gl, "అర్జునూడు వచ్చాడు")[0] == "అర్జునుడు వచ్చాడు"           # long u in the ending
    assert fix(gl, "బీష్ముడు")[0] == BHISHMA          # aspirate lost
    assert fix(gl, "కర్నుడు")[0] == KARNA              # dental n for retroflex N


def test_inflection_is_preserved_when_the_stem_is_fixed(gl):
    # the transcript shows the ending డికి on a correct form, so the same ending on a mis-heard stem is fixed
    text, changes, _ = fix(gl, "అర్జునుడికి చెప్పాడు అర్జునూడికి")
    assert text == "అర్జునుడికి చెప్పాడు అర్జునుడికి"
    assert changes[0]["rule"] == "inflected" and changes[0]["raw"] == "అర్జునూడికి"


def test_inflected_fixes_need_an_ending_the_transcript_shows_on_a_correct_form(gl):
    assert fix(gl, "అర్జునూడికి")[0] == "అర్జునూడికి"            # no correct example: left alone
    assert fix(gl, "అర్జునుడికి అర్జునూకఖగ")[0] == "అర్జునుడికి అర్జునూకఖగ"   # arbitrary suffix: left alone


def test_short_stems_never_use_the_inflected_path(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row("భీముడు", "Q7", "Bhima")])                 # stem key "bimu" is only 4 letters
    g = load_glossary(str(p))
    assert fix(g, "భీమునికి")[0] == "భీమునికి"
    assert fix(g, "భీమునికి భిమూనికి")[0] == "భీమునికి భిమూనికి"
    assert fix(g, "భిమూడు")[0] == "భీముడు"                          # the whole-word path still works


def test_punctuation_around_a_word_is_kept(gl):
    assert fix(gl, "అర్జునూడు, బీష్ముడు.")[0] == f"{ARJUNA}, {BHISHMA}."


def test_only_whole_words_are_changed(gl):
    text = "xxఅర్జునుడుyy ఏదో"
    assert fix(gl, text)[0] == text
    assert fix(gl, "రాజు మనిషి చెప్పాడు")[0] == "రాజు మనిషి చెప్పాడు"


def test_every_change_is_logged_with_its_raw_text(gl):
    text, changes, review = fix(gl, "ఇదిగో బీష్ముడు")
    ch = changes[0]
    assert (ch["raw"], ch["new"], ch["term"], ch["entity_id"], ch["segment"], ch["word"]) == \
        ("బీష్ముడు", BHISHMA, BHISHMA, "Q2", 0, 1)
    assert ch["score"] == 100.0 and review == []


# ---------- thresholds, margin, review ----------

def test_mid_confidence_goes_to_review_and_text_is_not_changed(gl):
    word = YUDHISHTHIRA.replace("ధ", "వ", 1)               # one edit in an 11-letter key: 90.9
    text, changes, review = fix(gl, word)
    assert text == word and changes == []
    assert review[0]["reason"] == "mid-confidence" and 86 <= review[0]["score"] < 92
    assert review[0]["term"] == YUDHISHTHIRA


def test_lowering_the_auto_score_applies_it(gl):
    word = YUDHISHTHIRA.replace("ధ", "వ", 1)
    assert fix(gl, word, auto=90.0)[0] == YUDHISHTHIRA


def test_two_entities_within_the_margin_are_ambiguous_and_not_applied(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(KARNA, "Q3", "Karna"), row("కర్నుడు", "Q9", "Other")])
    g = load_glossary(str(p))
    word = "కర్ణూడు"                                          # key karnudu: 100 for both entities
    text, changes, review = fix(g, word)
    assert text == word and changes == []
    assert review[0]["reason"].startswith("ambiguous")


def test_other_spellings_of_one_entity_do_not_block_a_fix(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(KARNA, "Q3", "Karna"), row("కర్నుడు", "Q3", "Karna")])
    g = load_glossary(str(p))
    assert fix(g, "కర్ణూడు")[1], "same entity under two spellings is not ambiguous"


def test_a_spelling_of_the_same_entity_is_never_rewritten_into_another(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(ARJUNA, "Q1", "Arjuna"), row("పార్థుడు", "Q1", "Arjuna")])
    g = load_glossary(str(p))
    assert fix(g, "పార్థుడు")[0] == "పార్థుడు"


def test_short_keys_are_reviewed_not_auto_applied_unless_one_character_apart(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row("శ్రీరామ", "Q10", "Rama")])
    g = load_glossary(str(p))
    assert c.phonetic_key("శ్రమ") == c.phonetic_key("శ్రీరామ") == "sram"      # the real-world collision
    text, changes, review = fix(g, "కష్టపడి శ్రమ చేశాడు")
    assert text == "కష్టపడి శ్రమ చేశాడు" and changes == []
    assert review[0]["reason"].startswith("short key")
    assert fix(g, "శ్రీరమ")[0] == "శ్రీరామ"                                      # one character off: fine


# ---------- exact-only rows, variants, protection ----------

def test_needs_context_and_other_types_match_exactly_only(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(ARJUNA, "Q1", needs_context="1"), row(BHISHMA, "Q2", typ="other")])
    g = load_glossary(str(p))
    assert fix(g, "అర్జునూడు బీష్ముడు")[0] == "అర్జునూడు బీష్ముడు"


def test_no_fuzzy_types_from_the_domain_pack_are_exact_only(tmp_path):
    p, pack = tmp_path / "a.csv", tmp_path / "domain_pack.json"
    write_aliases(p, [row(ARJUNA, "Q1", typ="place")])
    pack.write_text(json.dumps({"no_fuzzy_types": ["place"]}), encoding="utf-8")
    assert fix(load_glossary(str(p), pack_path=str(pack)), "అర్జునూడు")[0] == "అర్జునూడు"
    assert fix(load_glossary(str(p)), "అర్జునూడు")[0] == ARJUNA


def test_variants_are_rewritten_exactly(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(ARJUNA, "Q1", variants="అర్జుణుడు|వేరే", needs_context="1")])
    g = load_glossary(str(p))
    assert fix(g, "అర్జుణుడు వేరే")[0] == f"{ARJUNA} {ARJUNA}"


def test_protected_words_are_never_changed(tmp_path):
    p, prot = tmp_path / "a.csv", tmp_path / "protected_words.txt"
    write_aliases(p, [row(ARJUNA, "Q1")])
    assert fix(load_glossary(str(p)), "అర్జునూడు")[0] == ARJUNA
    prot.write_text("# ordinary words\nఅర్జునూడు\n", encoding="utf-8")
    assert fix(load_glossary(str(p), str(prot)), "అర్జునూడు")[0] == "అర్జునూడు"


def test_a_protected_name_is_also_protected_when_inflected(tmp_path):
    p, prot = tmp_path / "a.csv", tmp_path / "protected_words.txt"
    write_aliases(p, [row("వసుదేవుడు", "Q20", "Vasudeva")])
    seen = "వసుదేవుని వసుదేవుడి "                  # correct forms in the same file teach the endings
    words = "వాసుదేవుడు వాసుదేవుని వాసుదేవుడి"
    assert fix(load_glossary(str(p)), seen + words)[0] == seen + "వసుదేవుడు వసుదేవుని వసుదేవుడి"
    prot.write_text("వాసుదేవుడు\n", encoding="utf-8")
    assert fix(load_glossary(str(p), str(prot)), seen + words)[0] == seen + words


def test_multi_word_terms_match_whole_groups_and_protect_their_words(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row("కురుక్షేత్ర యుద్ధము", "Q5", "Kurukshetra War", typ="place")])
    g = load_glossary(str(p))
    assert fix(g, "ఇది కురుక్షేత్ర యుద్దము జరిగింది")[0] == "ఇది కురుక్షేత్ర యుద్ధము జరిగింది"
    assert fix(g, "కురుక్షేత్ర యుద్ధము")[0] == "కురుక్షేత్ర యుద్ధము"
    assert "కురుక్షేత్ర" in g.protected


# ---------- empty glossary and leading noise ----------

def test_missing_or_empty_glossary_passes_text_through(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    hdr = tmp_path / "hdr.csv"
    write_aliases(hdr, [])
    for g in (load_glossary(str(tmp_path / "nope.csv")), load_glossary(str(empty)), load_glossary(str(hdr))):
        assert not g and g.version == "empty"
        assert fix(g, "అర్జునూడు బీష్ముడు")[0] == "అర్జునూడు బీష్ముడు"


def test_leading_noise_segments_are_dropped_but_never_all():
    segs = [{"text": t} for t in ("ह", "हम", "नमस्ते दोस्तों", "हम")]
    n, dropped = drop_leading_noise(segs, 2, 3)
    assert n == 2 and [d["text"] for d in dropped] == ["ह", "हम"]
    assert drop_leading_noise(segs, 0, 3) == (0, [])
    assert drop_leading_noise([{"text": "ह"}], 2, 3)[0] == 0


# ---------- files ----------

def asr_file(path, texts, language="te"):
    segs = [{"text": t, "start": i * 10.0, "duration": 9.0, "speech_start": i * 10.0 + 0.2,
             "speech_end": i * 10.0 + 8.8} for i, t in enumerate(texts)]
    atomic_write_json(str(path), {"video_id": "VID", "language": language, "segments": segs, "stats": {"n": 1}})
    return segs


def test_correct_file_writes_raw_segments_changes_and_a_summary(tmp_path, gl):
    asr, out = tmp_path / "VID_asr.json", tmp_path / "VID_corrected.json"
    asr_file(asr, ["ह", "హ", "ఇదిగో బీష్ముడు వచ్చాడు", "మాట అర్జునూడికి చెప్పాడు", "అర్జునుడికి నమస్కారం"])
    summ = correct_file(str(asr), str(out), gl, {})
    d = json.load(open(out, encoding="utf-8"))
    assert len(d["segments"]) == len(d["raw_segments"]) == 3
    assert d["segments"][0]["text"] == f"ఇదిగో {BHISHMA} వచ్చాడు"
    assert d["raw_segments"][0]["text"] == "ఇదిగో బీష్ముడు వచ్చాడు"          # raw text is never lost
    assert d["segments"][1]["text"] == "మాట అర్జునుడికి చెప్పాడు"
    assert [x["raw"] for x in d["changes"]] == ["బీష్ముడు", "అర్జునూడికి"]
    assert len(d["correction"]["dropped_leading"]) == 2 and summ["n_changes"] == 2
    assert d["correction"]["glossary_version"] == gl.version and d["stats"] == {"n": 1}
    assert json.load(open(asr, encoding="utf-8"))["segments"][0]["text"] == "ह"   # input untouched


def test_empty_glossary_file_keeps_every_word_when_noise_dropping_is_off(tmp_path):
    asr, out = tmp_path / "VID_asr.json", tmp_path / "VID_corrected.json"
    segs = asr_file(asr, ["ह", "అర్జునూడు"])
    correct_file(str(asr), str(out), load_glossary(None), {"drop_leading_chars": 0})
    d = json.load(open(out, encoding="utf-8"))
    assert [s["text"] for s in d["segments"]] == [s["text"] for s in segs] == [s["text"] for s in d["raw_segments"]]


def test_corrected_file_feeds_spans_with_raw_text(tmp_path, gl):
    from spans import spans_from_asr_file
    asr, out = tmp_path / "VID_asr.json", tmp_path / "VID_corrected.json"
    asr_file(asr, ["ఇదిగో బీష్ముడు"] * 12)
    correct_file(str(asr), str(out), gl, {})
    sp = spans_from_asr_file(str(out))
    assert sp and BHISHMA in sp[0]["text"] and "బీష్ముడు" in sp[0]["raw_text"]


# ---------- LLM review ----------

def llm_case(gl, text, generate, cache=None):
    seg = [{"text": text, "start": 0.0, "duration": 5.0}]
    cor = Corrector(gl, {"use_llm": True})
    return correct_segments(seg, cor, generate, cache if cache is not None else {})


REVIEW_WORD = YUDHISHTHIRA.replace("ధ", "వ", 1)


def test_llm_review_accepts_only_what_the_code_proposed(gl):
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return '```json\n{"accept": true}\n```'
    out, ch, rv = llm_case(gl, f"మాట {REVIEW_WORD}", gen)
    assert out[0]["text"] == f"మాట {YUDHISHTHIRA}" and ch[0]["rule"] == "llm_review" and rv == []
    assert len(calls) == 1


def test_llm_reject_parse_error_and_cache(gl):
    for reply in ('{"accept": false}', "not json at all", '{"accept": "yes"}'):
        out, ch, rv = llm_case(gl, REVIEW_WORD, lambda p, r=reply: r)
        assert out[0]["text"] == REVIEW_WORD and ch == [] and len(rv) == 1
    calls, cache = [], {}

    def gen(p):
        calls.append(p)
        return '{"accept": false}'
    llm_case(gl, REVIEW_WORD, gen, cache)
    llm_case(gl, REVIEW_WORD, gen, cache)
    assert len(calls) == 1 and len(cache) == 1


def test_llm_api_error_propagates_so_the_stage_retries(gl):
    def boom(p):
        raise RuntimeError("api down")
    with pytest.raises(RuntimeError):
        llm_case(gl, REVIEW_WORD, boom)


def test_llm_is_off_by_default_and_skips_confident_and_ambiguous_cases(gl):
    def gen(p):
        raise AssertionError("must not be called")
    seg = [{"text": REVIEW_WORD + " బీష్ముడు", "start": 0.0, "duration": 5.0}]
    out, ch, rv = correct_segments(seg, Corrector(gl), gen, {})            # use_llm False
    assert len(rv) == 1 and len(ch) == 1
    out, ch, rv = correct_segments([{"text": "బీష్ముడు", "start": 0.0, "duration": 5.0}],
                                   Corrector(gl, {"use_llm": True}), gen, {})  # confident: no LLM needed
    assert out[0]["text"] == BHISHMA


# ---------- fingerprint and stages ----------

def test_glossary_version_and_thresholds_change_the_stage_fingerprint(tmp_path):
    from manifest import fingerprint
    p = tmp_path / "a.csv"
    write_aliases(p, [row(ARJUNA, "Q1")])
    g1 = load_glossary(str(p))
    write_aliases(p, [row(ARJUNA, "Q1"), row(BHISHMA, "Q2")])
    g2 = load_glossary(str(p))
    f = lambda g, **k: fingerprint(c.stage_settings(g, k, "te"))
    assert f(g1) != f(g2) and f(g1) == f(g1)
    assert f(g1) != f(g1, min_auto_key_len=4) and f(g1) != f(g1, auto=90.0) and f(g1) != f(g1, use_llm=True) and f(g1) != f(g1, drop_leading_chars=0)


def test_get_glossary_reloads_when_the_file_changes(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(ARJUNA, "Q1")])
    v1 = c.get_glossary(str(p)).version
    assert c.get_glossary(str(p)) is c.get_glossary(str(p))
    write_aliases(p, [row(ARJUNA, "Q1"), row(BHISHMA, "Q2")])
    assert c.get_glossary(str(p)).version != v1


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    m = types.ModuleType("config")
    m.ASR_OUTPUT_DIR = str(tmp_path / "asr")
    m.CAPTIONS_DIR = str(tmp_path / "caps")
    m.CORRECTED_DIR = str(tmp_path / "corrected")
    m.SPANS_DIR = str(tmp_path / "spans")
    m.MANIFEST_PATH = str(tmp_path / "manifest.db")
    m.ALIAS_TABLE_PATH = str(tmp_path / "aliases.csv")
    m.PROTECTED_WORDS_PATH = str(tmp_path / "protected_words.txt")
    m.DOMAIN_PACK_PATH = str(tmp_path / "domain_pack.json")
    m.TARGET_LANGUAGE = "te"
    m.TRANSCRIPT_LANGS = ["te"]
    m.HF_TOKEN = None
    m.ASR_DECODER, m.ASR_MAX_CHUNK_S, m.ASR_TOP_DB, m.ASR_PAD_S = "ctc", 8.0, 35.0, 0.2
    m.MAX_ATTEMPTS = 3
    m.SPAN_MIN_S, m.SPAN_MAX_S, m.SPAN_PAUSE_S = 45.0, 90.0, 1.0
    monkeypatch.setitem(sys.modules, "config", m)
    monkeypatch.delitem(sys.modules, "run_pipeline", raising=False)
    return m


def test_corrected_and_chunked_stages_run_resume_and_invalidate(cfg, tmp_path):
    import run_pipeline as rp
    write_aliases(cfg.ALIAS_TABLE_PATH, [row(ARJUNA, "Q1")])
    os.makedirs(cfg.ASR_OUTPUT_DIR)
    asr_file(rp.asr_path("VID"), ["ఇదిగో అర్జునూడు"] * 14)

    def upstream(name):
        return Stage(name, lambda v: [rp.asr_path(v)], lambda v: None, lambda: {"s": 1})
    real = rp.build_stages()
    stages = {"downloaded": upstream("downloaded"), "asr_done": upstream("asr_done"),
              "corrected": real["corrected"], "chunked": real["chunked"]}
    assert rp.main(["--ids", _ids(tmp_path), "--through", "chunked"], stages=stages) == 0
    d = json.load(open(rp.corrected_path("VID"), encoding="utf-8"))
    assert d["segments"][0]["text"] == f"ఇదిగో {ARJUNA}" and d["correction"]["n_changes"] == 14
    spans = json.load(open(rp.spans_path("VID"), encoding="utf-8"))
    assert spans["n_spans"] >= 1 and "అర్జునూడు" in spans["spans"][0]["raw_text"]
    m = Manifest(cfg.MANIFEST_PATH)
    assert m.last_completed("VID") == "chunked"
    t = m.get("VID", "chunked")["updated_at"]

    assert rp.main(["--ids", _ids(tmp_path), "--through", "chunked"], stages=stages) == 0   # nothing reruns
    assert m.get("VID", "chunked")["updated_at"] == t and m.get("VID", "corrected")["status"] == "done"

    write_aliases(cfg.ALIAS_TABLE_PATH, [row(ARJUNA, "Q1"), row(BHISHMA, "Q2")])             # glossary changed
    fp_before = m.get("VID", "corrected")["fingerprint"]
    assert rp.main(["--ids", _ids(tmp_path), "--through", "chunked"], stages=stages) == 0
    assert m.get("VID", "corrected")["fingerprint"] != fp_before
    assert m.last_completed("VID") == "chunked"


def _ids(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("VID\n")
    return str(p)