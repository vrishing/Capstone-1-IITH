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
    text, changes, review = fix(gl, f"{YUDHISHTHIRA} {word}")
    assert text == f"{YUDHISHTHIRA} {word}" and changes == []
    assert review[0]["reason"] == "mid-confidence" and 86 <= review[0]["score"] < 92
    assert review[0]["term"] == YUDHISHTHIRA


def test_lowering_the_auto_score_applies_it(gl):
    word = YUDHISHTHIRA.replace("ధ", "వ", 1)
    assert fix(gl, word, auto=90.0)[0] == YUDHISHTHIRA


def test_review_needs_the_name_spelled_correctly_somewhere_in_the_video(gl):
    word = YUDHISHTHIRA.replace("ధ", "వ", 1)
    assert fix(gl, word)[2] == []                                      # name never appears: dropped
    assert len(fix(gl, f"{word} యుధిష్ఠిరుడికి")[2]) == 1               # an inflected correct form counts


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


def test_short_keys_are_dropped_unless_one_character_apart(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row("శ్రీరామ", "Q10", "Rama")])
    g = load_glossary(str(p))
    assert c.phonetic_key("శ్రమ") == c.phonetic_key("శ్రీరామ") == "sram"      # the real-world collision
    text, changes, review = fix(g, "శ్రీరామ కష్టపడి శ్రమ చేశాడు")
    assert text == "శ్రీరామ కష్టపడి శ్రమ చేశాడు" and changes == [] and review == []   # not even reviewed
    assert fix(g, "శ్రీరమ")[0] == "శ్రీరామ"                                      # one character off: fine


def test_ordinary_words_that_start_with_another_sound_are_never_proposed(tmp_path):
    # pairs from the Mahabharata review list: (heard word, glossary name)
    pairs = [("అనుమతి", "భానుమతి"), ("రక్షకుడు", "తక్షకుడు"), ("తండ్రులు", "ఆంధ్రులు"),
             ("నాస్తికుడు", "ఆస్తీకుడు"), ("శాశ్వతం", "అశ్వత్థామ")]
    p = tmp_path / "a.csv"
    write_aliases(p, [row(name, f"Q{i}") for i, (_, name) in enumerate(pairs)])
    g = load_glossary(str(p))
    for heard, name in pairs:
        text, changes, review = fix(g, f"{name} {heard}", auto=50.0, review=50.0)   # even with loose thresholds
        assert changes == [] and review == [], heard


def test_a_phrase_must_match_word_by_word_onsets(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row("శ్రీకృష్ణ పరమాత్మ", "Q42", typ="person")])
    g = load_glossary(str(p))
    text, changes, review = fix(g, "శ్రీకృష్ణ పరమాత్మ కృష్ణ పరమాత్మ", review=50.0)
    assert changes == [] and review == []                            # dropping శ్రీ is not a misspelling


def test_real_misspellings_keep_working_under_the_guards(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row("విదురుడు", "Q2002511"), row("సంజయుడు", "Q8"), row("జరాసంధుడు", "Q2723062")])
    g = load_glossary(str(p))
    for heard in ("విధరుడు", "సంజీయుడు", "జలసంధుడు"):
        assert len(fix(g, f"విదురుడు సంజయుడు జరాసంధుడు {heard}")[2]) == 1, heard


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


# ---------- LLM review (one verdict per distinct pair, batched, cached) ----------

REVIEW_WORD = YUDHISHTHIRA.replace("ధ", "వ", 1)                    # mid-confidence (90.9)
IN_VIDEO = f"{YUDHISHTHIRA} {REVIEW_WORD}"                          # the name also appears correctly


class FakeLLM:
    """Answers each numbered item by looking the heard word up in `verdicts`; records prompts."""
    def __init__(self, verdicts, raw_reply=None):
        self.verdicts, self.raw_reply, self.prompts = verdicts, raw_reply, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.raw_reply is not None:
            return self.raw_reply
        items = []
        for line in prompt.splitlines():
            if ". heard: " in line and line.split(".")[0].strip().isdigit():
                i = int(line.split(".")[0])
                heard = line.split("heard: ")[1].split(" |")[0]
                if heard in self.verdicts:
                    items.append({"id": i, "verdict": self.verdicts[heard]})
        return json.dumps({"items": items}, ensure_ascii=False)


def run_llm(gl, texts, llm, cache=None, **cfg):
    segs = [{"text": t, "start": float(i), "duration": 1.0} for i, t in enumerate(texts)]
    cor = Corrector(gl, {"use_llm": True, "language": "te", **cfg})
    return correct_segments(segs, cor, llm, cache if cache is not None else {})


def test_llm_misheard_applies_a_mid_confidence_pair(gl):
    llm = FakeLLM({REVIEW_WORD: "misheard"})
    out, ch, rv = run_llm(gl, [IN_VIDEO], llm)
    assert out[0]["text"] == f"{YUDHISHTHIRA} {YUDHISHTHIRA}" and rv == []
    assert ch[0]["rule"] == "llm_review" and ch[0]["llm"] == "misheard"


def test_llm_keep_stops_even_an_auto_band_fix(gl):
    llm = FakeLLM({"బీష్ముడు": "keep"})                                  # e.g. a real word that scores 100
    out, ch, rv = run_llm(gl, ["ఇదిగో బీష్ముడు"], llm)
    assert out[0]["text"] == "ఇదిగో బీష్ముడు" and ch == []
    assert rv[0]["llm"] == "keep" and rv[0]["reason"].startswith("llm")
    out, ch, _ = run_llm(gl, ["ఇదిగో బీష్ముడు"], FakeLLM({}), llm_audit_auto=False, cache={})
    assert out[0]["text"] == f"ఇదిగో {BHISHMA}"                             # audit off: auto band trusted


def test_one_question_per_distinct_pair_in_batches(gl):
    texts = [IN_VIDEO] * 30 + ["బీష్ముడు"] * 30 + ["అర్జునూడు"] * 30
    llm = FakeLLM({REVIEW_WORD: "misheard", "బీష్ముడు": "misheard", "అర్జునూడు": "misheard"})
    out, ch, rv = run_llm(gl, texts, llm, llm_batch=2)
    assert len(ch) == 90 and rv == []
    assert len(llm.prompts) == 2                                         # 3 distinct pairs, batches of 2
    assert sum(p.count(". heard: ") for p in llm.prompts) == 3
    assert "example: " in llm.prompts[0]                                 # context sentences are sent


def test_verdicts_are_cached_across_videos(gl):
    cache, llm = {}, FakeLLM({REVIEW_WORD: "misheard"})
    run_llm(gl, [IN_VIDEO], llm, cache)
    run_llm(gl, [IN_VIDEO, IN_VIDEO], llm, cache)
    assert len(llm.prompts) == 1 and cache == {f"te\t{REVIEW_WORD}\t{YUDHISHTHIRA}": True}


def test_parse_errors_and_missing_answers_reject_and_are_asked_again(gl):
    for reply in ("not json at all", '{"items": []}', '{"items": [{"id": 1, "verdict": "yes"}]}'):
        cache = {}
        out, ch, rv = run_llm(gl, [IN_VIDEO], FakeLLM({}, raw_reply=reply), cache)
        assert out[0]["text"] == IN_VIDEO and ch == [] and len(rv) == 1
        assert cache == {}                                              # not cached: retried next run


def test_llm_api_error_propagates_so_the_stage_retries(gl):
    def boom(p):
        raise RuntimeError("api down")
    with pytest.raises(RuntimeError):
        run_llm(gl, [IN_VIDEO], boom)


def test_variants_and_ambiguous_pairs_are_never_sent(tmp_path):
    p = tmp_path / "a.csv"
    write_aliases(p, [row(KARNA, "Q3", variants="కర్నుడూ"), row("కర్నుడు", "Q9")])
    g = load_glossary(str(p))
    llm = FakeLLM({})
    out, ch, rv = run_llm(g, ["కర్నుడూ కర్ణూడు"], llm)
    assert ch[0]["rule"] == "variant" and rv[0]["reason"].startswith("ambiguous")
    assert llm.prompts == []


def test_llm_is_off_by_default(gl):
    def gen(p):
        raise AssertionError("must not be called")
    seg = [{"text": IN_VIDEO + " బీష్ముడు", "start": 0.0, "duration": 5.0}]
    out, ch, rv = correct_segments(seg, Corrector(gl), gen, {})
    assert len(rv) == 1 and len(ch) == 1


def test_correct_file_reads_language_and_saves_the_cache(tmp_path, gl):
    asr, out, cache_path = tmp_path / "VID_asr.json", tmp_path / "VID_corrected.json", tmp_path / "c" / "llm.json"
    asr_file(asr, [IN_VIDEO], language="te")
    llm = FakeLLM({REVIEW_WORD: "misheard"})
    summ = correct_file(str(asr), str(out), gl, {"use_llm": True}, llm, str(cache_path))
    assert summ["llm_review"] and summ["n_changes"] == 1
    assert json.load(open(cache_path, encoding="utf-8")) == {f"te\t{REVIEW_WORD}\t{YUDHISHTHIRA}": True}
    correct_file(str(asr), str(out), gl, {"use_llm": True}, llm, str(cache_path))
    assert len(llm.prompts) == 1                                         # second run: cache only


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
    assert f(g1, use_llm=True) != f(g1, use_llm=True, llm_audit_auto=False)
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


# ---------- review decisions -> variants / protected words ----------

def _corrected_with_review(path, reviews, texts=("x",)):
    atomic_write_json(str(path), {"video_id": os.path.basename(str(path)).rsplit("_", 1)[0],
                                  "segments": [{"text": t} for t in texts], "review": reviews})


def _rv(raw, term, eid, rule="whole", new=None):
    return {"segment": 0, "raw": raw, "term": term, "new": new or term, "entity_id": eid,
            "score": 90.0, "rule": rule, "reason": "mid-confidence", "runner_up": 0.0}


def test_review_export_aggregates_prefills_and_keeps_decisions(tmp_path):
    a, b, out = tmp_path / "A_corrected.json", tmp_path / "B_corrected.json", tmp_path / "dec.csv"
    _corrected_with_review(a, [_rv("విధరుడు", "విదురుడు", "Q1")] * 2 + [_rv("నాగరాజు", "నటరాజు", "Q2")]
                           + [_rv("కృష్ణ పరమాత్మ", "శ్రీకృష్ణ పరమాత్మ", "Q3", rule="phrase")])
    _corrected_with_review(b, [_rv("విధరుడు", "విదురుడు", "Q1")] + [_rv("నాగరాజు", "నటరాజు", "Q2")])
    res = c.export_review([str(a), str(b)], str(out), min_count=3)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert res["pairs"] == 2                                            # the phrase is not exported
    assert [(r["raw"], r["count"], r["videos"], r["decision"]) for r in rows] == \
        [("విధరుడు", "3", "2", "accept"), ("నాగరాజు", "2", "2", "")]
    rows[0]["decision"], rows[1]["decision"] = "", "reject"             # the user's edits survive a re-export
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=c.DECISION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    c.export_review([str(a), str(b)], str(out), min_count=3)
    assert [r["decision"] for r in csv.DictReader(open(out, encoding="utf-8"))] == ["", "reject"]


def test_review_apply_adds_variants_and_protects_rejects_and_the_corrector_uses_them(tmp_path):
    al, prot, dec = tmp_path / "aliases.csv", tmp_path / "protected_words.txt", tmp_path / "dec.csv"
    write_aliases(al, [row("విదురుడు", "Q1", variants="పాతది"), row("నటరాజు", "Q2"), row(KARNA, "Q3")])
    prot.write_text("ఇప్పటికే", encoding="utf-8")                       # no trailing newline
    with open(dec, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=c.DECISION_COLUMNS)
        w.writeheader()
        for d, raw, term, eid in [("accept", "విధరుడు", "విదురుడు", "Q1"), ("reject", "నాగరాజు", "నటరాజు", "Q2"),
                                  ("", "విద్రుడు", "విదురుడు", "Q1"), ("accept", KARNA, "విదురుడు", "Q1"),
                                  ("accept", "ఏదో", "లేనిపేరు", "Q9")]:
            w.writerow({"decision": d, "raw": raw, "term": term, "entity_id": eid, "count": 1})
    res = c.apply_decisions(str(dec), str(al), str(prot))
    assert (res["accepted"], res["rejected"], res["skipped_blank"]) == (1, 1, 1)
    assert res["skipped_conflict"] == [KARNA] and res["term_not_found"] == ["ఏదో -> లేనిపేరు"]
    assert os.path.exists(str(al) + ".bak")
    vrow = next(r for r in csv.DictReader(open(al, encoding="utf-8")) if r["term"] == "విదురుడు")
    assert vrow["variants"] == "పాతది|విధరుడు"
    assert prot.read_text(encoding="utf-8").split() == ["ఇప్పటికే", "నాగరాజు"]
    again = c.apply_decisions(str(dec), str(al), str(prot))             # running twice changes nothing
    assert (again["accepted"], again["already_variant"], again["already_protected"]) == (0, 1, 1)

    g = load_glossary(str(al), str(prot))
    text, changes, review = fix(g, "విధరుడు నాగరాజు నాగరాజుకి")
    assert text == "విదురుడు నాగరాజు నాగరాజుకి" and changes[0]["rule"] == "variant" and review == []


def test_llm_review_command_fills_the_cache_and_prints_verdicts(cfg, tmp_path, monkeypatch, capsys):
    import run_pipeline as rp
    write_aliases(cfg.ALIAS_TABLE_PATH, [row(BHISHMA, "Q2"), row(YUDHISHTHIRA, "Q4")])
    os.makedirs(cfg.ASR_OUTPUT_DIR)
    asr_file(rp.asr_path("VID"), ["ఇదిగో బీష్ముడు", IN_VIDEO, "బీష్ముడు"])
    llm = FakeLLM({"బీష్ముడు": "misheard", REVIEW_WORD: "keep"})
    monkeypatch.setattr(rp, "correction_llm", lambda: llm)
    c._CACHE.clear()
    assert c._main(["llm-review"]) == 0
    out = capsys.readouterr().out
    assert "2 distinct pairs, 2 not cached" in out and "1 LLM call(s), 2 new verdict(s)" in out
    assert "KEEP (1 pairs)" in out and "MISHEARD (1 pairs)" in out
    cache = json.load(open(rp.llm_cache_path(), encoding="utf-8"))
    assert cache == {f"te\tబీష్ముడు\t{BHISHMA}": True, f"te\t{REVIEW_WORD}\t{YUDHISHTHIRA}": False}
    assert json.load(open(rp.asr_path("VID"), encoding="utf-8"))["segments"][0]["text"] == "ఇదిగో బీష్ముడు"


def test_prefill_asks_once_for_the_whole_playlist_and_the_stage_then_makes_no_calls(tmp_path, gl):
    files = []
    for i in range(6):                                                   # six videos, same two pairs
        f = tmp_path / f"V{i}_asr.json"
        asr_file(f, [IN_VIDEO, "ఇదిగో బీష్ముడు"])
        files.append(str(f))
    cache_path = str(tmp_path / "corr" / "llm.json")
    llm = FakeLLM({REVIEW_WORD: "misheard", "బీష్ముడు": "keep"})
    res = c.prefill_llm_cache(files, gl, {"use_llm": True}, llm, cache_path, log=lambda *_: None)
    assert len(llm.prompts) == 1 and res["calls"] == 1 and res["new_verdicts"] == 2
    assert next(iter(res["items"].values()))["n"] >= 6                    # occurrences are counted
    for i, f in enumerate(files):                                        # per-video stage: cache only
        c.correct_file(f, str(tmp_path / f"o{i}.json"), gl, {"use_llm": True}, llm, cache_path)
    assert len(llm.prompts) == 1
    d = json.load(open(tmp_path / "o0.json", encoding="utf-8"))
    assert d["segments"][0]["text"] == f"{YUDHISHTHIRA} {YUDHISHTHIRA}"
    assert d["segments"][1]["text"] == "ఇదిగో బీష్ముడు"                    # LLM said keep
    c.prefill_llm_cache(files, gl, {"use_llm": True}, llm, cache_path, log=lambda *_: None)
    assert len(llm.prompts) == 1                                         # rerun: nothing new to ask


def test_a_quota_error_keeps_the_verdicts_already_answered(tmp_path, gl):
    f = tmp_path / "V_asr.json"
    asr_file(f, [IN_VIDEO, "బీష్ముడు"])
    cache_path = str(tmp_path / "llm.json")
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("429 quota exceeded")
        return FakeLLM({REVIEW_WORD: "misheard", "బీష్ముడు": "misheard"})(prompt)
    with pytest.raises(RuntimeError):
        c.prefill_llm_cache([str(f)], gl, {"use_llm": True, "llm_batch": 1}, flaky, cache_path, log=lambda *_: None)
    saved = json.load(open(cache_path, encoding="utf-8"))
    assert len(saved) == 1                                               # first batch was saved before the error
    llm = FakeLLM({REVIEW_WORD: "misheard", "బీష్ముడు": "misheard"})
    c.prefill_llm_cache([str(f)], gl, {"use_llm": True, "llm_batch": 1}, llm, cache_path, log=lambda *_: None)
    assert len(llm.prompts) == 1                                         # only the missing pair is asked
    assert len(json.load(open(cache_path, encoding="utf-8"))) == 2


def test_pipeline_prefills_once_then_stages_use_the_cache(cfg, tmp_path, monkeypatch):
    import run_pipeline as rp
    write_aliases(cfg.ALIAS_TABLE_PATH, [row(BHISHMA, "Q2"), row(YUDHISHTHIRA, "Q4")])
    cfg.CORRECTION_USE_LLM = True
    os.makedirs(cfg.ASR_OUTPUT_DIR)
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(f"VID{i:08d}" for i in range(4)) + "\n")
    for i in range(4):
        asr_file(rp.asr_path(f"VID{i:08d}"), [IN_VIDEO, "ఇదిగో బీష్ముడు"])
    llm = FakeLLM({REVIEW_WORD: "misheard", "బీష్ముడు": "misheard"})
    monkeypatch.setattr(rp, "correction_llm", lambda: llm)

    def upstream(name):
        return Stage(name, lambda v: [rp.asr_path(v)], lambda v: None, lambda: {"s": 1})
    real = rp.build_stages()
    stages = {"downloaded": upstream("downloaded"), "asr_done": upstream("asr_done"),
              "corrected": real["corrected"], "chunked": real["chunked"]}
    c._CACHE.clear()
    assert rp.main(["--ids", str(ids), "--through", "chunked"], stages=stages) == 0
    assert len(llm.prompts) == 1                                         # 4 videos, 2 pairs: one call in total
    d = json.load(open(rp.corrected_path("VID00000003"), encoding="utf-8"))
    assert d["correction"]["llm_review"] and d["segments"][1]["text"] == f"ఇదిగో {BHISHMA}"


def test_pipeline_stops_cleanly_when_the_llm_is_out_of_quota(cfg, tmp_path, monkeypatch, capsys):
    import run_pipeline as rp
    write_aliases(cfg.ALIAS_TABLE_PATH, [row(YUDHISHTHIRA, "Q4")])
    cfg.CORRECTION_USE_LLM = True
    os.makedirs(cfg.ASR_OUTPUT_DIR)
    ids = tmp_path / "ids.txt"
    ids.write_text("VID00000000\n")
    asr_file(rp.asr_path("VID00000000"), [IN_VIDEO])

    def quota(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(rp, "correction_llm", lambda: quota)

    def upstream(name):
        return Stage(name, lambda v: [rp.asr_path(v)], lambda v: None, lambda: {"s": 1})
    real = rp.build_stages()
    stages = {"downloaded": upstream("downloaded"), "asr_done": upstream("asr_done"),
              "corrected": real["corrected"], "chunked": real["chunked"]}
    c._CACHE.clear()
    assert rp.main(["--ids", str(ids), "--through", "chunked"], stages=stages) == 2
    assert "LLM review failed" in capsys.readouterr().err
    assert not os.path.exists(rp.corrected_path("VID00000000"))          # no transcript was half-corrected


# ---------- the LLM is told what the playlist is about ----------

def test_prompt_carries_the_topic_and_the_glossary_description():
    items = [{"raw": "వాసుదేవుడు", "new": "వసుదేవుడు", "term": "వసుదేవుడు", "roman": "Vasudeva",
              "desc": "father of Krishna", "examples": ["ఆ వాసుదేవుడు అన్నాడు"]}]
    p = c._llm_batch_prompt(items, "te", "Mahabharata pravachanam")
    assert "playlist about: Mahabharata pravachanam" in p
    assert "glossary term: వసుదేవుడు (Vasudeva), father of Krishna" in p
    assert "Mahabharata" not in c._llm_batch_prompt(items, "te", "")          # no topic: old wording
    assert "playlist about" not in c._llm_batch_prompt(items, "te", "")


def test_glossary_notes_become_the_term_description(tmp_path):
    p = tmp_path / "a.csv"
    r = row(ARJUNA, "Q1", "Arjuna")
    r["notes"] = "Pandava prince | multiword | also: wikipedia_title_te"
    write_aliases(p, [r])
    g = load_glossary(str(p))
    assert g.single[0].desc == "Pandava prince"


def test_verdicts_are_cached_per_topic_so_nothing_carries_over(gl):
    assert c.llm_key("te", "a", "b") == "te\ta\tb"                              # unchanged without a topic
    assert c.llm_key("te", "a", "b", "Mahabharata") != c.llm_key("te", "a", "b", "Machine learning")
    cache, llm = {}, FakeLLM({REVIEW_WORD: "misheard"})
    run_llm(gl, [IN_VIDEO], llm, cache, topic="Mahabharata")
    assert len(llm.prompts) == 1 and "playlist about: Mahabharata" in llm.prompts[0]
    run_llm(gl, [IN_VIDEO], llm, cache, topic="Mahabharata")
    assert len(llm.prompts) == 1                                                # same topic: cached
    run_llm(gl, [IN_VIDEO], llm, cache, topic="Something else")
    assert len(llm.prompts) == 2                                                # new topic: asked again
    assert run_llm(gl, [IN_VIDEO], FakeLLM({}), {})[1] == []                    # no topic, empty cache: no verdict


def test_topic_and_description_reach_the_prompt_through_prefill(tmp_path):
    p = tmp_path / "a.csv"
    r = row(YUDHISHTHIRA, "Q4", "Yudhishthira")
    r["notes"] = "eldest Pandava"
    write_aliases(p, [r])
    g = load_glossary(str(p))
    f = tmp_path / "V_asr.json"
    asr_file(f, [IN_VIDEO])
    llm = FakeLLM({REVIEW_WORD: "misheard"})
    c.prefill_llm_cache([str(f)], g, {"use_llm": True, "topic": "Mahabharata"}, llm, str(tmp_path / "l.json"),
                        log=lambda *_: None)
    assert "playlist about: Mahabharata" in llm.prompts[0] and "eldest Pandava" in llm.prompts[0]
    assert list(json.load(open(tmp_path / "l.json", encoding="utf-8")))[0].startswith("te~")


def test_topic_is_part_of_the_stage_fingerprint(tmp_path):
    from manifest import fingerprint
    p = tmp_path / "a.csv"
    write_aliases(p, [row(ARJUNA, "Q1")])
    g = load_glossary(str(p))
    f = lambda **k: fingerprint(c.stage_settings(g, k, "te"))
    assert f(use_llm=True, topic="A") != f(use_llm=True, topic="B")
    assert f(topic="A") == f(topic="B")                                         # LLM off: topic is irrelevant


def test_playlist_topic_combines_the_name_and_the_pack_topic(cfg, tmp_path):
    import run_pipeline as rp
    cfg.PLAYLIST_NAME = "Telugu pravachanam"
    assert rp.playlist_topic() == "Telugu pravachanam"
    pack = tmp_path / "domain_pack.json"
    pack.write_text(json.dumps({"topic": "Mahabharata"}), encoding="utf-8")
    cfg.DOMAIN_PACK_PATH = str(pack)
    assert rp.playlist_topic() == "Telugu pravachanam (Mahabharata)"
    cfg.PLAYLIST_NAME = "Mahabharata in Telugu"
    assert rp.playlist_topic() == "Mahabharata in Telugu"                       # not repeated
    assert rp.correction_cfg()["topic"] == "Mahabharata in Telugu"