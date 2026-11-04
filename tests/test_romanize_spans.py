"""Shared romanized key across scripts, and the 45-90 s span builder (offline)."""
import json

import pytest

from romanize import phonetic_key, roman_key, romanize, script_of
from spans import build_spans, point_id, spans_from_asr_file


# ---------- romanization ----------

@pytest.mark.parametrize("forms", [
    ["కుంతి", "कुंती", "Kuntī", "Kunti"],
    ["కృష్ణ", "कृष्ण", "Kṛṣṇa", "Krishna"],
    ["అర్జున", "अर्जुन", "Arjuna"],
    ["భీష్మ", "भीष्म", "Bhīṣma", "Bheeshma"],
    ["ద్రౌపది", "द्रौपदी", "Draupadī"],
    ["యుధిష్ఠిర", "Yudhiṣṭhira"],
    ["ಕರ್ಣ", "কর্ণ", "કર્ણ", "କର୍ଣ୍ଣ", "Karna"],
])
def test_same_name_same_key_across_scripts(forms):
    assert len({phonetic_key(f) for f in forms}) == 1, {f: phonetic_key(f) for f in forms}


def test_romanize_readable_forms():
    assert romanize("కుంతి") == "kunti"
    assert romanize("अर्जुन") == "arjuna"
    assert romanize("భీష్మ") == "bhiiSma"                 # length and retroflex kept
    assert romanize("హస్తినాపురం") == "hastinaapuram"     # final anusvara is m
    assert romanize("गंगा") == "gangaa"                    # anusvara before g is n
    assert romanize("कर्ण १२") == "karNa 12"               # virama, native digits
    assert romanize("കർണ്ണൻ") == "kaRNNan"                 # Malayalam chillu


def test_keys_handle_case_spaces_and_punctuation():
    assert phonetic_key("  Śrī  Kṛṣṇa! ") == phonetic_key("శ్రీ కృష్ణ")
    assert roman_key("Kuntī") == "kunti" and roman_key("Dhṛtarāṣṭra") == "dhrtarastra"
    assert phonetic_key("") == "" and romanize(None) == ""


def test_different_names_stay_apart():
    keys = {phonetic_key(w) for w in ["కర్ణుడు", "కృష్ణుడు", "కుంతి", "భీముడు", "భీష్ముడు", "నకులుడు", "సహదేవుడు"]}
    assert len(keys) == 7


def test_script_of():
    assert script_of("abc కుంతి") == "Telugu"
    assert script_of("कुंती") == "Devanagari"
    assert script_of("Kunti") is None


# ---------- spans ----------

def segs(*spec):
    """spec: (start, duration) pairs; text is the index."""
    return [{"text": f"w{i}", "start": s, "duration": d} for i, (s, d) in enumerate(spec)]


def steady(n, dur=6.0, gap=0.3, long_gaps=()):
    """n back-to-back segments with short gaps, and long pauses after the given indices."""
    out, t = [], 0.0
    for i in range(n):
        out.append({"text": f"w{i}", "start": round(t, 2), "duration": dur})
        t += dur + (2.0 if i in long_gaps else gap)
    return out


def test_spans_cut_at_long_pause_after_min():
    s = steady(30, long_gaps={8})                 # pause after ~57 s
    sp = build_spans("vid", s, "te")
    assert sp[0]["segment_range"] == [0, 8]
    assert 45 <= sp[0]["duration"] <= 90


def test_spans_never_exceed_max_and_never_overlap():
    s = steady(60)                                # no long pauses at all
    sp = build_spans("vid", s, "te")
    assert all(x["duration"] <= 90 for x in sp[:-1])
    for a, b in zip(sp, sp[1:]):
        assert a["segment_range"][1] + 1 == b["segment_range"][0]   # contiguous, no overlap
        assert a["end"] <= b["start"]
    assert sp[0]["segment_range"][0] == 0 and sp[-1]["segment_range"][1] == 59
    assert " ".join(x["text"] for x in sp).split() == [f"w{i}" for i in range(60)]


def test_pause_before_min_does_not_cut():
    s = steady(30, long_gaps={2})                 # pause at ~19 s: too early
    sp = build_spans("vid", s, "te")
    assert sp[0]["segment_range"][0] == 0 and sp[0]["segment_range"][1] > 2


def test_short_tail_is_merged():
    s = steady(16, long_gaps={7})                 # 8 segs ~50 s, then a ~50 s piece
    assert len(build_spans("vid", s)) == 2
    s = steady(10, long_gaps={7})                 # tail of 2 segments (~12 s) folds back in
    sp = build_spans("vid", s)
    assert len(sp) == 1 and sp[0]["segment_range"] == [0, 9]


def test_ids_are_stable_and_point_ids_deterministic():
    s = steady(30, long_gaps={8})
    a, b = build_spans("abcdefghijk", s), build_spans("abcdefghijk", s)
    assert [x["span_id"] for x in a] == [x["span_id"] for x in b]
    assert a[0]["span_id"] == "abcdefghijk_000000000"
    assert a[1]["point_id"] == point_id(a[1]["span_id"]) == b[1]["point_id"]
    assert len({x["point_id"] for x in a}) == len(a)


def test_raw_text_kept_beside_corrected_text():
    s = steady(10)
    raw = [dict(x, text=x["text"] + "_raw") for x in s]
    sp = build_spans("vid", s, raw_segments=raw)
    assert sp[0]["text"].startswith("w0 w1") and sp[0]["raw_text"].startswith("w0_raw w1_raw")
    with pytest.raises(ValueError):
        build_spans("vid", s, raw_segments=raw[:3])


def test_empty_and_unsorted_input():
    assert build_spans("vid", []) == []
    s = steady(12)
    assert build_spans("vid", list(reversed(s))) == build_spans("vid", s)
    with pytest.raises(ValueError):
        build_spans("vid", s, min_s=90, max_s=45)


def test_spans_from_asr_file(tmp_path):
    p = tmp_path / "abcdefghijk_asr.json"
    p.write_text(json.dumps({"video_id": "abcdefghijk", "language": "hi", "segments": steady(20)}),
                 encoding="utf-8")
    sp = spans_from_asr_file(str(p))
    assert sp and sp[0]["language"] == "hi" and sp[0]["video_id"] == "abcdefghijk"


def test_padded_segments_cut_at_largest_gap_without_threshold():
    # Like real IndicConformer output: segments padded into the silence, so every gap is small.
    s, t = [], 0.0
    gaps = [0.05] * 40
    gaps[10] = 0.35                               # the most natural pause, at ~66 s
    gaps[12] = 0.20
    for i in range(40):
        s.append({"text": f"w{i}", "start": round(t, 2), "duration": 6.0})
        t += 6.0 + gaps[i]
    sp = build_spans("vid", s, pause_s=1.0)
    assert sp[0]["segment_range"] == [0, 10]
    assert all(45 <= x["duration"] <= 90 for x in sp[:-1])


@pytest.mark.parametrize("n", range(14, 60))
def test_hard_cap_holds_for_every_length_including_the_last_span(n):
    sp = build_spans("vid", steady(n, dur=6.1, gap=0.1))
    assert all(x["duration"] <= 90 for x in sp), [x["duration"] for x in sp]
    assert sp[-1]["segment_range"][1] == n - 1


def test_avoids_leaving_a_scrap_at_the_end():
    sp = build_spans("vid", steady(16, dur=6.0, gap=0.1))   # ~97 s: 90 + 7 would be the naive cut
    assert len(sp) == 2 and min(x["duration"] for x in sp) >= 22.5


def test_spans_use_speech_bounds_when_present_and_fall_back_when_not():
    # Chunks touch (gap 0) but the speaker paused 1.5 s after segment 9: only speech bounds show it.
    s, t = [], 0.0
    for i in range(30):
        s.append({"text": f"w{i}", "start": round(t, 2), "duration": 6.0,
                  "speech_start": round(t + 0.3, 2), "speech_end": round(t + 6.0 - (1.5 if i == 9 else 0.3), 2)})
        t += 6.0
    assert build_spans("vid", s)[0]["segment_range"] == [0, 9]       # 60 s, cut at the real pause
    plain = [{k: v for k, v in x.items() if not k.startswith("speech_")} for x in s]
    assert build_spans("vid", plain)[0]["segment_range"][1] != 9     # old transcripts: no signal
