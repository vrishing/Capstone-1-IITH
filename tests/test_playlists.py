"""Playlist registry, per-playlist language, sample ordering and the 'unavailable' status (offline)."""
import json
import sys

import pytest

import playlists
from manifest import Manifest, Stage, Unavailable, atomic_write_json, process_video, run_stage

URL = "https://www.youtube.com/playlist?list=PLabcdefghijklmnopqrstuvwxyz012345"
PID = "PLabcdefghijklmnopqrstuvwxyz012345"


@pytest.fixture
def reg(tmp_path, monkeypatch):
    path = tmp_path / "playlists.json"
    path.write_text(json.dumps({"default": "mb", "playlists": {
        "mb": {"id": "PLD0OXwSZSSA8", "language": "te", "name": "Mahabharata"}}}), encoding="utf-8")
    monkeypatch.setattr(playlists, "REGISTRY_PATH", path)
    return path


# ---------- registry ----------

def test_parse_playlist_id_from_urls_and_bare_ids():
    assert playlists.parse_playlist_id(URL) == PID
    assert playlists.parse_playlist_id(f"https://www.youtube.com/watch?v=abcdefghijk&list={PID}&index=3") == PID
    assert playlists.parse_playlist_id(PID) == PID
    with pytest.raises(playlists.RegistryError):
        playlists.parse_playlist_id("not a url")


def test_add_then_resolve_by_key_url_and_id(reg):
    key, entry = playlists.add(URL, "hi", key="nptel")
    assert entry == {"id": PID, "language": "hi", "name": "nptel"}
    assert playlists.resolve("nptel")[1]["language"] == "hi"
    assert playlists.resolve(URL)[0] == "nptel"
    assert playlists.resolve(PID)[0] == "nptel"
    assert playlists.resolve(None)[0] == "mb"          # default unchanged


def test_add_is_idempotent_and_refuses_conflicts(reg):
    playlists.add(URL, "hi", key="nptel")
    assert playlists.add(URL, "hi")[0] == "nptel"      # same playlist again: no new entry
    with pytest.raises(playlists.RegistryError, match="already registered"):
        playlists.add(URL, "ta")
    with pytest.raises(playlists.RegistryError, match="already used"):
        playlists.add("https://www.youtube.com/playlist?list=PLzzzzzzzzzzzzz", "ta", key="nptel")
    assert len(playlists.load()["playlists"]) == 2


def test_make_default(reg):
    playlists.add(URL, "ta", key="tamil", make_default=True)
    assert playlists.resolve(None)[0] == "tamil"


def test_unsupported_language_is_refused(reg):
    with pytest.raises(playlists.RegistryError, match="not supported"):
        playlists.add(URL, "en")


def test_unknown_playlist_message(reg):
    with pytest.raises(playlists.RegistryError, match="not registered"):
        playlists.resolve("https://www.youtube.com/playlist?list=PLnotthereatall123")
    with pytest.raises(playlists.RegistryError, match="Known: mb"):
        playlists.resolve("nope")


def test_guess_language_only_for_unique_scripts():
    assert playlists.guess_language(["మహాభారతం ప్రవచనం - 1"]) == "te"
    assert playlists.guess_language(["தமிழ் பாடம்"]) == "ta"
    assert playlists.guess_language(["डेटा संरचना"]) is None      # Devanagari: hi, mr, ne, sa...
    assert playlists.guess_language(["Lecture 1 - Arrays"]) is None


def test_spread_sample():
    ids = [f"v{i}" for i in range(61)]
    s = playlists.spread_sample(ids, 10)
    assert len(s) == 10 and s[0] == "v0" and s[-1] == "v60" and len(set(s)) == 10
    assert playlists.spread_sample(ids[:3], 10) == ids[:3]
    assert playlists.spread_sample(ids, 0) == []


# ---------- config follows the selected playlist ----------

def test_config_switches_language_and_glossary_paths(reg):
    import config
    playlists.add(URL, "hi", key="nptel")
    try:
        config.select_playlist("nptel")
        assert config.TARGET_LANGUAGE == "hi"
        assert config.TRANSCRIPT_LANGS == ["hi", "hi-IN"]
        assert config.PLAYLIST_ID == PID
        assert config.ALIAS_TABLE_PATH.endswith(f"{PID}/aliases.csv".replace("/", __import__("os").sep))
        config.select_playlist(None)
        assert config.TARGET_LANGUAGE == "te" and config.PLAYLIST_ID == "PLD0OXwSZSSA8"
    finally:
        config.select_playlist(None)


# ---------- unavailable videos ----------

def _stages(tmp_path, calls, private=()):
    def run_dl(v):
        calls.append(("downloaded", v))
        if v in private:
            raise Unavailable("ERROR: Private video. Sign in if you've been granted access")
        atomic_write_json(str(tmp_path / f"{v}.dl"), {})

    def run_asr(v):
        calls.append(("asr_done", v))
        atomic_write_json(str(tmp_path / f"{v}.asr"), {})

    return {"downloaded": Stage("downloaded", lambda v: [str(tmp_path / f"{v}.dl")], run_dl, lambda: {}),
            "asr_done": Stage("asr_done", lambda v: [str(tmp_path / f"{v}.asr")], run_asr, lambda: {})}


def test_private_video_is_unavailable_and_never_retried(tmp_path):
    m, calls = Manifest(str(tmp_path / "m.db")), []
    st = _stages(tmp_path, calls, private={"P"})
    assert process_video(m, st, "P", "asr_done") == {"downloaded": "unavailable"}
    assert process_video(m, st, "P", "asr_done") == {"downloaded": "unavailable"}
    assert calls == [("downloaded", "P")]                        # second run did not try again
    assert m.summary()["downloaded"]["unavailable"] == 1
    assert m.failures(status="unavailable")[0]["video_id"] == "P"
    run_stage(m, st["downloaded"], "P", retry_failed=True)       # --retry-failed checks it again
    assert len(calls) == 2


def test_old_failed_rows_become_unavailable_with_retry_failed(tmp_path):
    m, calls = Manifest(str(tmp_path / "m.db")), []
    st = _stages(tmp_path, calls, private={"P"})
    for _ in range(3):
        m.mark_failed("P", "downloaded", __import__("manifest").fingerprint({}), "TimeoutError")
    assert run_stage(m, st["downloaded"], "P") == "gave_up"
    assert run_stage(m, st["downloaded"], "P", retry_failed=True) == "unavailable"
    assert m.failures() == []


def test_summary_and_failures_filter_by_ids(tmp_path):
    m, calls = Manifest(str(tmp_path / "m.db")), []
    st = _stages(tmp_path, calls, private={"P"})
    for v in ("A", "B", "P"):
        process_video(m, st, v, "asr_done")
    assert m.summary(["A"])["asr_done"]["done"] == 1
    assert m.summary(["A", "B", "P"])["downloaded"] == {"done": 2, "failed": 0, "unavailable": 1}
    assert m.summary([])["downloaded"]["done"] == 0
    assert m.failures(["A"], "unavailable") == []


def test_permanent_error_detection():
    from downloader import is_permanent_error
    assert is_permanent_error("ERROR: [youtube] abc: Private video. Sign in if you've been granted access")
    assert is_permanent_error("ERROR: [youtube] abc: Video unavailable. This video has been removed by the uploader")
    assert not is_permanent_error("ERROR: unable to download video data: HTTP Error 403: Forbidden")
    assert not is_permanent_error("timed out")


# ---------- runner ----------

@pytest.fixture
def runner(tmp_path, reg, monkeypatch):
    import config
    monkeypatch.setattr(config, "MANIFEST_PATH", str(tmp_path / "manifest.db"))
    monkeypatch.setattr(config, "ASR_OUTPUT_DIR", str(tmp_path / "asr"))
    monkeypatch.setattr(config, "CAPTIONS_DIR", str(tmp_path / "caps"))
    monkeypatch.setattr(config, "_p", lambda *parts: str(tmp_path.joinpath(*parts)))
    monkeypatch.setitem(sys.modules, "config", config)
    monkeypatch.delitem(sys.modules, "run_pipeline", raising=False)
    import downloader
    import run_pipeline

    videos = {"PLD0OXwSZSSA8": [f"te{i:09d}" for i in range(12)], PID: [f"hi{i:09d}" for i in range(4)]}

    def fake_list(pid):
        return {"playlist_id": pid, "playlist_name": "NPTEL Data Structures",
                "videos": [{"id": v, "title": "Lecture", "duration": 600} for v in videos[pid]]}

    monkeypatch.setattr(downloader, "list_playlist", fake_list)
    yield run_pipeline, videos
    config.select_playlist(None)


def _record_stages(tmp_path, seen):
    import config

    def mk(name):
        def run(v):
            seen.append((name, v, config.TARGET_LANGUAGE))
            atomic_write_json(str(tmp_path / f"{v}.{name}"), {})
        return Stage(name, lambda v: [str(tmp_path / f"{v}.{name}")], run, lambda: {})
    return {"downloaded": mk("downloaded"), "asr_done": mk("asr_done")}


def test_add_playlist_requires_lang_for_english_titles(runner, capsys):
    run_pipeline, _ = runner
    assert run_pipeline.main(["--add-playlist", URL]) == 2
    assert "--lang" in capsys.readouterr().err
    assert run_pipeline.main(["--add-playlist", URL, "--lang", "hi", "--key", "nptel"]) == 0
    assert playlists.resolve("nptel")[1]["language"] == "hi"


def test_each_playlist_runs_in_its_own_language(runner, tmp_path):
    run_pipeline, videos = runner
    assert run_pipeline.main(["--add-playlist", URL, "--lang", "hi", "--key", "nptel"]) == 0
    seen = []
    assert run_pipeline.main(["--playlist", "nptel"], stages=_record_stages(tmp_path, seen)) == 0
    assert {lang for _, _, lang in seen} == {"hi"} and len(seen) == 8
    seen.clear()
    assert run_pipeline.main(["--playlist", "mb", "--limit", "2"], stages=_record_stages(tmp_path, seen)) == 0
    assert {lang for _, _, lang in seen} == {"te"}


def test_sample_goes_first_and_stays_fixed(runner, tmp_path):
    import config
    run_pipeline, videos = runner
    seen = []
    assert run_pipeline.main(["--playlist", "mb", "--sample", "4", "--limit", "4"],
                             stages=_record_stages(tmp_path, seen)) == 0
    first = [v for s, v, _ in seen if s == "downloaded"]
    assert first == playlists.spread_sample(videos["PLD0OXwSZSSA8"], 4)
    saved = run_pipeline.read_ids(config.SAMPLE_IDS_PATH)
    assert saved == first
    seen.clear()
    assert run_pipeline.main(["--playlist", "mb", "--sample", "6", "--limit", "4"],
                             stages=_record_stages(tmp_path, seen)) == 0
    assert run_pipeline.read_ids(config.SAMPLE_IDS_PATH) == first   # not re-drawn


def test_status_is_per_playlist_and_reports_completeness(runner, tmp_path, capsys):
    run_pipeline, _ = runner
    run_pipeline.main(["--playlist", "mb"], stages=_record_stages(tmp_path, []))
    capsys.readouterr()
    assert run_pipeline.main(["--playlist", "mb", "--status"]) == 0
    out = capsys.readouterr().out
    assert "12 videos" in out and "COMPLETE" in out and "INCOMPLETE" not in out
