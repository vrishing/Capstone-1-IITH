import sys
import types

import pytest

from manifest import Stage, atomic_write_json


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = types.ModuleType("config")
    c.ASR_OUTPUT_DIR = str(tmp_path / "asr")
    c.CAPTIONS_DIR = str(tmp_path / "caps")
    c.TRANSCRIPT_LANGS = ["te"]
    c.TARGET_LANGUAGE = "te"
    c.HF_TOKEN = None
    c.ASR_DECODER = "ctc"
    c.ASR_MAX_CHUNK_S = 8.0
    c.ASR_TOP_DB = 35.0
    c.ASR_PAD_S = 0.2
    c.MANIFEST_PATH = str(tmp_path / "manifest.db")
    c.MAX_ATTEMPTS = 3
    monkeypatch.setitem(sys.modules, "config", c)
    monkeypatch.delitem(sys.modules, "run_pipeline", raising=False)
    return c


def fake_stages(calls):
    def mk(name):
        def run(v):
            calls.append((name, v))
            atomic_write_json(f"{name}_{v}.json".replace("/", "_"), {})
        return run
    return mk


def test_missing_config_names_are_reported(cfg, tmp_path, capsys):
    del cfg.ASR_DECODER
    import run_pipeline
    assert run_pipeline.main(["--status"]) == 2
    assert "ASR_DECODER" in capsys.readouterr().err


def test_ids_file_parsing_and_resume(cfg, tmp_path, capsys):
    import run_pipeline
    ids = tmp_path / "ids.txt"
    ids.write_text("# dev subset\nAAAAAAAAAAA\nhttps://www.youtube.com/watch?v=BBBBBBBBBBB\n\nAAAAAAAAAAA\n")
    assert run_pipeline.read_ids(str(ids)) == ["AAAAAAAAAAA", "BBBBBBBBBBB"]

    calls = []

    def stage(name):
        def run(v):
            calls.append((name, v))
            atomic_write_json(str(tmp_path / f"{v}.{name}.json"), {"x": 1})
        return Stage(name, lambda v: [str(tmp_path / f"{v}.{name}.json")], run, lambda: {"s": 1})

    stages = {"downloaded": stage("downloaded"), "asr_done": stage("asr_done")}
    assert run_pipeline.main(["--ids", str(ids)], stages=stages) == 0
    assert len(calls) == 4
    assert run_pipeline.main(["--ids", str(ids)], stages=stages) == 0
    assert len(calls) == 4                           # second run skipped everything
    assert "asr_done" in capsys.readouterr().out


def test_unbuilt_stage_is_refused(cfg, tmp_path, capsys):
    import run_pipeline
    (tmp_path / "ids.txt").write_text("AAAAAAAAAAA\n")
    assert run_pipeline.main(["--ids", str(tmp_path / "ids.txt"), "--through", "indexed"],
                             stages={}) == 2
    assert "not implemented" in capsys.readouterr().err
