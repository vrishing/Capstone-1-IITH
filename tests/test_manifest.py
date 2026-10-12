import os
import signal
import subprocess
import sys
import time

import pytest

from manifest import (Manifest, Stage, atomic_write_json, clean_tmp_files, fingerprint,
                      process_video, run_stage)

HERE = os.path.dirname(os.path.abspath(__file__))


def make_stages(tmp_path, calls, asr_settings=None, fail_ids=()):
    def out(name):
        return lambda v: [str(tmp_path / f"{v}.{name}")]

    def runner(name):
        def run(v):
            calls.append((name, v))
            if name == "asr_done" and v in fail_ids:
                raise RuntimeError("boom")
            atomic_write_json(str(tmp_path / f"{v}.{name}"), {"v": v})
        return run

    return {
        "downloaded": Stage("downloaded", out("downloaded"), runner("downloaded"), lambda: {"a": 1}),
        "asr_done": Stage("asr_done", out("asr_done"), runner("asr_done"),
                          lambda: asr_settings or {"decoder": "ctc"}),
    }


def test_kill_during_asr_then_restart_resumes_at_right_video(tmp_path):
    db, outdir, log = str(tmp_path / "m.db"), str(tmp_path / "out"), str(tmp_path / "log.txt")
    args = [sys.executable, os.path.join(HERE, "_kill_target.py"), db, outdir, log, "5", "0.6"]
    p = subprocess.Popen(args)
    deadline = time.time() + 20
    while time.time() < deadline:  # wait until video 3 is mid-ASR
        if os.path.exists(log) and "START asr_done v3" in open(log).read():
            break
        time.sleep(0.02)
    p.send_signal(signal.SIGKILL)  # hard kill, no cleanup handlers run
    p.wait()

    m = Manifest(db)
    assert m.get("v1", "asr_done")["status"] == "done"
    assert m.get("v2", "asr_done")["status"] == "done"
    assert m.get("v3", "asr_done") is None            # killed mid-ASR: no status written
    assert not os.path.exists(os.path.join(outdir, "v3_asr.json"))

    open(log, "a").write("--- restart ---\n")
    subprocess.run(args, check=True)
    after = open(log).read().split("--- restart ---\n")[1].splitlines()
    started = [l for l in after if l.startswith("START")]
    assert "START asr_done v1" not in started and "START asr_done v2" not in started
    assert "START downloaded v1" not in started and "START downloaded v2" not in started
    assert "START asr_done v3" in started               # resumes exactly at the interrupted video
    assert all(Manifest(db).get(f"v{i}", "asr_done")["status"] == "done" for i in range(1, 6))


def test_finished_videos_are_skipped(tmp_path):
    calls = []
    m = Manifest(str(tmp_path / "m.db"))
    st = make_stages(tmp_path, calls)
    for v in ("a", "b"):
        process_video(m, st, v, "asr_done")
    n = len(calls)
    for v in ("a", "b"):
        assert process_video(m, st, v, "asr_done") == {"downloaded": "skipped", "asr_done": "skipped"}
    assert len(calls) == n


def test_status_row_without_output_file_reruns(tmp_path):
    calls = []
    m = Manifest(str(tmp_path / "m.db"))
    st = make_stages(tmp_path, calls)
    process_video(m, st, "a", "asr_done")
    os.remove(tmp_path / "a.asr_done")
    assert run_stage(m, st["asr_done"], "a") == "done"


def test_changed_settings_rerun_stage_and_invalidate_downstream(tmp_path):
    calls = []
    m = Manifest(str(tmp_path / "m.db"))
    process_video(m, make_stages(tmp_path, calls), "a", "asr_done")
    m.mark_done("a", "corrected", "fp")  # pretend a later stage had completed
    assert m.last_completed("a") == "corrected"
    calls.clear()
    res = process_video(m, make_stages(tmp_path, calls, {"decoder": "rnnt"}), "a", "asr_done")
    assert res == {"downloaded": "skipped", "asr_done": "done"}
    assert calls == [("asr_done", "a")]
    assert m.get("a", "corrected") is None             # downstream stage must rerun


def test_failure_is_recorded_run_continues_and_gives_up(tmp_path):
    calls = []
    m = Manifest(str(tmp_path / "m.db"))
    st = make_stages(tmp_path, calls, fail_ids={"bad"})
    assert process_video(m, st, "bad", "asr_done")["asr_done"] == "failed"
    assert process_video(m, st, "good", "asr_done")["asr_done"] == "done"
    assert "boom" in m.get("bad", "asr_done")["error"]
    for _ in range(3):
        process_video(m, st, "bad", "asr_done", max_attempts=3)
    assert process_video(m, st, "bad", "asr_done", max_attempts=3)["asr_done"] == "gave_up"
    assert run_stage(m, st["asr_done"], "bad", retry_failed=True) == "failed"


def test_atomic_write_leaves_old_file_intact_on_error(tmp_path):
    p = str(tmp_path / "x.json")
    atomic_write_json(p, {"v": 1})
    with pytest.raises(TypeError):
        atomic_write_json(p, {"v": object()})
    assert open(p).read().count("1") == 1
    assert not os.path.exists(p + ".tmp")


def test_clean_tmp_files(tmp_path):
    (tmp_path / "a.json.tmp").write_text("half")
    (tmp_path / "b.json").write_text("ok")
    assert clean_tmp_files(str(tmp_path)) == 1
    assert (tmp_path / "b.json").exists()


def test_fingerprint_is_stable_and_order_independent():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
