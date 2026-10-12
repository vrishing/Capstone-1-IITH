"""Helper for the kill test: a tiny pipeline with a slow fake 'asr_done' stage."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manifest import Manifest, Stage, atomic_write_json, process_video

db, outdir, log, n, delay = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
os.makedirs(outdir, exist_ok=True)


def logline(msg):
    with open(log, "a") as f:
        f.write(msg + "\n")
        f.flush()
        os.fsync(f.fileno())


def fake_download(v):
    logline(f"START downloaded {v}")
    atomic_write_json(os.path.join(outdir, f"{v}.wav.json"), {"ok": 1})


def fake_asr(v):
    logline(f"START asr_done {v}")
    time.sleep(delay)  # the window in which the test kills the process
    atomic_write_json(os.path.join(outdir, f"{v}_asr.json"), {"text": "x" * 100})
    logline(f"END asr_done {v}")


stages = {
    "downloaded": Stage("downloaded", lambda v: [os.path.join(outdir, f"{v}.wav.json")], fake_download, lambda: {"a": 1}),
    "asr_done": Stage("asr_done", lambda v: [os.path.join(outdir, f"{v}_asr.json")], fake_asr, lambda: {"d": "ctc"}),
}
m = Manifest(db)
for i in range(1, n + 1):
    process_video(m, stages, f"v{i}", "asr_done")
