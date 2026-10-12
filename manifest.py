"""
Idempotent per-video, per-stage manifest (SQLite).

One row per (video_id, stage). A stage is marked 'done' only AFTER its output
files are fully written (atomic temp-file + os.replace). Killing the process at
any point therefore leaves either a finished stage or no row at all, never a
half-written output that looks finished.

A settings fingerprint is stored with each row. If the current settings differ,
the stage reruns and every later stage for that video is invalidated.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

STAGES = ["downloaded", "asr_done", "corrected", "chunked", "indexed"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(settings: dict) -> str:
    blob = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------- atomic file helpers ----------

def atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj) -> None:
    # serialize first so a serialization error never leaves a temp file behind
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def clean_tmp_files(*dirs: str) -> int:
    """Delete leftover *.tmp files from a killed run. Returns how many were removed."""
    n = 0
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.tmp")):
            os.remove(p)
            n += 1
    return n


# ---------- manifest ----------

class Manifest:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS manifest(
                    video_id    TEXT NOT NULL,
                    stage       TEXT NOT NULL,
                    status      TEXT NOT NULL,          -- 'done' | 'failed'
                    fingerprint TEXT,
                    updated_at  TEXT NOT NULL,
                    error       TEXT,
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (video_id, stage))"""
            )

    def get(self, video_id: str, stage: str):
        r = self.conn.execute(
            "SELECT * FROM manifest WHERE video_id=? AND stage=?", (video_id, stage)
        ).fetchone()
        return dict(r) if r else None

    def is_done(self, video_id: str, stage: str, fp: str, outputs: list[str]) -> bool:
        row = self.get(video_id, stage)
        if not row or row["status"] != "done" or row["fingerprint"] != fp:
            return False
        # reconcile with disk: a deleted or empty output means the stage is not done
        return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in outputs)

    def failed_attempts(self, video_id: str, stage: str, fp: str) -> int:
        row = self.get(video_id, stage)
        if row and row["status"] == "failed" and row["fingerprint"] == fp:
            return row["attempts"]
        return 0

    def mark_done(self, video_id: str, stage: str, fp: str) -> None:
        later = STAGES[STAGES.index(stage) + 1:]
        with self.conn:  # one transaction
            self.conn.execute(
                """INSERT INTO manifest(video_id,stage,status,fingerprint,updated_at,error,attempts)
                   VALUES(?,?,'done',?,?,NULL,0)
                   ON CONFLICT(video_id,stage) DO UPDATE SET
                     status='done', fingerprint=excluded.fingerprint,
                     updated_at=excluded.updated_at, error=NULL, attempts=0""",
                (video_id, stage, fp, _now()),
            )
            if later:  # outputs downstream of a (re)run stage are stale
                marks = ",".join("?" * len(later))
                self.conn.execute(
                    f"DELETE FROM manifest WHERE video_id=? AND stage IN ({marks})",
                    (video_id, *later),
                )

    def mark_failed(self, video_id: str, stage: str, fp: str, error: str) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO manifest(video_id,stage,status,fingerprint,updated_at,error,attempts)
                   VALUES(?,?,'failed',?,?,?,1)
                   ON CONFLICT(video_id,stage) DO UPDATE SET
                     status='failed', updated_at=excluded.updated_at, error=excluded.error,
                     attempts=CASE WHEN manifest.fingerprint=excluded.fingerprint
                                   THEN manifest.attempts+1 ELSE 1 END,
                     fingerprint=excluded.fingerprint""",
                (video_id, stage, fp, _now(), error[:2000]),
            )

    def last_completed(self, video_id: str) -> str | None:
        last = None
        for s in STAGES:
            row = self.get(video_id, s)
            if row and row["status"] == "done":
                last = s
            else:
                break
        return last

    def summary(self) -> dict:
        out = {s: {"done": 0, "failed": 0} for s in STAGES}
        for r in self.conn.execute("SELECT stage,status,COUNT(*) c FROM manifest GROUP BY stage,status"):
            out[r["stage"]][r["status"]] = r["c"]
        return out

    def failures(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT video_id,stage,attempts,error FROM manifest WHERE status='failed' ORDER BY video_id"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- stage wrapper ----------

@dataclass
class Stage:
    name: str
    outputs: Callable[[str], list[str]]   # deterministic output paths for a video
    run: Callable[[str], None]            # writes outputs atomically; raises on failure
    settings: Callable[[], dict]          # settings that define the fingerprint


def run_stage(m: Manifest, stage: Stage, video_id: str, max_attempts: int = 3,
              retry_failed: bool = False) -> str:
    """Returns 'skipped' | 'done' | 'failed' | 'gave_up'. Checks status first; writes it last."""
    fp = fingerprint(stage.settings())
    outs = stage.outputs(video_id)
    if m.is_done(video_id, stage.name, fp, outs):
        return "skipped"
    if not retry_failed and m.failed_attempts(video_id, stage.name, fp) >= max_attempts:
        return "gave_up"
    try:
        stage.run(video_id)
        missing = [p for p in outs if not (os.path.exists(p) and os.path.getsize(p) > 0)]
        if missing:
            raise RuntimeError(f"stage {stage.name} produced no output: {missing}")
        m.mark_done(video_id, stage.name, fp)   # status written only after outputs exist
        return "done"
    except Exception as e:  # KeyboardInterrupt / SIGKILL are NOT caught: no row is written
        m.mark_failed(video_id, stage.name, fp, f"{type(e).__name__}: {e}")
        return "failed"


def process_video(m: Manifest, stages: dict[str, Stage], video_id: str, through: str,
                  max_attempts: int = 3, retry_failed: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}
    for name in STAGES[: STAGES.index(through) + 1]:
        res = run_stage(m, stages[name], video_id, max_attempts, retry_failed)
        results[name] = res
        if res in ("failed", "gave_up"):
            break   # never run a later stage on top of a failed one
    return results
