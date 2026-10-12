"""
Silence-aware audio chunking for ASR (numpy + librosa only, no torch).

Rules: cut only at silence, never mid-word; split any single speech run longer
than max_chunk_s at its quietest nearby point; no overlap between chunks.
Returns (start_s, end_s) boundaries so callers keep real timestamps.
"""
from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

SR = 16000


def load_audio(path: str) -> np.ndarray:
    """Mono float32 at 16 kHz, whatever the file's format or rate."""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def find_quietest_split(y: np.ndarray, start: int, end: int,
                        window_s: float = 0.2, radius_s: float = 1.5) -> int:
    """Lowest-energy sample index near the midpoint of [start, end)."""
    mid = (start + end) // 2
    radius = int(radius_s * SR)
    lo, hi = max(start, mid - radius), min(end, mid + radius)
    win = max(1, int(window_s * SR))
    best_idx, best_e = mid, float("inf")
    for i in range(lo, max(lo, hi - win), max(1, win // 2)):
        e = float(np.sqrt(np.mean(y[i:i + win] ** 2)))
        if e < best_e:
            best_e, best_idx = e, i + win // 2
    return best_idx


def chunk_by_silence(y: np.ndarray, max_chunk_s: float, top_db: float = 35.0,
                     pad_s: float = 0.2) -> list[tuple[float, float]]:
    """Non-overlapping (start_s, end_s) chunks, each at most max_chunk_s + 2*pad_s long."""
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        return []  # digital silence (librosa would call an all-zero signal one long speech run)
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return []  # all silence

    max_samples = int(max_chunk_s * SR)

    # 1) split overlong speech runs at their quietest point
    safe: list[tuple[int, int]] = []
    for s, e in intervals:
        s, e = int(s), int(e)
        stack, pieces = [(s, e)], []
        while stack:
            a, b = stack.pop()
            if b - a <= max_samples:
                pieces.append((a, b))
            else:
                sp = find_quietest_split(y, a, b)
                if sp <= a or sp >= b:
                    sp = (a + b) // 2
                stack.append((a, sp))
                stack.append((sp, b))
        safe.extend(sorted(pieces))

    # 2) greedily group neighbouring pieces up to max_chunk_s (silence between them is kept)
    groups: list[tuple[int, int]] = []
    cur_s, cur_e = safe[0]
    for s, e in safe[1:]:
        if (e - cur_s) <= max_samples:
            cur_e = e
        else:
            groups.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    groups.append((cur_s, cur_e))

    # 3) pad each chunk with real silence, but never into a neighbour (no overlap)
    pad = int(pad_s * SR)
    total = len(y)
    out: list[tuple[float, float]] = []
    for i, (s, e) in enumerate(groups):
        prev_e = groups[i - 1][1] if i > 0 else None
        next_s = groups[i + 1][0] if i + 1 < len(groups) else None
        lo = max(0, s - pad) if prev_e is None else max(s - pad, (prev_e + s) // 2)
        hi = min(total, e + pad) if next_s is None else min(e + pad, (e + next_s) // 2)
        out.append((lo / SR, hi / SR))
    return out
