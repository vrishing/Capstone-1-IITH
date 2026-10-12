import numpy as np

from chunking import SR, chunk_by_silence, load_audio
import soundfile as sf


def speech(seconds, seed=0):
    return np.random.default_rng(seed).normal(0, 0.3, int(seconds * SR)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_cuts_only_in_silence_and_never_overlaps():
    y = np.concatenate([silence(0.5), speech(3), silence(1), speech(4), silence(0.1), speech(2), silence(0.5)])
    chunks = chunk_by_silence(y, max_chunk_s=8.0)
    for (_, e1), (s2, _) in zip(chunks, chunks[1:]):
        assert e1 <= s2 + 1e-9                      # no overlap, even with a 0.1 s gap
    speech_idx = np.nonzero(np.abs(y) > 0)[0]
    covered = np.zeros(len(y), bool)
    for s, e in chunks:
        covered[int(s * SR):int(e * SR)] = True
    assert covered[speech_idx].all()                # every speech sample is inside some chunk


def test_chunks_respect_max_length():
    y = np.concatenate([speech(3), silence(0.6), speech(3), silence(0.6), speech(3), silence(0.6), speech(3)])
    chunks = chunk_by_silence(y, max_chunk_s=8.0, pad_s=0.2)
    assert len(chunks) >= 2
    assert all(e - s <= 8.0 + 0.4 + 1e-6 for s, e in chunks)


def test_long_continuous_speech_is_split_at_the_quiet_dip():
    y = np.concatenate([speech(9, 1), speech(0.25, 2) * 0.002, speech(9, 3)])  # near-silent dip at ~9 s
    chunks = chunk_by_silence(y, max_chunk_s=10.0, top_db=60.0)
    assert len(chunks) >= 2
    assert all(e - s <= 10.0 + 0.4 + 1e-6 for s, e in chunks)
    assert chunks[0][1] < 11.0                      # first cut lands near the dip, not at an arbitrary second


def test_all_silence_returns_no_chunks():
    assert chunk_by_silence(silence(5), max_chunk_s=8.0) == []


def test_load_audio_downmixes_and_resamples(tmp_path):
    stereo = np.stack([speech(1), speech(1, 5)], axis=1)
    sf.write(str(tmp_path / "s.wav"), stereo, 44100)
    y = load_audio(str(tmp_path / "s.wav"))
    assert y.ndim == 1 and y.dtype == np.float32
    assert abs(len(y) - SR * len(stereo) / 44100) < 5
