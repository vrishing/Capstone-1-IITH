"""
IndicConformer ASR wrapper. Differences from the old ASR_TEST.py class:
- no overlap parameters or text-merging code (chunks never overlap)
- fp32 only, no autocast
- raises on failure instead of returning an empty transcript, so the manifest
  can record the failure instead of marking a video 'done'
- returns per-chunk timestamps as `segments`
"""
from __future__ import annotations

import time

from chunking import SR, chunk_by_silence, load_audio

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerASR:
    def __init__(self, language_code: str, decoder_type: str, token: str | None,
                 max_chunk_s: float, top_db: float, pad_s: float, force_cpu: bool = False):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.language_code = language_code
        self.decoder_type = decoder_type
        self.max_chunk_s = max_chunk_s
        self.top_db = top_db
        self.pad_s = pad_s
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print(f" -> IndicConformer ({decoder_type}) on [{self.device.upper()}]")
        self.model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, token=token
        ).to(self.device)
        self.model.eval()

    def _transcribe_chunk(self, y_chunk) -> str:
        wav = self.torch.from_numpy(y_chunk).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.model(wav, self.language_code, self.decoder_type)
        out = out[0] if isinstance(out, list) else out
        return str(out).strip()

    def transcribe(self, audio_path: str, debug_html_path: str | None = None) -> dict:
        t0 = time.time()
        y = load_audio(audio_path)
        duration = len(y) / SR
        segments = []
        for start_s, end_s in chunk_by_silence(y, self.max_chunk_s, self.top_db, self.pad_s):
            if end_s - start_s < 0.2:
                continue
            text = self._transcribe_chunk(y[int(start_s * SR):int(end_s * SR)])
            if text:
                segments.append({"text": text, "start": round(start_s, 2),
                                 "duration": round(end_s - start_s, 2)})
        if debug_html_path:
            dump_debug_html([s["text"] for s in segments], debug_html_path)
        return {
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
            "duration_s": round(duration, 2),
            "latency_sec": round(time.time() - t0, 2),
            "engine": f"indiconformer_{self.decoder_type}",
        }


def dump_debug_html(chunk_texts: list[str], out_path: str) -> None:
    rows = "".join(
        f"<div style='border:1px solid #ccc;margin:8px;padding:8px;font-family:sans-serif'>"
        f"<b>Chunk {i}</b><p style='font-size:1.1em'>{t}</p></div>"
        for i, t in enumerate(chunk_texts)
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='UTF-8'></head><body>{rows}</body></html>")
