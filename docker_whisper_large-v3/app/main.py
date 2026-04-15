from __future__ import annotations

import asyncio
import base64
import os
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

from .settings import settings


class Pcm16Request(BaseModel):
    audio_base64: str
    sample_rate: int = 16000
    language: str | None = None
    beam_size: int | None = None


class WhisperEngine:
    def __init__(self) -> None:
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._model = WhisperModel(
                settings.whisper_model_dir,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )

    async def transcribe_pcm16(
        self,
        pcm16: bytes,
        sample_rate: int,
        language: str | None = None,
        beam_size: int | None = None,
    ) -> str:
        if not pcm16:
            return ""
        return await asyncio.to_thread(
            self._transcribe_sync,
            pcm16,
            sample_rate,
            language or settings.whisper_language,
            beam_size or settings.whisper_beam_size,
        )

    def _transcribe_sync(self, pcm16: bytes, sample_rate: int, language: str, beam_size: int) -> str:
        self.load()
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        with self._infer_lock:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
            )
        return "".join(segment.text for segment in segments).strip()


os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
app = FastAPI(title=settings.app_name)
engine = WhisperEngine()


@app.on_event("startup")
async def startup_event() -> None:
    if settings.whisper_preload:
        await asyncio.to_thread(engine.load)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.app_name,
        "model_dir": settings.whisper_model_dir,
        "model_loaded": engine._model is not None,
    }


@app.post("/transcribe/pcm16")
async def transcribe_pcm16(payload: Pcm16Request) -> dict:
    try:
        pcm16 = base64.b64decode(payload.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid audio_base64: {exc}") from exc

    try:
        text = await engine.transcribe_pcm16(
            pcm16,
            sample_rate=payload.sample_rate,
            language=payload.language,
            beam_size=payload.beam_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"ok": True, "text": text}
