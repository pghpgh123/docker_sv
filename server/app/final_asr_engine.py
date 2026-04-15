from __future__ import annotations

import asyncio
import base64
import threading

import httpx
import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperEngine:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int,
        service_url: str = "",
        request_timeout_sec: float = 180.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.service_url = service_url.rstrip("/")
        self.request_timeout_sec = request_timeout_sec
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def load(self) -> None:
        if self.service_url:
            return
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

    async def transcribe_pcm16(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        if not pcm16:
            return ""
        if self.service_url:
            return await self._transcribe_remote(pcm16, sample_rate)
        return await asyncio.to_thread(self._transcribe_sync, pcm16, sample_rate)

    async def _transcribe_remote(self, pcm16: bytes, sample_rate: int) -> str:
        payload = {
            "audio_base64": base64.b64encode(pcm16).decode("ascii"),
            "sample_rate": sample_rate,
            "language": self.language,
            "beam_size": self.beam_size,
        }
        timeout = httpx.Timeout(self.request_timeout_sec, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.service_url}/transcribe/pcm16", json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("text", "")).strip()

    def _transcribe_sync(self, pcm16: bytes, sample_rate: int) -> str:
        self.load()
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        with self._infer_lock:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = "".join(segment.text for segment in segments).strip()
        return text