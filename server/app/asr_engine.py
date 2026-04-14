from __future__ import annotations

import asyncio
import re
import threading
from typing import Any, Iterable

import numpy as np
from funasr import AutoModel


class SenseVoiceEngine:
    def __init__(
        self,
        model_name: str,
        device: str,
        fallback_to_cpu: bool,
        language: str,
        hotwords: str,
        text_rewrite: str,
        use_itn: bool,
        batch_size_s: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.fallback_to_cpu = fallback_to_cpu
        self.language = language
        self.hotwords = hotwords.strip()
        self.text_rewrite_rules = self._parse_rewrite_rules(text_rewrite)
        self.use_itn = use_itn
        self.batch_size_s = batch_size_s
        self._model = None
        self._lock = threading.Lock()
        self._rewrite_lock = threading.Lock()
        self._tag_pattern = re.compile(r"<\|[^|]+\|>")

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            self._model = self._build_model(self.device)
        except Exception as exc:
            msg = str(exc).lower()
            should_fallback = (
                self.fallback_to_cpu
                and "cuda out of memory" in msg
                and self.device.startswith("cuda")
            )
            if not should_fallback:
                raise

            self._model = self._build_model("cpu")
            self.device = "cpu"

    def _build_model(self, device: str):
        return AutoModel(
            model=self.model_name,
            trust_remote_code=True,
            device=device,
            disable_update=True,
        )

    async def transcribe_pcm16(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        if not pcm16:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, pcm16, sample_rate)

    def _transcribe_sync(self, pcm16: bytes, sample_rate: int) -> str:
        if self._model is None:
            raise RuntimeError("SenseVoice model not loaded")

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        with self._lock:
            result = self._model.generate(
                input=audio,
                cache={},
                language=self.language,
                hotword=self.hotwords if self.hotwords else None,
                use_itn=self.use_itn,
                batch_size_s=self.batch_size_s,
            )

        return self._extract_text(result)

    def _extract_text(self, result: Any) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            text = result.get("text") or result.get("sentence") or ""
            return self._clean_text(str(text))

        if isinstance(result, Iterable):
            texts = []
            for item in result:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("sentence")
                    if text:
                        texts.append(str(text).strip())
                elif isinstance(item, str):
                    texts.append(item.strip())
            return self._clean_text(" ".join([t for t in texts if t]))

        return self._clean_text(str(result))

    def _clean_text(self, text: str) -> str:
        cleaned = self._tag_pattern.sub("", text)
        cleaned = " ".join(cleaned.split()).strip()
        rules = self.get_rewrite_rules()
        for src, dst in rules:
            cleaned = cleaned.replace(src, dst)
        return cleaned

    def _parse_rewrite_rules(self, text_rewrite: str):
        pairs = []
        for item in text_rewrite.split(";"):
            piece = item.strip()
            if not piece or "=>" not in piece:
                continue
            src, dst = piece.split("=>", 1)
            src = src.strip()
            dst = dst.strip()
            if src and dst:
                pairs.append((src, dst))
        return pairs

    def get_rewrite_rules(self):
        with self._rewrite_lock:
            return list(self.text_rewrite_rules)

    def set_rewrite_rules(self, text_rewrite: str):
        rules = self._parse_rewrite_rules(text_rewrite)
        with self._rewrite_lock:
            self.text_rewrite_rules = rules
        return rules
