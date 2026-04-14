from __future__ import annotations

from dataclasses import dataclass
from typing import List

import webrtcvad


@dataclass
class VadEvent:
    event_type: str
    pcm16: bytes


class FrameVadSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        start_trigger_frames: int = 4,
        end_trigger_frames: int = 10,
        max_segment_ms: int = 15000,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 for webrtcvad")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.start_trigger_frames = start_trigger_frames
        self.end_trigger_frames = end_trigger_frames
        self.max_segment_frames = max(1, max_segment_ms // frame_ms)

        self._vad = webrtcvad.Vad(aggressiveness)
        self._remainder = bytearray()

        self._in_speech = False
        self._speech_count = 0
        self._silence_count = 0

        self._pre_roll = bytearray()
        self._current = bytearray()

    def process_pcm(self, pcm16: bytes) -> List[VadEvent]:
        self._remainder.extend(pcm16)
        events: List[VadEvent] = []

        while len(self._remainder) >= self.frame_bytes:
            frame = bytes(self._remainder[: self.frame_bytes])
            del self._remainder[: self.frame_bytes]

            is_speech = self._vad.is_speech(frame, self.sample_rate)

            if not self._in_speech:
                self._handle_idle(frame, is_speech)
            else:
                maybe_event = self._handle_speech(frame, is_speech)
                if maybe_event is not None:
                    events.append(maybe_event)

        return events

    def flush(self) -> List[VadEvent]:
        events: List[VadEvent] = []
        if self._in_speech and len(self._current) > 0:
            events.append(VadEvent(event_type="segment", pcm16=bytes(self._current)))

        self._reset_state()
        self._remainder.clear()
        return events

    def current_active_pcm(self) -> bytes:
        if not self._in_speech:
            return b""
        return bytes(self._current)

    def _handle_idle(self, frame: bytes, is_speech: bool) -> None:
        self._pre_roll.extend(frame)
        max_pre_roll = self.frame_bytes * max(self.start_trigger_frames, 1)
        if len(self._pre_roll) > max_pre_roll:
            self._pre_roll = self._pre_roll[-max_pre_roll:]

        if is_speech:
            self._speech_count += 1
            if self._speech_count >= self.start_trigger_frames:
                self._in_speech = True
                self._current = bytearray(self._pre_roll)
                self._silence_count = 0
                self._pre_roll.clear()
        else:
            self._speech_count = 0

    def _handle_speech(self, frame: bytes, is_speech: bool) -> VadEvent | None:
        self._current.extend(frame)

        if is_speech:
            self._silence_count = 0
        else:
            self._silence_count += 1

        if len(self._current) // self.frame_bytes >= self.max_segment_frames:
            event = VadEvent(event_type="segment", pcm16=bytes(self._current))
            self._reset_state()
            return event

        if self._silence_count >= self.end_trigger_frames:
            trim_bytes = self._silence_count * self.frame_bytes
            speech_only = self._current[:-trim_bytes] if trim_bytes < len(self._current) else b""
            event = VadEvent(event_type="segment", pcm16=bytes(speech_only))
            self._reset_state()
            return event

        return None

    def _reset_state(self) -> None:
        self._in_speech = False
        self._speech_count = 0
        self._silence_count = 0
        self._pre_roll.clear()
        self._current.clear()
