from __future__ import annotations

import io
import json
import re
import time
import uuid
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .asr_engine import SenseVoiceEngine
from .settings import settings
from .vad import FrameVadSegmenter
from .rtsp_ws import router as rtsp_ws_router

app = FastAPI(title=settings.app_name)
app.include_router(rtsp_ws_router)


class RewriteUpdateRequest(BaseModel):
    text_rewrite: str


class LearningConfirmRequest(BaseModel):
    session_id: str = ""
    seq: int
    partial_text: str
    final_text: str
    accepted: bool = True


class AdaptiveStatsResponse(BaseModel):
    ok: bool
    minutes: int
    generated_at: float
    durations_sec: dict
    switch_count: int
    tracked_sessions: int


class AdaptiveSessionStatsResponse(BaseModel):
    ok: bool
    session_id: str
    minutes: int
    generated_at: float
    durations_sec: dict
    switch_count: int
    tracked: bool


@dataclass
class AdaptiveSplitTuner:
    sample_rate: int
    frame_ms: int
    base_end_ms: int
    base_min_ms: int
    speech_rms_threshold: float
    mode: str = "balanced"
    speechy_score: float = 0.0
    noisy_score: float = 0.0

    def observe(self, pcm16: bytes, accepted: bool) -> tuple[bool, str, int, int]:
        rms = self._rms(pcm16)
        duration_ms = int(len(pcm16) / 2 / self.sample_rate * 1000)

        self.speechy_score *= 0.92
        self.noisy_score *= 0.92

        if accepted:
            if duration_ms < 900:
                self.speechy_score += 0.6
            else:
                self.speechy_score *= 0.8
                self.noisy_score *= 0.8
        else:
            if rms >= self.speech_rms_threshold:
                self.speechy_score += 1.2
            else:
                self.noisy_score += 1.2

        target_mode = "balanced"
        if self.speechy_score >= self.noisy_score + 2.0:
            target_mode = "short"
        elif self.noisy_score >= self.speechy_score + 2.0:
            target_mode = "noise"

        changed = target_mode != self.mode
        self.mode = target_mode
        end_ms, min_ms = self.mode_params()
        return changed, self.mode, end_ms, min_ms

    def mode_params(self) -> tuple[int, int]:
        if self.mode == "short":
            end_ms = max(800, int(self.base_end_ms * 0.7))
            min_ms = max(350, int(self.base_min_ms * 0.6))
            return end_ms, min_ms
        if self.mode == "noise":
            end_ms = min(3000, int(self.base_end_ms * 1.35))
            min_ms = min(2200, int(self.base_min_ms * 1.6))
            return end_ms, min_ms
        return self.base_end_ms, self.base_min_ms

    def _rms(self, pcm16: bytes) -> float:
        if not pcm16:
            return 0.0
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))


class AdaptiveModeStatsTracker:
    def __init__(self, max_keep_hours: int = 24) -> None:
        self._events = deque()
        self._max_keep_sec = max_keep_hours * 3600

    def mark(self, session_id: str, mode: str) -> None:
        now = time.time()
        self._events.append((now, session_id, mode))
        self._prune(now)

    def summarize(self, minutes: int) -> dict:
        now = time.time()
        self._prune(now)

        window_sec = max(60, int(minutes) * 60)
        cutoff = now - window_sec

        durations = {"short": 0.0, "balanced": 0.0, "noise": 0.0}
        by_session = {}

        for ts, sid, mode in list(self._events):
            if sid not in by_session:
                by_session[sid] = {"before": None, "inside": []}
            if ts < cutoff:
                by_session[sid]["before"] = (ts, mode)
            else:
                by_session[sid]["inside"].append((ts, mode))

        switch_count = 0
        tracked_sessions = 0

        for sid_data in by_session.values():
            before = sid_data["before"]
            inside = sid_data["inside"]
            points = []

            if before is not None:
                points.append((cutoff, before[1]))
            points.extend(inside)

            if not points:
                continue

            tracked_sessions += 1
            if len(inside) > 1:
                switch_count += len(inside) - 1

            for idx, (start_ts, mode) in enumerate(points):
                end_ts = points[idx + 1][0] if idx + 1 < len(points) else now
                if end_ts <= start_ts:
                    continue
                if mode not in durations:
                    durations[mode] = 0.0
                durations[mode] += end_ts - start_ts

        return {
            "ok": True,
            "minutes": int(minutes),
            "generated_at": now,
            "durations_sec": {k: round(v, 3) for k, v in durations.items()},
            "switch_count": switch_count,
            "tracked_sessions": tracked_sessions,
        }

    def summarize_session(self, session_id: str, minutes: int) -> dict:
        now = time.time()
        self._prune(now)

        window_sec = max(60, int(minutes) * 60)
        cutoff = now - window_sec

        durations = {"short": 0.0, "balanced": 0.0, "noise": 0.0}

        before = None
        inside = []
        for ts, sid, mode in list(self._events):
            if sid != session_id:
                continue
            if ts < cutoff:
                before = (ts, mode)
            else:
                inside.append((ts, mode))

        points = []
        if before is not None:
            points.append((cutoff, before[1]))
        points.extend(inside)

        if not points:
            return {
                "ok": True,
                "session_id": session_id,
                "minutes": int(minutes),
                "generated_at": now,
                "durations_sec": durations,
                "switch_count": 0,
                "tracked": False,
            }

        for idx, (start_ts, mode) in enumerate(points):
            end_ts = points[idx + 1][0] if idx + 1 < len(points) else now
            if end_ts <= start_ts:
                continue
            if mode not in durations:
                durations[mode] = 0.0
            durations[mode] += end_ts - start_ts

        switch_count = max(0, len(inside) - 1)
        return {
            "ok": True,
            "session_id": session_id,
            "minutes": int(minutes),
            "generated_at": now,
            "durations_sec": {k: round(v, 3) for k, v in durations.items()},
            "switch_count": switch_count,
            "tracked": True,
        }

    def _prune(self, now: float) -> None:
        cutoff = now - self._max_keep_sec
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


@app.on_event("startup")
async def startup_event() -> None:
    app.state.asr = SenseVoiceEngine(
        model_name=settings.sensevoice_model,
        device=settings.sensevoice_device,
        fallback_to_cpu=settings.sensevoice_fallback_to_cpu,
        language=settings.sensevoice_language,
        hotwords=settings.sensevoice_hotwords,
        text_rewrite=settings.sensevoice_text_rewrite,
        use_itn=settings.sensevoice_use_itn,
        batch_size_s=settings.sensevoice_batch_size_s,
    )
    app.state.asr.load()
    app.state.mode_stats = AdaptiveModeStatsTracker(max_keep_hours=24)
    app.state.rewrite_learn_counts = {}
    app.state.rewrite_learn_threshold = 3
    init_rewrite_rules_from_file()


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": settings.app_name}


@app.get("/config/rewrite")
async def get_rewrite() -> dict:
    rules = app.state.asr.get_rewrite_rules()
    return {
        "ok": True,
        "text_rewrite": ";".join([f"{k}=>{v}" for k, v in rules]),
        "rules": [{"src": k, "dst": v} for k, v in rules],
    }


@app.post("/config/rewrite")
async def update_rewrite(payload: RewriteUpdateRequest) -> dict:
    rules = app.state.asr.set_rewrite_rules(payload.text_rewrite)
    persist_rewrite_rules(rules)
    return {
        "ok": True,
        "text_rewrite": ";".join([f"{k}=>{v}" for k, v in rules]),
        "rules": [{"src": k, "dst": v} for k, v in rules],
    }


@app.post("/learning/confirm")
async def learning_confirm(payload: LearningConfirmRequest) -> dict:
    if not payload.accepted:
        return {"ok": True, "accepted": False, "extracted": [], "added": []}

    extracted = extract_rewrite_candidates(payload.partial_text, payload.final_text)
    if not extracted:
        return {
            "ok": True,
            "accepted": True,
            "extracted": [],
            "added": [],
            "threshold": app.state.rewrite_learn_threshold,
        }

    existing_rules = app.state.asr.get_rewrite_rules()
    existing_map = {src: dst for src, dst in existing_rules}
    learn_counts = app.state.rewrite_learn_counts
    threshold = int(app.state.rewrite_learn_threshold)
    added: list[tuple[str, str]] = []
    hits: list[dict] = []

    for src, dst in extracted:
        key = (src, dst)
        learn_counts[key] = int(learn_counts.get(key, 0)) + 1
        hit_count = learn_counts[key]
        hits.append({"src": src, "dst": dst, "count": hit_count})

        if hit_count < threshold:
            continue
        if src in existing_map:
            continue
        existing_map[src] = dst
        existing_rules.append((src, dst))
        added.append((src, dst))

    if added:
        merged_text = ";".join([f"{src}=>{dst}" for src, dst in existing_rules])
        rules = app.state.asr.set_rewrite_rules(merged_text)
        persist_rewrite_rules(rules)

    return {
        "ok": True,
        "accepted": True,
        "session_id": payload.session_id,
        "seq": payload.seq,
        "threshold": threshold,
        "extracted": [{"src": s, "dst": d} for s, d in extracted],
        "hits": hits,
        "added": [{"src": s, "dst": d} for s, d in added],
    }


def extract_rewrite_candidates(partial_text: str, final_text: str) -> list[tuple[str, str]]:
    src = normalize_learning_text(partial_text)
    dst = normalize_learning_text(final_text)
    if not src or not dst or src == dst:
        return []

    matcher = SequenceMatcher(None, src, dst, autojunk=False)
    out: list[tuple[str, str]] = []
    seen = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        a = src[i1:i2]
        b = dst[j1:j2]
        if not a or not b:
            continue
        if a == b:
            continue
        if len(a) < 2 or len(b) < 2:
            continue
        if len(a) > 8 or len(b) > 8:
            continue
        if is_noise_token(a) or is_noise_token(b):
            continue
        pair = (a, b)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)

    return out


def normalize_learning_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "", str(text or ""))
    cleaned = re.sub(r"[，。！？!?；;、,.~`'\"“”‘’（）()【】\[\]{}<>《》:：]", "", cleaned)
    return cleaned.strip()


def is_noise_token(token: str) -> bool:
    low_value = {
        "嗯",
        "啊",
        "呃",
        "额",
        "呀",
        "吧",
        "呢",
        "哦",
        "诶",
        "唉",
        "哈",
    }
    return token in low_value


def init_rewrite_rules_from_file() -> None:
    rewrite_file = Path(settings.sensevoice_rewrite_file)
    rewrite_file.parent.mkdir(parents=True, exist_ok=True)

    text_from_file = ""
    if rewrite_file.exists():
        text_from_file = rewrite_file.read_text(encoding="utf-8").strip()

    if text_from_file:
        normalized = normalize_rewrite_text(text_from_file)
        rules = app.state.asr.set_rewrite_rules(normalized)
        persist_rewrite_rules(rules)
        return

    # Fall back to env default and persist it for next starts.
    rules = app.state.asr.set_rewrite_rules(settings.sensevoice_text_rewrite)
    persist_rewrite_rules(rules)


def normalize_rewrite_text(raw: str) -> str:
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks.extend([x.strip() for x in line.split(";") if x.strip()])
    return ";".join(chunks)


def persist_rewrite_rules(rules: list[tuple[str, str]]) -> None:
    rewrite_file = Path(settings.sensevoice_rewrite_file)
    rewrite_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{src}=>{dst}" for src, dst in rules]
    rewrite_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.get("/stats/adaptive", response_model=AdaptiveStatsResponse)
async def get_adaptive_stats(minutes: int = 10) -> dict:
    return app.state.mode_stats.summarize(minutes=max(1, minutes))


@app.get("/stats/adaptive/session/{session_id}", response_model=AdaptiveSessionStatsResponse)
async def get_adaptive_stats_by_session(session_id: str, minutes: int = 10) -> dict:
    return app.state.mode_stats.summarize_session(session_id=session_id, minutes=max(1, minutes))


@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        data, sample_rate = sf.read(io.BytesIO(content), dtype="int16", always_2d=False)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"invalid wav/audio file: {exc}") from exc

    if data.ndim > 1:
        data = data[:, 0]

    if sample_rate != 16000:
        data = resample_to_16k(data, sample_rate)
        sample_rate = 16000

    pcm16 = data.astype("int16").tobytes()
    text = await app.state.asr.transcribe_pcm16(pcm16=pcm16, sample_rate=sample_rate)

    return JSONResponse(
        {
            "ok": True,
            "text": text,
            "sample_rate": sample_rate,
            "bytes": len(pcm16),
        }
    )


def resample_to_16k(data, src_sr: int):
    if src_sr == 16000:
        return data

    import math

    g = math.gcd(src_sr, 16000)
    up = 16000 // g
    down = src_sr // g

    resampled = resample_poly(data.astype("float32"), up=up, down=down)
    return resampled.astype("int16")


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()

    session_id = uuid.uuid4().hex[:12]
    sample_rate = 16000
    segmenter = build_segmenter(sample_rate=sample_rate)
    base_vad_end_ms = getattr(segmenter, "vad_end_ms", settings.vad_end_trigger_frames * settings.vad_frame_ms)
    min_segment_ms = settings.vad_min_segment_ms
    min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
    auto_adapt_enabled = settings.vad_auto_adapt_enabled
    tuner = AdaptiveSplitTuner(
        sample_rate=sample_rate,
        frame_ms=settings.vad_frame_ms,
        base_end_ms=base_vad_end_ms,
        base_min_ms=min_segment_ms,
        speech_rms_threshold=settings.vad_auto_speech_rms_threshold,
    )
    last_partial_ts = 0.0
    final_seq = 0

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "sample_rate": sample_rate,
            "message": "send start command then pcm16 binary chunks",
        }
    )

    try:
        running = False
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                running, sample_rate, segmenter, started, final_seq = await _handle_text_message(
                    websocket=websocket,
                    text=message["text"],
                    running=running,
                    sample_rate=sample_rate,
                    segmenter=segmenter,
                    min_segment_ms=min_segment_ms,
                    final_seq=final_seq,
                )
                if started:
                    app.state.mode_stats.mark(session_id, getattr(segmenter, "adaptive_mode", "balanced"))
                min_segment_ms = getattr(segmenter, "min_segment_ms", min_segment_ms)
                base_vad_end_ms = getattr(segmenter, "vad_end_ms", base_vad_end_ms)
                auto_adapt_enabled = getattr(segmenter, "auto_adapt_enabled", auto_adapt_enabled)
                tuner = AdaptiveSplitTuner(
                    sample_rate=sample_rate,
                    frame_ms=settings.vad_frame_ms,
                    base_end_ms=base_vad_end_ms,
                    base_min_ms=min_segment_ms,
                    speech_rms_threshold=settings.vad_auto_speech_rms_threshold,
                )
                min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
                continue

            if "bytes" in message and message["bytes"] is not None:
                if not running:
                    await websocket.send_json({"type": "warning", "message": "audio ignored before start"})
                    continue

                pcm = message["bytes"]
                events = segmenter.process_pcm(pcm)

                for event in events:
                    accepted = len(event.pcm16) >= min_segment_bytes
                    if not accepted:
                        if auto_adapt_enabled:
                            changed, mode, new_end_ms, new_min_ms = tuner.observe(event.pcm16, accepted=False)
                            segmenter.end_trigger_frames = max(1, int(new_end_ms // settings.vad_frame_ms))
                            segmenter.vad_end_ms = new_end_ms
                            segmenter.min_segment_ms = new_min_ms
                            segmenter.adaptive_mode = mode
                            min_segment_ms = new_min_ms
                            min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
                            if changed:
                                app.state.mode_stats.mark(session_id, mode)
                                await websocket.send_json(
                                    {
                                        "type": "adaptive_mode",
                                        "mode": mode,
                                        "vad_end_ms": new_end_ms,
                                        "vad_min_segment_ms": new_min_ms,
                                        "session_id": session_id,
                                    }
                                )
                        continue

                    text = await app.state.asr.transcribe_pcm16(event.pcm16, sample_rate=sample_rate)
                    if text:
                        final_seq += 1
                        await websocket.send_json(
                            {
                                "type": "final",
                                "text": text,
                                "seq": final_seq,
                                "session_id": session_id,
                            }
                        )

                    if auto_adapt_enabled:
                        changed, mode, new_end_ms, new_min_ms = tuner.observe(event.pcm16, accepted=True)
                        segmenter.end_trigger_frames = max(1, int(new_end_ms // settings.vad_frame_ms))
                        segmenter.vad_end_ms = new_end_ms
                        segmenter.min_segment_ms = new_min_ms
                        segmenter.adaptive_mode = mode
                        min_segment_ms = new_min_ms
                        min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
                        if changed:
                            app.state.mode_stats.mark(session_id, mode)
                            await websocket.send_json(
                                {
                                    "type": "adaptive_mode",
                                    "mode": mode,
                                    "vad_end_ms": new_end_ms,
                                    "vad_min_segment_ms": new_min_ms,
                                    "session_id": session_id,
                                }
                            )

                now = time.monotonic()
                active_pcm = segmenter.current_active_pcm()
                min_partial_bytes = int(sample_rate * settings.partial_min_audio_ms / 1000) * 2
                if (
                    active_pcm
                    and len(active_pcm) >= min_partial_bytes
                    and (now - last_partial_ts) >= settings.partial_interval_sec
                ):
                    partial_text = await app.state.asr.transcribe_pcm16(active_pcm, sample_rate=sample_rate)
                    if partial_text:
                        await websocket.send_json(
                            {
                                "type": "partial",
                                "text": partial_text,
                                "seq": final_seq + 1,
                                "session_id": session_id,
                            }
                        )
                        last_partial_ts = now

    except WebSocketDisconnect:
        return


def build_segmenter(sample_rate: int) -> FrameVadSegmenter:
    seg = FrameVadSegmenter(
        sample_rate=sample_rate,
        frame_ms=settings.vad_frame_ms,
        aggressiveness=settings.vad_aggressiveness,
        start_trigger_frames=settings.vad_start_trigger_frames,
        end_trigger_frames=settings.vad_end_trigger_frames,
        max_segment_ms=settings.vad_max_segment_ms,
    )
    seg.vad_end_ms = settings.vad_end_trigger_frames * settings.vad_frame_ms
    seg.min_segment_ms = settings.vad_min_segment_ms
    seg.auto_adapt_enabled = settings.vad_auto_adapt_enabled
    seg.adaptive_mode = "balanced"
    return seg


def build_segmenter_with_overrides(
    sample_rate: int,
    vad_end_ms: int | None,
    min_segment_ms: int | None,
    auto_adapt_enabled: bool,
) -> FrameVadSegmenter:
    frame_ms = settings.vad_frame_ms
    end_frames = settings.vad_end_trigger_frames
    if vad_end_ms is not None:
        end_frames = max(1, int(vad_end_ms // frame_ms))

    seg = FrameVadSegmenter(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        aggressiveness=settings.vad_aggressiveness,
        start_trigger_frames=settings.vad_start_trigger_frames,
        end_trigger_frames=end_frames,
        max_segment_ms=settings.vad_max_segment_ms,
    )
    effective_end_ms = vad_end_ms if vad_end_ms is not None else end_frames * frame_ms
    seg.vad_end_ms = int(effective_end_ms)
    seg.min_segment_ms = min_segment_ms if min_segment_ms is not None else settings.vad_min_segment_ms
    seg.auto_adapt_enabled = auto_adapt_enabled
    seg.adaptive_mode = "balanced"
    return seg


async def _handle_text_message(
    websocket: WebSocket,
    text: str,
    running: bool,
    sample_rate: int,
    segmenter: FrameVadSegmenter,
    min_segment_ms: int,
    final_seq: int,
) -> tuple[bool, int, FrameVadSegmenter, bool, int]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "invalid json command"})
        return running, sample_rate, segmenter, False, final_seq

    cmd = payload.get("type", "")
    if cmd == "start":
        requested_sr = int(payload.get("sample_rate", 16000))
        if requested_sr != 16000:
            await websocket.send_json({"type": "error", "message": "only 16kHz is supported"})
            return False, sample_rate, segmenter, False, final_seq

        requested_vad_end_ms = payload.get("vad_end_ms")
        if requested_vad_end_ms is not None:
            requested_vad_end_ms = int(requested_vad_end_ms)
            requested_vad_end_ms = max(300, min(3000, requested_vad_end_ms))

        requested_min_segment_ms = payload.get("vad_min_segment_ms")
        if requested_min_segment_ms is not None:
            requested_min_segment_ms = int(requested_min_segment_ms)
            requested_min_segment_ms = max(400, min(4000, requested_min_segment_ms))
        else:
            requested_min_segment_ms = min_segment_ms

        auto_adapt_enabled = payload.get("auto_vad_adapt", settings.vad_auto_adapt_enabled)
        auto_adapt_enabled = bool(auto_adapt_enabled)

        sample_rate = requested_sr
        running = True
        segmenter = build_segmenter_with_overrides(
            sample_rate=sample_rate,
            vad_end_ms=requested_vad_end_ms,
            min_segment_ms=requested_min_segment_ms,
            auto_adapt_enabled=auto_adapt_enabled,
        )
        await websocket.send_json(
            {
                "type": "started",
                "sample_rate": sample_rate,
                "vad_end_ms": requested_vad_end_ms,
                "vad_min_segment_ms": requested_min_segment_ms,
                "auto_vad_adapt": auto_adapt_enabled,
            }
        )
        return running, sample_rate, segmenter, True, final_seq

    if cmd == "stop":
        session_min_segment_ms = getattr(segmenter, "min_segment_ms", settings.vad_min_segment_ms)
        min_segment_bytes = int(sample_rate * session_min_segment_ms / 1000) * 2
        for event in segmenter.flush():
            if len(event.pcm16) < min_segment_bytes:
                continue
            text_out = await app.state.asr.transcribe_pcm16(event.pcm16, sample_rate=sample_rate)
            if text_out:
                final_seq += 1
                await websocket.send_json({"type": "final", "text": text_out, "seq": final_seq})

        running = False
        segmenter = build_segmenter(sample_rate=sample_rate)
        await websocket.send_json({"type": "stopped"})
        return running, sample_rate, segmenter, False, final_seq

    await websocket.send_json({"type": "warning", "message": f"unknown command: {cmd}"})
    return running, sample_rate, segmenter, False, final_seq
