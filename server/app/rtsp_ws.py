import asyncio
import audioop
import json
import re
import subprocess
import threading
import queue
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .vad import FrameVadSegmenter
from .settings import settings
from starlette.websockets import WebSocketState
from urllib.parse import urlparse, urlunparse

router = APIRouter()
_NOISE_ONLY_TEXT_RE = re.compile(r"[\s，。！？!?,、；;：:“”‘’\"'（）()【】\[\]{}<>《》~`.-]+")
_FILLER_TEXTS = {"嗯", "啊", "呃", "额", "哦", "唉", "诶", "呀", "哈", "哎"}
_FILLER_ALTS = "|".join(sorted(_FILLER_TEXTS, key=len, reverse=True))
_FILLER_PUNCT = r"[，。！？!?,、；;：:“”‘’\"'（）()【】\[\]{}<>《》~`.-]*"
_FILLER_EDGE_RE = re.compile(rf"^(?:({_FILLER_ALTS}){_FILLER_PUNCT})+")
_FILLER_TAIL_RE = re.compile(rf"(?:{_FILLER_PUNCT}({_FILLER_ALTS}))+$")

class _AudioQueue:
    def __init__(self):
        self.q = queue.Queue()
        self._closed = False
    def put(self, data):
        if not self._closed:
            self.q.put(data)
    def get(self, timeout=None):
        return self.q.get(timeout=timeout)
    def close(self):
        self._closed = True
        self.q.put(None)

def _ffmpeg_pull_audio(rtsp_url, audio_queue, stop_event):
    cmd = [
        "ffmpeg", "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "s16le", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while not stop_event.is_set():
            data = proc.stdout.read(3200)  # 100ms, 16kHz, 16bit, 1ch
            if not data:
                break
            audio_queue.put(data)
    finally:
        proc.terminate()
        audio_queue.close()

def _map_external_to_internal(rtsp_url):
    parsed = urlparse(rtsp_url)
    if parsed.hostname == "182.150.55.26":
        new_netloc = "192.168.0.51:554"
        parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(parsed)
    return rtsp_url


class _AdaptiveSplitTuner:
    def __init__(self, sample_rate: int, frame_ms: int, base_end_ms: int, base_min_ms: int, speech_rms_threshold: float):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.base_end_ms = base_end_ms
        self.base_min_ms = base_min_ms
        self.speech_rms_threshold = speech_rms_threshold
        self.mode = "balanced"
        self.speechy_score = 0.0
        self.noisy_score = 0.0

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
        return float(audioop.rms(pcm16, 2)) / 32768.0


def _pcm16_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    return float(audioop.rms(pcm16, 2)) / 32768.0


def _build_segmenter(sample_rate: int, strategy: str) -> FrameVadSegmenter:
    if strategy in {"adaptive", "mirror_mic"}:
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
        seg.auto_adapt_enabled = True
        seg.adaptive_mode = "balanced"
        seg.rtsp_strategy = strategy
        return seg

    rtsp_frame_ms = settings.vad_frame_ms
    rtsp_end_ms = max(settings.vad_end_trigger_frames * rtsp_frame_ms, 1000)
    rtsp_end_frames = max(1, int(rtsp_end_ms // rtsp_frame_ms))
    rtsp_start_frames = max(settings.vad_start_trigger_frames, 8)
    seg = FrameVadSegmenter(
        sample_rate=sample_rate,
        frame_ms=rtsp_frame_ms,
        aggressiveness=settings.vad_aggressiveness,
        start_trigger_frames=rtsp_start_frames,
        end_trigger_frames=rtsp_end_frames,
        max_segment_ms=settings.vad_max_segment_ms,
    )
    seg.vad_end_ms = rtsp_end_ms
    seg.min_segment_ms = max(settings.vad_min_segment_ms, 1200)
    seg.auto_adapt_enabled = False
    seg.adaptive_mode = "conservative"
    seg.rtsp_strategy = "conservative"
    return seg


def _is_meaningful_text(text: str) -> bool:
    cleaned = _NOISE_ONLY_TEXT_RE.sub("", str(text or "")).strip()
    return bool(cleaned)


def _clean_final_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _FILLER_EDGE_RE.sub("", cleaned).strip()
        cleaned = _FILLER_TAIL_RE.sub("", cleaned).strip()

    cleaned_meaning = _NOISE_ONLY_TEXT_RE.sub("", cleaned).strip()
    if not cleaned_meaning or cleaned_meaning in _FILLER_TEXTS:
        return ""
    return cleaned

@router.websocket("/ws/rtsp_transcribe")
async def ws_rtsp_transcribe(ws: WebSocket):
    await ws.accept()
    app = ws.app
    sample_rate = 16000
    stop_event = threading.Event()
    audio_queue = _AudioQueue()
    vad = None
    ffmpeg_thread = None
    tuner = None
    final_seq = 0
    last_partial_ts = 0.0
    last_partial_seq = 0
    last_partial_text_by_seq: dict[int, str] = {}
    try:
        await ws.send_json({"type": "ready", "sample_rate": sample_rate, "message": "send rtsp_url json"})
        # 第一个消息应为 {"rtsp_url": "..."}
        message = await ws.receive()
        raw_text = message.get("text")
        if raw_text is None:
            await ws.send_json({"error": "missing text payload"})
            await ws.close()
            return
        msg = json.loads(raw_text)
        rtsp_url = msg.get("rtsp_url", "").strip()
        rtsp_strategy = str(msg.get("rtsp_strategy", "conservative") or "conservative").strip().lower()
        rtsp_min_rms = float(msg.get("rtsp_min_rms", settings.vad_auto_speech_rms_threshold) or 0.0)
        rtsp_min_rms = max(0.0, min(0.2, rtsp_min_rms))
        if rtsp_strategy not in {"conservative", "adaptive", "mirror_mic"}:
            rtsp_strategy = "conservative"
        print(f"[RTSP_WS] received rtsp_url={rtsp_url}")
        if not rtsp_url.startswith("rtsp://"):
            await ws.send_json({"error": "invalid rtsp url"})
            await ws.close()
            return
        internal_url = _map_external_to_internal(rtsp_url)
        vad = _build_segmenter(sample_rate=sample_rate, strategy=rtsp_strategy)
        mirror_mic_mode = rtsp_strategy == "mirror_mic"
        min_segment_ms = getattr(vad, "min_segment_ms", settings.vad_min_segment_ms)
        min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
        min_partial_bytes = int(sample_rate * settings.partial_min_audio_ms / 1000) * 2
        if getattr(vad, "auto_adapt_enabled", False):
            tuner = _AdaptiveSplitTuner(
                sample_rate=sample_rate,
                frame_ms=settings.vad_frame_ms,
                base_end_ms=getattr(vad, "vad_end_ms", settings.vad_end_trigger_frames * settings.vad_frame_ms),
                base_min_ms=min_segment_ms,
                speech_rms_threshold=settings.vad_auto_speech_rms_threshold,
            )
        await ws.send_json(
            {
                "type": "started",
                "rtsp_url": rtsp_url,
                "internal_rtsp_url": internal_url,
                "rtsp_strategy": rtsp_strategy,
                "rtsp_min_rms": rtsp_min_rms,
                "vad_end_ms": getattr(vad, "vad_end_ms", None),
                "vad_min_segment_ms": min_segment_ms,
                "auto_vad_adapt": getattr(vad, "auto_adapt_enabled", False),
            }
        )
        print(f"[RTSP_WS] started internal_rtsp_url={internal_url} strategy={rtsp_strategy}")
        ffmpeg_thread = threading.Thread(target=_ffmpeg_pull_audio, args=(internal_url, audio_queue, stop_event), daemon=True)
        ffmpeg_thread.start()
        while ws.application_state == WebSocketState.CONNECTED:
            try:
                pcm = await asyncio.to_thread(audio_queue.get, 1)
                if pcm is None:
                    break
                for event in vad.process_pcm(pcm):
                    if event.pcm16 is None:
                        continue
                    event_rms = _pcm16_rms(event.pcm16)
                    accepted = len(event.pcm16) >= min_segment_bytes
                    if accepted and event_rms < rtsp_min_rms:
                        accepted = False
                    if not accepted:
                        if tuner is not None:
                            changed, mode, new_end_ms, new_min_ms = tuner.observe(event.pcm16, accepted=False)
                            vad.end_trigger_frames = max(1, int(new_end_ms // settings.vad_frame_ms))
                            vad.vad_end_ms = new_end_ms
                            vad.min_segment_ms = new_min_ms
                            vad.adaptive_mode = mode
                            min_segment_ms = new_min_ms
                            min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
                            if changed:
                                await ws.send_json(
                                    {
                                        "type": "adaptive_mode",
                                        "mode": mode,
                                        "vad_end_ms": new_end_ms,
                                        "vad_min_segment_ms": new_min_ms,
                                        "rtsp_strategy": rtsp_strategy,
                                    }
                                )
                        continue
                    text = await app.state.asr.transcribe_pcm16(event.pcm16, sample_rate=sample_rate)
                    if text and _is_meaningful_text(text):
                        next_seq = final_seq + 1
                        if not mirror_mic_mode:
                            last_partial_text = last_partial_text_by_seq.get(next_seq, "")
                            if last_partial_seq != next_seq or last_partial_text != text:
                                print(f"[RTSP_WS][partial-before-final][{next_seq}] {text}")
                                await ws.send_json({"type": "partial", "text": text, "seq": next_seq})
                                last_partial_seq = next_seq
                                last_partial_text_by_seq[next_seq] = text
                        final_text = text if mirror_mic_mode else _clean_final_text(text)
                        if not final_text:
                            continue
                        final_seq = next_seq
                        if not mirror_mic_mode and final_text != text:
                            print(f"[RTSP_WS][final-clean][{final_seq}] raw={text} cleaned={final_text}")
                        else:
                            print(f"[RTSP_WS][final][{final_seq}] {final_text}")
                        await ws.send_json({"type": "final", "text": final_text, "seq": final_seq})
                    if tuner is not None:
                        changed, mode, new_end_ms, new_min_ms = tuner.observe(event.pcm16, accepted=True)
                        vad.end_trigger_frames = max(1, int(new_end_ms // settings.vad_frame_ms))
                        vad.vad_end_ms = new_end_ms
                        vad.min_segment_ms = new_min_ms
                        vad.adaptive_mode = mode
                        min_segment_ms = new_min_ms
                        min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
                        if changed:
                            await ws.send_json(
                                {
                                    "type": "adaptive_mode",
                                    "mode": mode,
                                    "vad_end_ms": new_end_ms,
                                    "vad_min_segment_ms": new_min_ms,
                                    "rtsp_strategy": rtsp_strategy,
                                }
                            )

                now = time.monotonic()
                active_pcm = vad.current_active_pcm()
                if (
                    active_pcm
                    and len(active_pcm) >= min_partial_bytes
                    and _pcm16_rms(active_pcm) >= rtsp_min_rms
                    and (now - last_partial_ts) >= settings.partial_interval_sec
                ):
                    partial_text = await app.state.asr.transcribe_pcm16(active_pcm, sample_rate=sample_rate)
                    if partial_text and _is_meaningful_text(partial_text):
                        next_seq = final_seq + 1
                        print(f"[RTSP_WS][partial][{next_seq}] {partial_text}")
                        await ws.send_json({"type": "partial", "text": partial_text, "seq": next_seq})
                        last_partial_seq = next_seq
                        last_partial_text_by_seq[next_seq] = partial_text
                        last_partial_ts = now
            except queue.Empty:
                continue
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"error": str(exc)})
    finally:
        stop_event.set()
        if ffmpeg_thread:
            ffmpeg_thread.join(timeout=2)
        if vad is not None and ws.application_state == WebSocketState.CONNECTED:
            min_segment_ms = getattr(vad, "min_segment_ms", settings.vad_min_segment_ms)
            min_segment_bytes = int(sample_rate * min_segment_ms / 1000) * 2
            for event in vad.flush():
                if len(event.pcm16) < min_segment_bytes:
                    continue
                text = await app.state.asr.transcribe_pcm16(event.pcm16, sample_rate=sample_rate)
                if text and _is_meaningful_text(text):
                    next_seq = final_seq + 1
                    if not mirror_mic_mode:
                        last_partial_text = last_partial_text_by_seq.get(next_seq, "")
                        if last_partial_seq != next_seq or last_partial_text != text:
                            print(f"[RTSP_WS][flush-partial-before-final][{next_seq}] {text}")
                            await ws.send_json({"type": "partial", "text": text, "seq": next_seq})
                            last_partial_seq = next_seq
                            last_partial_text_by_seq[next_seq] = text
                    final_text = text if mirror_mic_mode else _clean_final_text(text)
                    if not final_text:
                        continue
                    final_seq = next_seq
                    if not mirror_mic_mode and final_text != text:
                        print(f"[RTSP_WS][flush-final-clean][{final_seq}] raw={text} cleaned={final_text}")
                    else:
                        print(f"[RTSP_WS][flush-final][{final_seq}] {final_text}")
                    await ws.send_json({"type": "final", "text": final_text, "seq": final_seq})
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
