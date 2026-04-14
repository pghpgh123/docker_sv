import subprocess
import threading
import queue
from typing import Optional
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .vad import FrameVadSegmenter

router = APIRouter()


class RtspIngestRequest(BaseModel):
    rtsp_url: str


class _AudioQueue:
    def __init__(self) -> None:
        self.q: "queue.Queue[Optional[bytes]]" = queue.Queue()

    def put(self, data: Optional[bytes]) -> None:
        self.q.put(data)

    def get(self) -> Optional[bytes]:
        return self.q.get()


def _ffmpeg_pull_audio(rtsp_url: str, audio_q: _AudioQueue, sample_rate: int = 16000, channels: int = 1) -> None:
    cmd = [
        "ffmpeg",
        "-i",
        rtsp_url,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-t",
        "5",  # 最多拉取约 5 秒音频
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        frame_bytes = int(sample_rate * 0.1) * 2  # 100ms
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            audio_q.put(chunk)
    finally:
        proc.terminate()
        audio_q.put(None)


def _ingest_and_recognize(app, rtsp_url: str) -> str:
    sample_rate = 16000
    seg = FrameVadSegmenter(sample_rate=sample_rate)
    audio_q = _AudioQueue()
    t = threading.Thread(target=_ffmpeg_pull_audio, args=(rtsp_url, audio_q, sample_rate, 1), daemon=True)
    t.start()

    texts: list[str] = []

    try:
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            events = seg.process_pcm(chunk)
            for ev in events:
                if not ev.pcm16:
                    continue
                try:
                    text = app.state.asr._transcribe_sync(ev.pcm16, sample_rate=sample_rate)
                    if text:
                        texts.append(text)
                        print(f"[RTSP][{rtsp_url}] {text}")
                except Exception:
                    # 简单容错，避免后台任务崩溃
                    continue

                # 为避免长时间阻塞，这里限制最多处理若干段
                if len(texts) >= 5:
                    raise StopIteration
    finally:
        for ev in seg.flush():
            if not ev.pcm16:
                continue
            try:
                text = app.state.asr._transcribe_sync(ev.pcm16, sample_rate=sample_rate)
                if text:
                    texts.append(text)
                    print(f"[RTSP][{rtsp_url}] {text}")
            except Exception:
                continue
        t.join(timeout=2.0)

    # 返回合并文本，客户端可直接显示
    return " ".join(texts)


@router.post("/ingest/rtsp")
async def ingest_rtsp(payload: RtspIngestRequest, request: Request) -> dict:
    external_url = payload.rtsp_url.strip()
    if not external_url.startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="invalid rtsp url")

    # 将外网地址映射为内网地址：182.150.55.26 -> 192.168.0.51:554
    parsed = urlparse(external_url)
    internal_url = external_url
    if parsed.hostname == "182.150.55.26":
        # 保留路径和查询参数，只替换主机和端口
        new_netloc = "192.168.0.51:554"
        parsed = parsed._replace(netloc=new_netloc)
        internal_url = urlunparse(parsed)
        print(f"[RTSP] map external {external_url} -> internal {internal_url}")

    app = request.app
    text = _ingest_and_recognize(app, internal_url)
    return {"ok": True, "rtsp_url": external_url, "internal_rtsp_url": internal_url, "text": text}
