import subprocess
import threading
import queue
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional

# 假设已有ASR识别接口
from app.main import asr_recognize_pcm_stream

router = APIRouter()

# 音频帧队列
class AudioStreamQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.closed = False
    def put(self, data):
        self.q.put(data)
    def get(self):
        return self.q.get()
    def close(self):
        self.closed = True
        self.q.put(None)

def ffmpeg_pull_audio(rtsp_url: str, audio_queue: AudioStreamQueue, sample_rate=16000, channels=1):
    cmd = [
        'ffmpeg', '-i', rtsp_url,
        '-vn', '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', str(channels),
        '-f', 's16le', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            data = proc.stdout.read(3200)  # 100ms @ 16kHz, 16bit, 1ch
            if not data:
                break
            audio_queue.put(data)
    finally:
        proc.terminate()
        audio_queue.close()

# 后台任务：拉流并识别
def ingest_and_recognize(rtsp_url: str):
    audio_queue = AudioStreamQueue()
    t = threading.Thread(target=ffmpeg_pull_audio, args=(rtsp_url, audio_queue))
    t.start()
    # 识别主循环
    for pcm_chunk in iter(audio_queue.get, None):
        # 这里调用已有的ASR识别接口
        asr_result = asr_recognize_pcm_stream(pcm_chunk)
        print(f"ASR: {asr_result}")
    t.join()

@router.post("/ingest/rtsp")
def ingest_rtsp(rtsp_url: str, background_tasks: BackgroundTasks):
    if not rtsp_url.startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="Invalid RTSP URL")
    background_tasks.add_task(ingest_and_recognize, rtsp_url)
    return {"status": "started", "rtsp_url": rtsp_url}
