# SenseVoice + VAD Realtime ASR Validation Project

This project provides:

- Ubuntu server (Docker): SenseVoice + VAD realtime speech-to-text
- Windows client (Python GUI): microphone streaming, wav upload, realtime transcript display

## Project layout

```text
docker_sv/
  docker-compose.yml
  README.md
  server/
    Dockerfile
    requirements.txt
    .env.example
    app/
      main.py
      asr_engine.py
      vad.py
      settings.py
  windows_client/
    app.py
    requirements.txt
    README.md
```

## 1) Server (Ubuntu + Docker)

### Prerequisites

- NVIDIA driver and CUDA runtime on host
- Docker + Docker Compose with GPU support

### Start server

```bash
cd /home/pgh/work/docker_sv
docker compose up -d --build
```

### Check service

```bash
curl -s http://127.0.0.1:19000/health
```

Expected:

```json
{"ok": true, "service": "sensevoice-vad-server"}
```

## 2) Server API

### Health

- `GET /health`

### WAV upload

- `POST /transcribe/file`
- multipart field name: `file`
- in this MVP, 16kHz wav is expected

Example:

```bash
curl -X POST http://127.0.0.1:19000/transcribe/file \
  -F "file=@./test.wav"
```

### Realtime WebSocket

- URL: `ws://<host>:19000/ws/transcribe`

Commands:

- start: `{"type":"start", "sample_rate":16000, "vad_end_ms":2000, "vad_min_segment_ms":1400, "auto_vad_adapt":true}`
- stop: `{"type":"stop"}`

Notes:

- `vad_end_ms` controls silence duration before split (300~3000 ms).
- `vad_min_segment_ms` controls minimum valid segment duration (400~4000 ms).
- `auto_vad_adapt` enables automatic server-side switching between short-sentence and noise-suppression behavior.

Additional websocket event:

- `adaptive_mode`: server-side auto mode switch notification with tuned `vad_end_ms` and `vad_min_segment_ms`.

Sequence mapping:

- `partial` and `final` include `seq`.
- `partial.seq` represents the in-progress utterance id.
- `final.seq` represents finalized utterance id for one-to-one comparison.

Adaptive mode stats:

- `GET /stats/adaptive?minutes=10`
- returns recent window durations for `short|balanced|noise`, switch count and tracked sessions.
- `GET /stats/adaptive/session/{session_id}?minutes=10`
- returns per-session durations and switch count for the given session.

Audio payload:

- binary frames
- PCM16 mono 16kHz

Server events:

- `ready`
- `started`
- `partial` (intermediate transcript)
- `final` (segment final transcript)
- `stopped`
- `warning` / `error`

### Runtime rewrite dictionary

- `GET /config/rewrite`
- `POST /config/rewrite`
- rewrite rules are persisted to `server/config/rewrite_rules.txt` and auto-loaded on startup

Request body for update:

```json
{
  "text_rewrite": "寨上=>站上;三脚痛=>三角筒;新判断=>请判断"
}
```

## 3) Windows client (validation)

See `windows_client/README.md`.

Quick start on Windows:

```bash
cd windows_client
pip install -r requirements.txt
python app.py
```

GUI includes:

- Realtime transcript view (partial + final)
- WAV upload test
- Microphone start/stop + realtime streaming
- Connection status and logs

## 4) Validation flow recommendation

1. File test first (WAV upload)
2. Realtime WAV stream test
3. Microphone realtime test from Windows to Ubuntu
4. Collect latency and transcript quality observations

## Notes

- This is an MVP for early validation.
- If GPU memory is occupied, service auto-falls back to CPU by `SENSEVOICE_FALLBACK_TO_CPU=true`.
- For production, add authentication, TLS, metrics, and request/session tracing.
- For stronger Sichuan-accent performance, add domain hotwords and task-specific tuning data.
