# Whisper large-v3 standalone Docker

This directory runs `faster-whisper large-v3` as an independent local HTTP service.

## Start

```bash
cd /home/pgh/work/docker_sv/docker_whisper_large-v3
docker compose up -d --build
```

The container downloads the model from `hf-mirror.com` into `./models` on first startup.
The first startup can take several minutes because `large-v3` is downloaded and preloaded before the service becomes ready.

## Health

```bash
curl -s http://127.0.0.1:19100/health
```

Expected shape:

```json
{"ok": true, "service": "whisper-large-v3-service", "model_loaded": true}
```

## API

- `POST /transcribe/pcm16`
- request JSON fields:
  - `audio_base64`
  - `sample_rate`
  - `language` optional
  - `beam_size` optional

## Main service integration

Recommended startup order:

```bash
cd /home/pgh/work/docker_sv/docker_whisper_large-v3
docker compose up -d --build

cd /home/pgh/work/docker_sv
docker compose up -d --build
```

The main server connects to this service through `http://host.docker.internal:19100` by default.

## Integration

The main `docker_sv` server defaults `FASTER_WHISPER_SERVICE_URL` to `http://host.docker.internal:19100`.
If this standalone service is running, RTSP strategy `Whisper large-v3（仅最终文本）` will use it automatically.

## Storage

- model files are persisted in `./models`
- deleting `./models` forces a re-download on next startup