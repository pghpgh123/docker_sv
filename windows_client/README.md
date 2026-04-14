# Windows validation client (Python)

This client is for early-stage validation only.

## Features

- Realtime transcript display from server (partial and final)
- WAV upload test
- Microphone control: start/stop capture and realtime upload
- WebSocket connection control
- Silence split threshold control for realtime mode (`断句静音时长(ms)`)
- One-click presets: low latency (1000ms), balanced (2000ms), high accuracy (3000ms)
- Server can auto-adapt split behavior during runtime; mode switch is shown in client logs
- Current session ID display and one-click query for per-session adaptive stats
- Scrollable numbered rewrite-rule table with add/delete/edit operations
- Dictionary import/export buttons (text format: `src=>dst`, one rule per line)

## Quick start

1. Install Python 3.10+ on Windows.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python app.py
```

4. Set `Base URL` to your Ubuntu server, for example:

```text
http://182.150.55.26:19000
```

The app converts it internally to:

- WebSocket: `ws://182.150.55.26:19000/ws/transcribe`
- Upload API: `http://182.150.55.26:19000/transcribe/file`

## Audio format

- 16kHz
- mono
- PCM16
