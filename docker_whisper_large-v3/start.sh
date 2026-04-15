#!/bin/sh
set -e

if [ "${AUTO_DOWNLOAD_MODEL:-true}" = "true" ]; then
  python /app/download_model.py
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 19100
