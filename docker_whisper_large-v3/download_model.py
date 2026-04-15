from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from app.settings import settings


def main() -> None:
    if not settings.whisper_auto_download:
        return

    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    model_dir = Path(settings.whisper_model_dir)
    marker = model_dir / "model.bin"
    if marker.exists():
        print(f"model already present: {marker}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=settings.whisper_model_repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"model downloaded to {model_dir}")


if __name__ == "__main__":
    main()
