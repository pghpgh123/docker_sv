from pydantic_settings import BaseSettings, SettingsConfigDict


class WhisperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "whisper-large-v3-service"
    app_host: str = "0.0.0.0"
    app_port: int = 19100

    whisper_model_repo_id: str = "Systran/faster-whisper-large-v3"
    whisper_model_dir: str = "/models/faster-whisper-large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "zh"
    whisper_beam_size: int = 5
    whisper_preload: bool = True
    whisper_auto_download: bool = True
    hf_endpoint: str = "https://hf-mirror.com"


settings = WhisperSettings()
