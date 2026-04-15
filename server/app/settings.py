from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "sensevoice-vad-server"
    app_host: str = "0.0.0.0"
    app_port: int = 19000

    sensevoice_model: str = "iic/SenseVoiceSmall"
    sensevoice_device: str = "cuda:0"
    sensevoice_fallback_to_cpu: bool = True
    sensevoice_language: str = "zh"
    sensevoice_hotwords: str = ""
    sensevoice_text_rewrite: str = "寨上=>站上;三脚痛=>三角筒;新判断=>请判断"
    sensevoice_rewrite_file: str = "/app/config/rewrite_rules.txt"
    sensevoice_use_itn: bool = True
    sensevoice_batch_size_s: int = 60

    faster_whisper_model: str = "large-v3"
    faster_whisper_device: str = "cuda"
    faster_whisper_compute_type: str = "float16"
    faster_whisper_language: str = "zh"
    faster_whisper_beam_size: int = 5
    faster_whisper_service_url: str = ""
    faster_whisper_timeout_sec: float = 180.0

    # VAD controls
    vad_aggressiveness: int = 1
    vad_frame_ms: int = 20
    vad_start_trigger_frames: int = 6
    vad_end_trigger_frames: int = 18
    vad_max_segment_ms: int = 15000
    vad_min_segment_ms: int = 1200
    vad_auto_adapt_enabled: bool = True
    vad_auto_speech_rms_threshold: float = 0.012

    # Realtime partial controls
    partial_interval_sec: float = 1.6
    partial_min_audio_ms: int = 1800


settings = AppSettings()
