"""Central configuration for Subtitler.

Keeping every tunable value in one module makes it trivial to change the
Whisper model, add new languages, or adjust file-handling behaviour without
hunting through the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LanguageOption:
    """A language the UI can offer to the user."""

    code: str
    label: str


# Only Dutch is enabled today, but the UI and API already treat this as a
# list so additional languages can be appended here later without touching
# any other layer of the application.
SUPPORTED_LANGUAGES: tuple[LanguageOption, ...] = (
    LanguageOption(code="nl", label="Dutch"),
)

DEFAULT_LANGUAGE_CODE = "nl"


@dataclass(frozen=True)
class TranscriptionConfig:
    """Everything related to the speech-recognition engine.

    ``model_name`` is intentionally isolated so switching to a future
    Whisper release only requires changing this one value.
    """

    model_name: str = "large-v3-turbo"
    compute_type_gpu: str = "float16"
    compute_type_cpu: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    condition_on_previous_text: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Application-wide settings."""

    supported_extensions: tuple[str, ...] = (
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".m4v",
        ".wmv",
        ".webm",
        ".mpg",
        ".mpeg",
    )
    subtitle_suffix: str = "_mk"
    subtitle_extension: str = ".srt"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    host: str = "127.0.0.1"
    port: int = 8756
    log_history_limit: int = 500
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)


APP_CONFIG = AppConfig()

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMP_AUDIO_DIR = PROJECT_ROOT / ".runtime" / "audio_cache"
MODELS_DIR = PROJECT_ROOT / "models"
