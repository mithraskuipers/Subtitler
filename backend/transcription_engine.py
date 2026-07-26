"""Speech-to-text engine built on top of faster-whisper.

The engine loads the Whisper Large-v3 Turbo model once and reuses it for
every video in a batch. GPU acceleration is used automatically when a CUDA
device is available, otherwise the engine transparently falls back to CPU
inference. Segments are yielded incrementally so callers can react to a
stop request without waiting for an entire file to finish transcribing.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, TranscriptionConfig
from .subtitle_writer import SubtitleSegment


class TranscriptionError(Exception):
    """Raised when audio extraction or transcription fails."""


@dataclass
class DeviceInfo:
    device: str
    compute_type: str
    device_name: str


def _gpu_device_name() -> str:
    """Best-effort human-readable GPU name, without requiring torch."""

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        return name if isinstance(name, str) else name.decode()
    except Exception:
        return "CUDA GPU"


def cuda_available() -> bool:
    """Whether a CUDA-capable GPU is usable via ctranslate2.

    Detection goes through ctranslate2 (the runtime faster-whisper is built
    on) so no additional heavyweight dependency such as torch is required
    just to check for a GPU.
    """

    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        # ctranslate2 missing or CUDA not usable.
        return False


def detect_device(config: TranscriptionConfig, preference: str = "auto") -> DeviceInfo:
    """Resolve which device to run on, honouring an explicit user preference.

    ``preference`` is one of ``"auto"`` (GPU if available, otherwise CPU),
    ``"gpu"`` (force CUDA, raising if unavailable) or ``"cpu"`` (force CPU
    even if a GPU is present).
    """

    if preference == "cpu":
        return DeviceInfo(device="cpu", compute_type=config.compute_type_cpu, device_name="CPU")

    if preference == "gpu":
        if cuda_available():
            return DeviceInfo(device="cuda", compute_type=config.compute_type_gpu, device_name=_gpu_device_name())
        raise TranscriptionError(
            "GPU processing was requested but no CUDA-capable GPU was detected on this "
            "machine. Choose CPU or Auto instead."
        )

    if cuda_available():
        return DeviceInfo(device="cuda", compute_type=config.compute_type_gpu, device_name=_gpu_device_name())
    return DeviceInfo(device="cpu", compute_type=config.compute_type_cpu, device_name="CPU")


def _resolve_ffmpeg_binary() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TranscriptionError(
            "No ffmpeg binary could be found or downloaded. Install ffmpeg or "
            "ensure the 'imageio-ffmpeg' package installed correctly."
        ) from exc


class TranscriptionEngine:
    """Loads a Whisper model once and transcribes videos one at a time."""

    def __init__(self, config: AppConfig, temp_audio_dir: Path, models_dir: Path) -> None:
        self._config = config
        self._temp_audio_dir = temp_audio_dir
        self._models_dir = models_dir
        self._model = None
        self._device_info: DeviceInfo | None = None
        self._device_preference: str | None = None
        self._ffmpeg_binary: str | None = None

    @property
    def device_info(self) -> DeviceInfo | None:
        return self._device_info

    def ensure_loaded(self, log: Callable[[str], None], device_preference: str = "auto") -> DeviceInfo:
        """Load the model if needed, returning device info.

        If the model was already loaded under a different ``device_preference``
        than the one requested now, it is reloaded so the new preference (e.g.
        switching from GPU to CPU) actually takes effect.
        """

        if (
            self._model is not None
            and self._device_info is not None
            and self._device_preference == device_preference
        ):
            return self._device_info

        from faster_whisper import WhisperModel

        from .model_manager import is_model_downloaded

        self._ffmpeg_binary = _resolve_ffmpeg_binary()
        self._device_info = detect_device(self._config.transcription, device_preference)
        self._device_preference = device_preference

        model_name = self._config.transcription.model_name
        self._models_dir.mkdir(parents=True, exist_ok=True)

        if not is_model_downloaded(model_name, self._models_dir):
            raise TranscriptionError(
                f"Model '{model_name}' is not downloaded yet. Download it from the "
                "Models panel before running a batch."
            )

        log(
            f"Loading model '{model_name}' on "
            f"{self._device_info.device.upper()} ({self._device_info.device_name})"
        )
        # local_files_only guarantees this never silently reaches out to the
        # network - loading only ever uses what was explicitly downloaded.
        self._model = WhisperModel(
            model_name,
            device=self._device_info.device,
            compute_type=self._device_info.compute_type,
            download_root=str(self._models_dir),
            local_files_only=True,
        )

        return self._device_info

    def extract_audio(self, video_path: str) -> Path:
        """Extract mono 16kHz PCM audio from ``video_path`` into a temp WAV file."""

        if self._ffmpeg_binary is None:
            self._ffmpeg_binary = _resolve_ffmpeg_binary()

        self._temp_audio_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._temp_audio_dir / f"{uuid.uuid4().hex}.wav"

        command = [
            self._ffmpeg_binary,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self._config.audio_sample_rate),
            "-ac",
            str(self._config.audio_channels),
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        if result.returncode != 0 or not output_path.exists():
            stderr_text = result.stderr.decode(errors="ignore")[-2000:]
            raise TranscriptionError(f"Audio extraction failed: {stderr_text.strip() or 'unknown ffmpeg error'}")

        return output_path

    def transcribe(
        self,
        audio_path: Path,
        language_code: str,
        should_stop: Callable[[], bool],
    ) -> Iterator[SubtitleSegment]:
        """Yield :class:`SubtitleSegment` objects as they become available.

        The generator checks ``should_stop`` between segments so callers can
        interrupt a long transcription without losing already-produced
        segments.
        """

        if self._model is None:
            raise TranscriptionError("The transcription model has not been loaded yet.")

        cfg = self._config.transcription
        segments_iter, _info = self._model.transcribe(
            str(audio_path),
            language=language_code,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": cfg.vad_min_silence_ms},
            condition_on_previous_text=cfg.condition_on_previous_text,
        )

        for index, segment in enumerate(segments_iter, start=1):
            yield SubtitleSegment(
                index=index,
                start_seconds=segment.start,
                end_seconds=segment.end,
                text=segment.text,
            )
            if should_stop():
                return

    @staticmethod
    def cleanup_audio(audio_path: Path) -> None:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass
