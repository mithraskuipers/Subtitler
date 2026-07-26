"""Registry of speech-recognition models and their local download status.

Models are cached inside the project's own ``models/`` directory (passed to
faster-whisper as ``download_root``) instead of the user's global Hugging
Face cache. This keeps everything self-contained next to the venv and makes
it possible to show, per model, whether it has already been downloaded -
important once more languages/models are added later.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm as _TqdmBase


@dataclass(frozen=True)
class ModelDefinition:
    """A speech-recognition model that can be selected for transcription."""

    name: str
    label: str
    languages: tuple[str, ...]
    approx_size_mb: int


# Adding support for another language or model later is a one-line addition
# here; nothing else in the backend needs to change to make it selectable.
MODEL_REGISTRY: tuple[ModelDefinition, ...] = (
    ModelDefinition(
        name="large-v3-turbo",
        label="Whisper Large v3 Turbo",
        languages=("nl",),
        approx_size_mb=1600,
    ),
)


# faster-whisper resolves short model names to specific HF repos, and not
# all of them live under the Systran org (e.g. large-v3-turbo is published
# by mobiuslabsgmbh). Add an entry here whenever a new model is added to
# MODEL_REGISTRY above and it isn't hosted under Systran/faster-whisper-<name>.
_MODEL_REPO_IDS: dict[str, str] = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _repo_dir_name(model_name: str) -> str:
    """Directory name Hugging Face Hub uses to cache a faster-whisper repo."""

    repo_id = _MODEL_REPO_IDS.get(model_name, f"Systran/faster-whisper-{model_name}")
    return f"models--{repo_id.replace('/', '--')}"


def is_model_downloaded(model_name: str, models_dir: Path) -> bool:
    """Whether a fully-downloaded model already exists in ``models_dir``."""

    repo_dir = models_dir / _repo_dir_name(model_name)
    if not repo_dir.exists():
        return False
    return any(repo_dir.rglob("model.bin"))


def get_model_definition(model_name: str) -> ModelDefinition | None:
    for definition in MODEL_REGISTRY:
        if definition.name == model_name:
            return definition
    return None


# Models are only ever downloaded in response to an explicit user action from
# the frontend (never implicitly, e.g. when a batch starts). This set tracks
# which models currently have a download in flight so concurrent requests
# and the status endpoint can reflect that.
_download_lock = threading.Lock()
_downloading_models: set[str] = set()

# Byte-level progress for in-flight downloads, keyed by model name. Read by
# the status endpoint so the frontend can render an actual progress bar
# instead of just an indeterminate "Downloading..." badge.
_progress_lock = threading.Lock()
_download_progress: dict[str, dict] = {}


def is_model_downloading(model_name: str) -> bool:
    with _download_lock:
        return model_name in _downloading_models


def get_download_progress(model_name: str) -> dict | None:
    """Current download progress for ``model_name``, or ``None`` if idle."""

    with _progress_lock:
        progress = _download_progress.get(model_name)
        return dict(progress) if progress else None


def _set_download_progress(model_name: str, downloaded_bytes: int, total_bytes: int) -> None:
    percent = None
    if total_bytes > 0:
        # Cap below 100 while still downloading: total_bytes is only an
        # estimate (registry's approx_size_mb), so actual bytes can slightly
        # overshoot it. The badge flips to "Downloaded" once the transfer
        # actually completes, so this never gets stuck at 99%.
        percent = min(99.0, round(downloaded_bytes / total_bytes * 100, 1))
    with _progress_lock:
        _download_progress[model_name] = {
            "downloadedBytes": downloaded_bytes,
            "totalBytes": total_bytes,
            "percent": percent,
        }


def _clear_download_progress(model_name: str) -> None:
    with _progress_lock:
        _download_progress.pop(model_name, None)


def download_model_files(model_name: str, models_dir: Path, log: Callable[[str], None]) -> None:
    """Download a model's files into ``models_dir`` if not already present.

    This only fetches the model weights (via huggingface_hub); it never
    loads the model into memory, so it is cheap to call purely to warm the
    local cache ahead of time. Progress is tracked in ``_download_progress``
    as bytes stream in, keyed against the registry's approximate model size,
    so the status endpoint can expose real download progress.
    """

    if is_model_downloaded(model_name, models_dir):
        log(f"Model '{model_name}' is already downloaded.")
        return

    with _download_lock:
        if model_name in _downloading_models:
            log(f"Model '{model_name}' is already being downloaded.")
            return
        _downloading_models.add(model_name)

    definition = get_model_definition(model_name)
    approx_total_bytes = (definition.approx_size_mb if definition else 0) * 1024 * 1024
    _set_download_progress(model_name, 0, approx_total_bytes)
    downloaded_state = {"bytes": 0}
    state_lock = threading.Lock()

    class _ProgressTqdm(_TqdmBase):
        """tqdm subclass huggingface_hub instantiates once per downloaded
        file; every ``update()`` call reports bytes moved for that file, so
        accumulating ``n`` across instances gives total bytes downloaded."""

        def update(self, n: int = 1) -> None:
            result = super().update(n)
            if n:
                with state_lock:
                    downloaded_state["bytes"] += n
                    total = downloaded_state["bytes"]
                _set_download_progress(model_name, total, approx_total_bytes)
            return result

    try:
        from huggingface_hub import snapshot_download

        repo_id = _MODEL_REPO_IDS.get(model_name, f"Systran/faster-whisper-{model_name}")
        allow_patterns = [
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ]

        models_dir.mkdir(parents=True, exist_ok=True)
        log(f"Downloading model '{model_name}'... this may take a few minutes.")
        snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(models_dir),
            tqdm_class=_ProgressTqdm,
        )
        log(f"Model '{model_name}' downloaded and ready.")
    finally:
        _clear_download_progress(model_name)
        with _download_lock:
            _downloading_models.discard(model_name)


def list_models_status(models_dir: Path) -> list[dict]:
    """Serializable status of every registered model, for the UI."""

    return [
        {
            "name": definition.name,
            "label": definition.label,
            "languages": list(definition.languages),
            "approxSizeMb": definition.approx_size_mb,
            "installed": is_model_downloaded(definition.name, models_dir),
            "downloading": is_model_downloading(definition.name),
            "downloadProgress": get_download_progress(definition.name),
        }
        for definition in MODEL_REGISTRY
    ]
