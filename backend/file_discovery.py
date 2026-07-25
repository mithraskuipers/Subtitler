"""Recursive discovery of video files inside a chosen directory."""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .models import VideoFile


class InvalidDirectoryError(Exception):
    """Raised when the supplied path is not a usable directory."""


def discover_video_files(directory: str, config: AppConfig) -> list[VideoFile]:
    """Recursively scan ``directory`` for supported video files.

    Results are sorted by relative path so the UI presents a stable,
    predictable ordering regardless of filesystem enumeration order.
    """

    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise InvalidDirectoryError(f"The folder '{directory}' does not exist.")
    if not root.is_dir():
        raise InvalidDirectoryError(f"'{directory}' is not a folder.")

    extensions = {ext.lower() for ext in config.supported_extensions}
    discovered: list[VideoFile] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            discovered.append(VideoFile.from_path(root, path))
        except OSError:
            # File may have been removed/locked between rglob() and stat().
            continue

    discovered.sort(key=lambda video: video.relative_path.lower())
    return discovered
