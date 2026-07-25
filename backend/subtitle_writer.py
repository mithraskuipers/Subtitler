"""Conversion of transcription segments into SRT subtitle files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig


@dataclass
class SubtitleSegment:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


def _format_timestamp(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0.0
    milliseconds = round(total_seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def segments_to_srt(segments: list[SubtitleSegment]) -> str:
    """Render a list of segments into standard SRT text."""

    blocks: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        block = (
            f"{segment.index}\n"
            f"{_format_timestamp(segment.start_seconds)} --> {_format_timestamp(segment.end_seconds)}\n"
            f"{text}\n"
        )
        blocks.append(block)
    return "\n".join(blocks) + ("\n" if blocks else "")


def subtitle_path_for_video(video_path: str, config: AppConfig) -> Path:
    """Compute the destination ``.srt`` path for a given video.

    ``movie.mp4`` becomes ``movie_mk.srt`` in the same directory as the
    source video, matching the naming convention expected by Kodi, VLC and
    Jellyfin.
    """

    path = Path(video_path)
    return path.with_name(f"{path.stem}{config.subtitle_suffix}{config.subtitle_extension}")


def write_srt_file(segments: list[SubtitleSegment], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(segments_to_srt(segments), encoding="utf-8")
