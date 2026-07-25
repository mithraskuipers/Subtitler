"""Typed data structures shared across the backend layers."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BatchStatus(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    AWAITING_INTERRUPT_DECISION = "awaiting_interrupt_decision"
    FINISHED = "finished"


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class ProcessingStage(str, Enum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    GENERATING_SUBTITLES = "generating_subtitles"
    SAVING = "saving"
    FINISHED = "finished"
    ERROR = "error"


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class VideoFile:
    """A single discovered video file and its processing state."""

    id: str
    absolute_path: str
    relative_path: str
    filename: str
    size_bytes: int
    selected: bool = True
    status: VideoStatus = VideoStatus.PENDING
    error_message: str | None = None
    subtitle_path: str | None = None
    duration_seconds: float | None = None

    @staticmethod
    def from_path(root: Path, path: Path) -> "VideoFile":
        stat = path.stat()
        return VideoFile(
            id=str(uuid.uuid4()),
            absolute_path=str(path),
            relative_path=str(path.relative_to(root)),
            filename=path.name,
            size_bytes=stat.st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "absolutePath": self.absolute_path,
            "relativePath": self.relative_path,
            "filename": self.filename,
            "sizeBytes": self.size_bytes,
            "selected": self.selected,
            "status": self.status.value,
            "errorMessage": self.error_message,
            "subtitlePath": self.subtitle_path,
            "durationSeconds": self.duration_seconds,
        }


@dataclass
class LogEntry:
    message: str
    level: LogLevel = LogLevel.INFO
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "level": self.level.value,
            "timestamp": self.timestamp,
        }


@dataclass
class CurrentVideoProgress:
    """Fine-grained progress for the video currently being processed."""

    video_id: str | None = None
    filename: str | None = None
    stage: ProcessingStage = ProcessingStage.QUEUED
    progress_fraction: float = 0.0
    started_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "videoId": self.video_id,
            "filename": self.filename,
            "stage": self.stage.value,
            "progressFraction": self.progress_fraction,
            "startedAt": self.started_at,
        }


@dataclass
class BatchSummary:
    successful: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_processing_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "totalProcessingSeconds": self.total_processing_seconds,
        }


@dataclass
class PendingInterruptDecision:
    video_id: str
    filename: str
    partial_subtitle_path: str
