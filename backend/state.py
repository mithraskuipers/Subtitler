"""Thread-safe, in-memory application state.

The batch processor runs on a background thread while the API layer runs on
the asyncio event loop, so all mutations to shared state go through a single
lock. Snapshots handed to the API/WebSocket layer are plain dictionaries,
never live references, so callers can never mutate state accidentally.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from .config import DEFAULT_LANGUAGE_CODE
from .models import (
    BatchStatus,
    BatchSummary,
    CurrentVideoProgress,
    LogEntry,
    LogLevel,
    PendingInterruptDecision,
    VideoFile,
)


class AppState:
    def __init__(self, log_history_limit: int = 500) -> None:
        self._lock = threading.RLock()
        self._status: BatchStatus = BatchStatus.IDLE
        self._source_directory: str | None = None
        self._language_code: str = DEFAULT_LANGUAGE_CODE
        self._videos: list[VideoFile] = []
        self._current_progress = CurrentVideoProgress()
        self._logs: deque[LogEntry] = deque(maxlen=log_history_limit)
        self._summary: BatchSummary | None = None
        self._pending_decision: PendingInterruptDecision | None = None
        self._batch_started_at: float | None = None
        self._device_label: str | None = None

    # -- status -----------------------------------------------------------
    @property
    def status(self) -> BatchStatus:
        with self._lock:
            return self._status

    def set_status(self, status: BatchStatus) -> None:
        with self._lock:
            self._status = status

    # -- directory / language ----------------------------------------------
    def set_scan_result(self, directory: str, videos: list[VideoFile]) -> None:
        with self._lock:
            self._source_directory = directory
            self._videos = videos
            self._summary = None
            self._current_progress = CurrentVideoProgress()

    def set_language(self, language_code: str) -> None:
        with self._lock:
            self._language_code = language_code

    @property
    def language_code(self) -> str:
        with self._lock:
            return self._language_code

    @property
    def source_directory(self) -> str | None:
        with self._lock:
            return self._source_directory

    # -- videos -------------------------------------------------------------
    def get_videos(self) -> list[VideoFile]:
        with self._lock:
            return list(self._videos)

    def get_selected_videos(self) -> list[VideoFile]:
        with self._lock:
            return [v for v in self._videos if v.selected]

    def set_video_selected(self, video_id: str, selected: bool) -> bool:
        with self._lock:
            for video in self._videos:
                if video.id == video_id:
                    video.selected = selected
                    return True
            return False

    def set_all_selected(self, selected: bool) -> None:
        with self._lock:
            for video in self._videos:
                video.selected = selected

    def update_video(self, video_id: str, **fields: Any) -> None:
        with self._lock:
            for video in self._videos:
                if video.id == video_id:
                    for key, value in fields.items():
                        setattr(video, key, value)
                    return

    # -- progress -------------------------------------------------------------
    def set_progress(self, progress: CurrentVideoProgress) -> None:
        with self._lock:
            self._current_progress = progress

    def get_progress(self) -> CurrentVideoProgress:
        with self._lock:
            return self._current_progress

    # -- logs -------------------------------------------------------------
    def add_log(self, message: str, level: LogLevel = LogLevel.INFO) -> LogEntry:
        entry = LogEntry(message=message, level=level)
        with self._lock:
            self._logs.append(entry)
        return entry

    def get_logs(self) -> list[LogEntry]:
        with self._lock:
            return list(self._logs)

    # -- summary / timing ---------------------------------------------------
    def set_summary(self, summary: BatchSummary | None) -> None:
        with self._lock:
            self._summary = summary

    def get_summary(self) -> BatchSummary | None:
        with self._lock:
            return self._summary

    def start_timer(self) -> None:
        with self._lock:
            self._batch_started_at = time.time()

    def elapsed_seconds(self) -> float:
        with self._lock:
            if self._batch_started_at is None:
                return 0.0
            return time.time() - self._batch_started_at

    def set_device_label(self, label: str) -> None:
        with self._lock:
            self._device_label = label

    # -- interrupt decision ---------------------------------------------------
    def set_pending_decision(self, decision: PendingInterruptDecision | None) -> None:
        with self._lock:
            self._pending_decision = decision

    def get_pending_decision(self) -> PendingInterruptDecision | None:
        with self._lock:
            return self._pending_decision

    # -- snapshot -------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_decision
            return {
                "status": self._status.value,
                "sourceDirectory": self._source_directory,
                "languageCode": self._language_code,
                "videos": [v.to_dict() for v in self._videos],
                "currentProgress": self._current_progress.to_dict(),
                "logs": [entry.to_dict() for entry in self._logs],
                "summary": self._summary.to_dict() if self._summary else None,
                "elapsedSeconds": self.elapsed_seconds() if self._batch_started_at else 0.0,
                "deviceLabel": self._device_label,
                "pendingInterruptDecision": (
                    {
                        "videoId": pending.video_id,
                        "filename": pending.filename,
                        "partialSubtitlePath": pending.partial_subtitle_path,
                    }
                    if pending
                    else None
                ),
            }
