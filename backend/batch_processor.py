"""Sequential batch orchestration for transcribing multiple videos.

This module owns the single background worker thread that performs all
transcription work. The FastAPI layer only ever talks to it through the
thread-safe methods exposed here (``start``, ``request_stop``,
``resolve_interrupt_decision``); no transcription work ever runs on the
asyncio event loop.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import AppConfig, MODELS_DIR, TEMP_AUDIO_DIR
from .model_manager import is_model_downloaded
from .models import (
    BatchStatus,
    BatchSummary,
    CurrentVideoProgress,
    LogLevel,
    PendingInterruptDecision,
    ProcessingStage,
    VideoStatus,
)
from .state import AppState
from .subtitle_writer import SubtitleSegment, subtitle_path_for_video, write_srt_file
from .transcription_engine import TranscriptionEngine, TranscriptionError
from .websocket_manager import ConnectionManager


class InterruptDecision:
    KEEP = "keep"
    DELETE = "delete"


class BatchProcessor:
    def __init__(
        self,
        config: AppConfig,
        state: AppState,
        connection_manager: ConnectionManager,
    ) -> None:
        self._config = config
        self._state = state
        self._connection_manager = connection_manager
        self._engine = TranscriptionEngine(config, TEMP_AUDIO_DIR, MODELS_DIR)

        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._decision_event = threading.Event()
        self._decision_result: str | None = None
        self._run_lock = threading.Lock()

    # -- public control surface, safe to call from the event loop -----------
    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def start(self, language_code: str) -> None:
        if self.is_running():
            raise RuntimeError("A batch is already running.")

        selected = self._state.get_selected_videos()
        if not selected:
            raise RuntimeError("No videos are selected for processing.")

        model_name = self._config.transcription.model_name
        if not is_model_downloaded(model_name, MODELS_DIR):
            raise RuntimeError(
                f"The '{model_name}' model is not downloaded yet. Download it from the "
                "Models panel before running a batch."
            )

        self._state.set_language(language_code)
        self._stop_event.clear()
        self._decision_event.clear()
        self._decision_result = None

        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()

    def request_stop(self) -> None:
        if not self.is_running():
            return
        self._stop_event.set()
        self._state.set_status(BatchStatus.STOPPING)
        self._log("Stop requested. Finishing the current segment...", LogLevel.WARNING)
        self._emit()

    def resolve_interrupt_decision(self, keep: bool) -> None:
        self._decision_result = InterruptDecision.KEEP if keep else InterruptDecision.DELETE
        self._decision_event.set()

    # -- internal helpers -----------------------------------------------------
    def _log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        self._state.add_log(message, level)

    def _emit(self) -> None:
        self._connection_manager.broadcast_threadsafe(self._state.snapshot())

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def _set_progress(
        self,
        video_id: str | None,
        filename: str | None,
        stage: ProcessingStage,
        fraction: float,
        started_at: float | None = None,
    ) -> None:
        current = self._state.get_progress()
        self._state.set_progress(
            CurrentVideoProgress(
                video_id=video_id,
                filename=filename,
                stage=stage,
                progress_fraction=fraction,
                started_at=started_at if started_at is not None else current.started_at,
            )
        )
        self._emit()

    # -- main worker loop -----------------------------------------------------
    def _run(self) -> None:
        with self._run_lock:
            self._execute_batch()

    def _execute_batch(self) -> None:
        state = self._state
        language_code = state.language_code
        summary = BatchSummary()
        state.set_summary(None)
        state.set_status(BatchStatus.RUNNING)
        state.start_timer()
        batch_start_time = time.time()
        self._log("Batch started.", LogLevel.INFO)
        self._emit()

        try:
            device_info = self._engine.ensure_loaded(lambda msg: self._log(msg))
            state.set_device_label(f"{device_info.device.upper()} \u2014 {device_info.device_name}")
            self._log(f"Model ready on {device_info.device.upper()} ({device_info.device_name}).", LogLevel.SUCCESS)
            self._emit()
        except Exception as exc:
            self._log(f"Failed to load the transcription model: {exc}", LogLevel.ERROR)
            state.set_status(BatchStatus.IDLE)
            self._emit()
            return

        videos = [v for v in state.get_videos() if v.selected]
        interrupted = False

        for video in videos:
            if self._should_stop():
                video.status = VideoStatus.SKIPPED
                summary.skipped.append(video.filename)
                state.update_video(video.id, status=VideoStatus.SKIPPED)
                continue

            outcome = self._process_single_video(video, language_code)

            if outcome == "done":
                summary.successful.append(video.filename)
            elif outcome == "failed":
                summary.failed.append({"filename": video.filename, "error": video.error_message or "Unknown error"})
            elif outcome == "interrupted":
                summary.skipped.append(video.filename)
                interrupted = True
                # Mark every remaining, not-yet-started video as skipped.
                remaining = videos[videos.index(video) + 1 :]
                for skipped_video in remaining:
                    skipped_video.status = VideoStatus.SKIPPED
                    state.update_video(skipped_video.id, status=VideoStatus.SKIPPED)
                    summary.skipped.append(skipped_video.filename)
                break

        summary.total_processing_seconds = time.time() - batch_start_time
        state.set_summary(summary)
        state.set_progress(CurrentVideoProgress())
        state.set_status(BatchStatus.FINISHED if not interrupted else BatchStatus.IDLE)

        if interrupted:
            self._log("Batch stopped by user.", LogLevel.WARNING)
        else:
            self._log(
                f"Batch finished: {len(summary.successful)} succeeded, "
                f"{len(summary.failed)} failed, {len(summary.skipped)} skipped.",
                LogLevel.SUCCESS,
            )
        self._emit()

        self._stop_event.clear()
        state.set_status(BatchStatus.FINISHED if not interrupted else BatchStatus.IDLE)
        self._emit()

    def _process_single_video(self, video, language_code: str) -> str:
        """Returns one of: 'done', 'failed', 'interrupted'."""

        state = self._state
        video.status = VideoStatus.PROCESSING
        state.update_video(video.id, status=VideoStatus.PROCESSING, error_message=None)
        started_at = time.time()
        self._log(f"Processing '{video.relative_path}'", LogLevel.INFO)
        self._set_progress(video.id, video.filename, ProcessingStage.EXTRACTING_AUDIO, 0.05, started_at)

        audio_path: Path | None = None
        try:
            audio_path = self._engine.extract_audio(video.absolute_path)
            self._set_progress(video.id, video.filename, ProcessingStage.TRANSCRIBING, 0.15, started_at)

            segments: list[SubtitleSegment] = []
            for segment in self._engine.transcribe(audio_path, language_code, self._should_stop):
                segments.append(segment)
                fraction = min(0.15 + segment.end_seconds / 3600.0, 0.9)
                self._set_progress(video.id, video.filename, ProcessingStage.TRANSCRIBING, fraction, started_at)

            destination = subtitle_path_for_video(video.absolute_path, self._config)

            if self._should_stop():
                return self._handle_interruption(video, segments, destination)

            self._set_progress(video.id, video.filename, ProcessingStage.GENERATING_SUBTITLES, 0.92, started_at)
            self._set_progress(video.id, video.filename, ProcessingStage.SAVING, 0.97, started_at)
            write_srt_file(segments, destination)

            video.status = VideoStatus.DONE
            video.subtitle_path = str(destination)
            state.update_video(video.id, status=VideoStatus.DONE, subtitle_path=str(destination))
            self._set_progress(video.id, video.filename, ProcessingStage.FINISHED, 1.0, started_at)
            self._log(f"Saved subtitles to '{destination.name}'", LogLevel.SUCCESS)
            return "done"

        except TranscriptionError as exc:
            video.status = VideoStatus.FAILED
            video.error_message = str(exc)
            state.update_video(video.id, status=VideoStatus.FAILED, error_message=str(exc))
            self._log(f"Failed '{video.relative_path}': {exc}", LogLevel.ERROR)
            return "failed"
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error per-video
            video.status = VideoStatus.FAILED
            video.error_message = str(exc)
            state.update_video(video.id, status=VideoStatus.FAILED, error_message=str(exc))
            self._log(f"Unexpected error on '{video.relative_path}': {exc}", LogLevel.ERROR)
            return "failed"
        finally:
            if audio_path is not None:
                self._engine.cleanup_audio(audio_path)

    def _handle_interruption(self, video, segments: list[SubtitleSegment], destination: Path) -> str:
        state = self._state
        video.status = VideoStatus.INTERRUPTED

        if segments:
            partial_destination = destination.with_name(destination.stem + ".partial" + destination.suffix)
            write_srt_file(segments, partial_destination)
            state.set_pending_decision(
                PendingInterruptDecision(
                    video_id=video.id,
                    filename=video.filename,
                    partial_subtitle_path=str(partial_destination),
                )
            )
            state.set_status(BatchStatus.AWAITING_INTERRUPT_DECISION)
            self._log(
                f"Processing of '{video.filename}' was interrupted. Waiting for your decision "
                "on the partial subtitle file.",
                LogLevel.WARNING,
            )
            self._emit()

            self._decision_event.wait()

            final_path: str | None = None
            if self._decision_result == InterruptDecision.KEEP:
                final_destination = destination
                partial_destination.replace(final_destination)
                final_path = str(final_destination)
                self._log(f"Kept partial subtitles as '{final_destination.name}'.", LogLevel.INFO)
            else:
                partial_destination.unlink(missing_ok=True)
                self._log("Discarded the partial subtitle file.", LogLevel.INFO)

            state.set_pending_decision(None)
            state.update_video(video.id, status=VideoStatus.INTERRUPTED, subtitle_path=final_path)
        else:
            state.update_video(video.id, status=VideoStatus.INTERRUPTED)
            self._log(f"Processing of '{video.filename}' was interrupted before any text was produced.", LogLevel.WARNING)

        return "interrupted"
