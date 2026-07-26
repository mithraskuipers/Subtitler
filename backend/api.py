"""HTTP and WebSocket API exposed to the frontend."""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .batch_processor import BatchProcessor
from .config import APP_CONFIG, MODELS_DIR, SUPPORTED_LANGUAGES
from .file_discovery import InvalidDirectoryError, discover_video_files
from .model_manager import (
    download_model_files,
    get_model_definition,
    is_model_downloaded,
    is_model_downloading,
    list_models_status,
)
from .models import BatchStatus, LogLevel
from .state import AppState
from .transcription_engine import cuda_available
from .websocket_manager import ConnectionManager

router = APIRouter()

state = AppState(log_history_limit=APP_CONFIG.log_history_limit)
connection_manager = ConnectionManager()
batch_processor = BatchProcessor(APP_CONFIG, state, connection_manager)


class ScanRequest(BaseModel):
    directory: str


class SelectionRequest(BaseModel):
    videoId: str
    selected: bool


class SelectAllRequest(BaseModel):
    selected: bool


class StartRequest(BaseModel):
    languageCode: str
    devicePreference: str = "auto"


class InterruptDecisionRequest(BaseModel):
    keep: bool


def _open_native_folder_dialog() -> str | None:
    """Show a native OS "select folder" dialog and return the chosen path.

    Runs off the event loop via ``asyncio.to_thread`` since Tkinter dialogs
    are blocking calls.
    """

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title="Select a folder with video files")
    finally:
        root.destroy()
    return selected or None


@router.post("/api/browse-folder")
async def browse_folder() -> dict:
    if batch_processor.is_running():
        raise HTTPException(status_code=409, detail="Cannot browse while a batch is running.")
    try:
        selected = await asyncio.to_thread(_open_native_folder_dialog)
    except Exception as exc:  # pragma: no cover - platform dependent
        raise HTTPException(status_code=500, detail=f"Could not open the folder picker: {exc}") from exc
    if not selected:
        raise HTTPException(status_code=400, detail="No folder was selected.")
    return {"directory": selected}


@router.get("/api/models")
def get_models() -> list[dict]:
    return list_models_status(MODELS_DIR)


@router.post("/api/models/{model_name}/download")
def download_model(model_name: str) -> dict:
    """Kick off a background download for a model, only when explicitly requested."""

    definition = get_model_definition(model_name)
    if definition is None:
        raise HTTPException(status_code=404, detail="Unknown model.")

    if is_model_downloaded(model_name, MODELS_DIR):
        return {"status": "already_downloaded"}
    if is_model_downloading(model_name):
        return {"status": "already_downloading"}

    def _run() -> None:
        def _log(message: str) -> None:
            state.add_log(message, LogLevel.INFO)
            connection_manager.broadcast_threadsafe(state.snapshot())

        try:
            download_model_files(model_name, MODELS_DIR, _log)
            state.add_log(f"Model '{model_name}' is ready to use.", LogLevel.SUCCESS)
        except Exception as exc:  # noqa: BLE001 - surface any download failure
            state.add_log(f"Failed to download model '{model_name}': {exc}", LogLevel.ERROR)
        finally:
            connection_manager.broadcast_threadsafe(state.snapshot())

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@router.get("/api/languages")
def get_languages() -> list[dict[str, str]]:
    return [{"code": lang.code, "label": lang.label} for lang in SUPPORTED_LANGUAGES]


@router.get("/api/device-options")
def get_device_options() -> dict:
    return {
        "options": [
            {"value": "auto", "label": "Auto (recommended)"},
            {"value": "gpu", "label": "GPU (CUDA)"},
            {"value": "cpu", "label": "CPU"},
        ],
        "gpuAvailable": cuda_available(),
    }


@router.get("/api/state")
def get_state() -> dict:
    return state.snapshot()


@router.post("/api/scan")
def scan_directory(payload: ScanRequest) -> dict:
    if batch_processor.is_running():
        raise HTTPException(status_code=409, detail="Cannot scan a new folder while a batch is running.")

    state.set_status(BatchStatus.SCANNING)
    try:
        videos = discover_video_files(payload.directory, APP_CONFIG)
    except InvalidDirectoryError as exc:
        state.set_status(BatchStatus.IDLE)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.set_scan_result(payload.directory, videos)
    state.set_status(BatchStatus.READY if videos else BatchStatus.IDLE)
    state.add_log(f"Found {len(videos)} video file(s) in '{payload.directory}'.")
    return state.snapshot()


@router.post("/api/videos/selection")
def set_video_selection(payload: SelectionRequest) -> dict:
    if batch_processor.is_running():
        raise HTTPException(status_code=409, detail="Cannot change selection while a batch is running.")
    found = state.set_video_selected(payload.videoId, payload.selected)
    if not found:
        raise HTTPException(status_code=404, detail="Video not found.")
    return state.snapshot()


@router.post("/api/videos/selection/all")
def set_all_selection(payload: SelectAllRequest) -> dict:
    if batch_processor.is_running():
        raise HTTPException(status_code=409, detail="Cannot change selection while a batch is running.")
    state.set_all_selected(payload.selected)
    return state.snapshot()


@router.post("/api/start")
def start_batch(payload: StartRequest) -> dict:
    valid_codes = {lang.code for lang in SUPPORTED_LANGUAGES}
    if payload.languageCode not in valid_codes:
        raise HTTPException(status_code=400, detail="Unsupported language.")
    if payload.devicePreference not in ("auto", "gpu", "cpu"):
        raise HTTPException(status_code=400, detail="Unsupported processing device.")
    try:
        batch_processor.start(payload.languageCode, payload.devicePreference)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.snapshot()


@router.post("/api/stop")
def stop_batch() -> dict:
    batch_processor.request_stop()
    return state.snapshot()


@router.post("/api/interrupt-decision")
def submit_interrupt_decision(payload: InterruptDecisionRequest) -> dict:
    if state.get_pending_decision() is None:
        raise HTTPException(status_code=409, detail="There is no pending interruption decision.")
    batch_processor.resolve_interrupt_decision(payload.keep)
    return state.snapshot()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await connection_manager.connect(websocket)
    await websocket.send_json(state.snapshot())
    try:
        while True:
            # The frontend does not need to send anything; this simply keeps
            # the connection open and detects disconnects promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
