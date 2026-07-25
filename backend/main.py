"""Application entrypoint: wires together the API router and static frontend."""

from __future__ import annotations

import asyncio
import webbrowser
from contextlib import asynccontextmanager
from threading import Timer

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import api
from .config import APP_CONFIG, FRONTEND_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    api.connection_manager.bind_loop(asyncio.get_event_loop())
    yield


app = FastAPI(title="Subtitler", lifespan=lifespan)
app.include_router(api.router)

app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def _open_browser() -> None:
    webbrowser.open(f"http://{APP_CONFIG.host}:{APP_CONFIG.port}/")


if __name__ == "__main__":
    import uvicorn

    Timer(1.2, _open_browser).start()
    uvicorn.run(app, host=APP_CONFIG.host, port=APP_CONFIG.port, log_level="warning")
