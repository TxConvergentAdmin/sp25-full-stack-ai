from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import uvicorn
    import webview
    from dotenv import load_dotenv
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - runtime guard
    missing = exc.name or "a required package"
    raise SystemExit(
        "Missing dependency: "
        f"{missing}. Install the required packages with:\n"
        "pip3 install fastapi uvicorn pywebview python-multipart python-dotenv httpx"
    ) from exc

from assistant.agent import ScreenAssistantAgent
from assistant.capture import ScreenCaptureService
from assistant.recorder import AudioRecorder


BASE_DIR = Path(__file__).resolve().parent
OVERLAY_HTML = BASE_DIR / "assistant" / "overlay" / "index.html"
ENV_FILE = BASE_DIR / ".env"
HOST = "127.0.0.1"
PORT = 8123
WINDOW_WIDTH = 760
WINDOW_MIN_HEIGHT = 220
WINDOW_MAX_HEIGHT = 760
WINDOW_TOP_PADDING = 24
WINDOW_SIDE_PADDING = 0

load_dotenv(ENV_FILE)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise SystemExit("Missing GROQ_API_KEY in .env.")


app = FastAPI(title="Screen Assistant UI Prototype")
capture_service = ScreenCaptureService()
assistant_agent = ScreenAssistantAgent(api_key=GROQ_API_KEY)
audio_recorder = AudioRecorder()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class CaptureMetadata(BaseModel):
    captured_at: str
    width: int
    height: int


class AskResponse(BaseModel):
    answer: str
    capture: CaptureMetadata
    transcript: Optional[str] = None


def process_audio_question(
    *,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
) -> AskResponse:
    try:
        capture = capture_service.capture()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Screen capture failed. On macOS, make sure Python has Screen Recording "
                f"permission in System Settings. Original error: {exc}"
            ),
        ) from exc

    try:
        voice_result = assistant_agent.answer_audio_question(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            capture=capture,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Voice request failed: {exc}",
        ) from exc

    return AskResponse(
        answer=voice_result.answer,
        transcript=voice_result.transcript,
        capture=CaptureMetadata(
            captured_at=capture.captured_at,
            width=capture.width,
            height=capture.height,
        ),
    )


@app.get("/")
async def index() -> FileResponse:
    if not OVERLAY_HTML.exists():
        raise HTTPException(status_code=500, detail="Overlay UI file is missing.")
    return FileResponse(OVERLAY_HTML)


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        capture = capture_service.capture()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Screen capture failed. On macOS, make sure Python has Screen Recording "
                f"permission in System Settings. Original error: {exc}"
            ),
        ) from exc

    try:
        agent_result = assistant_agent.answer_question(question=question, capture=capture)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vision request failed: {exc}",
        ) from exc

    return AskResponse(
        answer=agent_result.answer,
        capture=CaptureMetadata(
            captured_at=capture.captured_at,
            width=capture.width,
            height=capture.height,
        ),
    )


@app.post("/voice", response_model=AskResponse)
async def voice_ask(file: UploadFile = File(...)) -> AskResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    content_type = file.content_type or "audio/webm"
    filename = file.filename or "voice.webm"
    return process_audio_question(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )


class OverlayBridge:
    def __init__(self) -> None:
        self.window: webview.Window | None = None
        self._last_height = WINDOW_MIN_HEIGHT

    def attach(self, window: webview.Window) -> None:
        self.window = window

    def resize_overlay(self, content_height: int) -> None:
        if self.window is None:
            return

        target_height = int(content_height) + WINDOW_TOP_PADDING + WINDOW_SIDE_PADDING
        target_height = max(WINDOW_MIN_HEIGHT, min(WINDOW_MAX_HEIGHT, target_height))

        if abs(target_height - self._last_height) < 4:
            return

        self._last_height = target_height
        self.window.resize(WINDOW_WIDTH, target_height)

    def close_window(self) -> None:
        def shutdown() -> None:
            if self.window is not None:
                try:
                    self.window.hide()
                except Exception:
                    pass
            os._exit(0)

        threading.Thread(target=shutdown, daemon=True).start()

    def start_voice_recording(self) -> dict[str, str]:
        try:
            audio_recorder.start()
        except Exception as exc:
            raise RuntimeError(f"Unable to start recording: {exc}") from exc
        return {"status": "recording"}

    def stop_voice_recording(self) -> dict:
        try:
            audio_bytes = audio_recorder.stop()
            response = process_audio_question(
                audio_bytes=audio_bytes,
                filename="voice.wav",
                content_type="audio/wav",
            )
        except HTTPException as exc:
            raise RuntimeError(str(exc.detail)) from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to stop recording: {exc}") from exc

        return response.model_dump()


def wait_for_server(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"FastAPI server did not start on {host}:{port}.")


def run_server() -> None:
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


def main() -> None:
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    wait_for_server(HOST, PORT)

    bridge = OverlayBridge()
    window = webview.create_window(
        "Screen Assistant",
        url=f"http://{HOST}:{PORT}",
        js_api=bridge,
        width=WINDOW_WIDTH,
        height=WINDOW_MIN_HEIGHT,
        min_size=(WINDOW_WIDTH, WINDOW_MIN_HEIGHT),
        resizable=True,
        frameless=True,
        easy_drag=True,
        on_top=True,
        background_color="#212121",
        transparent=False,
        vibrancy=False,
    )
    bridge.attach(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
