from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from threading import Lock

import mss
from PIL import Image


@dataclass
class CaptureResult:
    captured_at: str
    width: int
    height: int
    png_bytes: bytes


class ScreenCaptureService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: CaptureResult | None = None

    def capture(self) -> CaptureResult:
        with self._lock:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                shot = sct.grab(monitor)

            image = Image.frombytes("RGB", shot.size, shot.rgb)
            buffer = BytesIO()
            image.save(buffer, format="PNG")

            result = CaptureResult(
                captured_at=datetime.now(timezone.utc).isoformat(),
                width=shot.width,
                height=shot.height,
                png_bytes=buffer.getvalue(),
            )
            self._latest = result
            return result

    def latest(self) -> CaptureResult | None:
        with self._lock:
            return self._latest
