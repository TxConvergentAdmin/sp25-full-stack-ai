from __future__ import annotations

from io import BytesIO
from threading import Lock
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


class AudioRecorder:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._recording = False

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            raise RuntimeError(str(status))
        self._chunks.append(indata.copy())

    def start(self) -> None:
        with self._lock:
            if self._recording:
                raise RuntimeError("Recording is already in progress.")

            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True

    def stop(self) -> bytes:
        with self._lock:
            if not self._recording or self._stream is None:
                raise RuntimeError("Recording has not started.")

            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._recording = False

            if not self._chunks:
                raise RuntimeError("No audio was captured.")

            audio = np.concatenate(self._chunks, axis=0)
            buffer = BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(np.dtype(DTYPE).itemsize)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio.tobytes())

            return buffer.getvalue()

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording
