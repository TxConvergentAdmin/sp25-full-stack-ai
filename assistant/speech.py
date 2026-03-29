from __future__ import annotations

import httpx


GROQ_AUDIO_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"


def transcribe_audio(
    *,
    api_key: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str = DEFAULT_TRANSCRIPTION_MODEL,
) -> str:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            GROQ_AUDIO_TRANSCRIPTIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": model,
                "response_format": "json",
            },
            files={
                "file": (filename, audio_bytes, content_type),
            },
        )

    response.raise_for_status()
    payload = response.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise ValueError("Groq returned an empty transcription.")
    return text
