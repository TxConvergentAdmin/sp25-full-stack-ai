from __future__ import annotations

import base64
from io import BytesIO

import httpx
from PIL import Image


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_BASE64_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600


def _prepare_image_data_url(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

    quality = 85
    while True:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue())
        if len(encoded) <= MAX_BASE64_IMAGE_BYTES or quality <= 45:
            return f"data:image/jpeg;base64,{encoded.decode('utf-8')}"
        quality -= 10


def ask_groq_vision(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    screenshot_png_bytes: bytes,
) -> str:
    image_data_url = _prepare_image_data_url(screenshot_png_bytes)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_completion_tokens": 500,
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Groq returned no choices.")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("Groq returned an empty response.")
    return content
