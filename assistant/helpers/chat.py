"""Generic Groq chat completion helper.

This is used for simple text-to-text tasks like intent classification (Stage 3).
"""

from __future__ import annotations

import httpx


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_groq_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> str:
    """Send a chat completion request to Groq and return the response text.

    This is a lightweight wrapper for classification and other simple text tasks.
    For vision tasks, use ask_groq_vision() instead.

    Args:
        api_key: Groq API key.
        model: Model name (e.g., "meta-llama/llama-4-scout-17b-16e-instruct").
        messages: List of message dicts with "role" and "content" keys.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response text.

    Example:
        >>> response = ask_groq_chat(
        ...     api_key="...",
        ...     model="meta-llama/llama-4-scout-17b-16e-instruct",
        ...     messages=[
        ...         {"role": "system", "content": "Classify as 'vision' or 'search'. Reply with one word."},
        ...         {"role": "user", "content": "What's the weather today?"},
        ...     ],
        ...     temperature=0.0,
        ...     max_tokens=10,
        ... )
        >>> print(response)  # "search"
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
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
        raise ValueError("Groq chat returned no choices.")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("Groq chat returned an empty response.")
    return content
