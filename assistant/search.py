from __future__ import annotations

import httpx


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_WEB_MODEL = "groq/compound-mini"
WEB_SYSTEM_PROMPT = (
    "You are a concise assistant. Use web search when needed and answer with current, "
    "direct information. Keep the answer brief and practical."
)


def ask_groq_web_search(*, api_key: str, question: str, model: str = DEFAULT_WEB_MODEL) -> str:
    # Keep the payload simple: Compound Mini can automatically use web search.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "compound_custom": {
            "tools": {
                "enabled_tools": ["web_search"],
            }
        },
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
        raise ValueError("Groq web search returned no choices.")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("Groq web search returned an empty response.")
    return content
