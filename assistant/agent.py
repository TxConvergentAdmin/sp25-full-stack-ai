from __future__ import annotations

from dataclasses import dataclass

from assistant.capture import CaptureResult
from assistant.speech import DEFAULT_TRANSCRIPTION_MODEL, transcribe_audio
from assistant.vision import ask_groq_vision

# Stage 2: add these when you introduce web search routing
# import re
# from assistant.search import ask_groq_web_search

# Stage 3+: add helpers when you introduce classification, RAG, and tools
# from assistant.helpers import ask_groq_chat
# from assistant.helpers import retrieve_context
# from assistant.helpers import ask_with_tools, execute_tool


DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Stage 2: add this constant when you build keyword routing
# WEB_SEARCH_TERMS = (
#     "today",
#     "latest",
#     "current",
#     "right now",
#     "news",
#     "weather",
#     "forecast",
#     "deal",
#     "deals",
#     "discount",
#     "price",
#     "prices",
#     "sale",
#     "stock",
#     "score",
#     "scores",
#     "trending",
#     "release date",
# )

SYSTEM_PROMPT = (
    "You are a personal assistant. Give direct, concise answers. "
    "Use the screenshot when it is helpful, but do not say things like "
    "'this is not visible in the screenshot' or talk about screenshot limitations "
    "unless the user is explicitly asking about something on screen and the answer truly depends on it. "
    "If the question is general, answer it normally without mentioning the screenshot. "
    "Do not claim to have done a web search or accessed live data unless that actually happened."
)


@dataclass
class AgentResult:
    answer: str
    model: str


@dataclass
class VoiceResult:
    transcript: str
    transcription_model: str
    answer: str
    model: str


class ScreenAssistantAgent:
    def __init__(self, *, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> None:
        self.api_key = api_key
        self.model = model

        # Stage 4: uncomment these when you add conversation memory
        # self.conversation_history: list[dict] = []
        # self.max_history = 10

    def answer_question(self, *, question: str, capture: CaptureResult) -> AgentResult:
        """Workshop starter version.
        """
        return AgentResult(
            answer=(
                "Stage 1 TODO: update assistant/agent.py so this method calls "
                "ask_groq_vision() with the user's question and the screenshot."
            ),
            model=self.model,
        )

    def answer_audio_question(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        capture: CaptureResult,
    ) -> VoiceResult:
        transcript = transcribe_audio(
            api_key=self.api_key,
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
        )
        result = self.answer_question(question=transcript, capture=capture)
        return VoiceResult(
            transcript=transcript,
            transcription_model=DEFAULT_TRANSCRIPTION_MODEL,
            answer=result.answer,
            model=result.model,
        )
