from __future__ import annotations

from dataclasses import dataclass
import re

from assistant.capture import CaptureResult
from assistant.search import ask_groq_web_search
from assistant.speech import DEFAULT_TRANSCRIPTION_MODEL, transcribe_audio
from assistant.vision import ask_groq_vision


DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WEB_SEARCH_TERMS = (
    "today",
    "latest",
    "current",
    "right now",
    "news",
    "weather",
    "forecast",
    "deal",
    "deals",
    "discount",
    "price",
    "prices",
    "sale",
    "stock",
    "score",
    "scores",
    "trending",
    "release date",
)
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

    def needs_web_search(self, question: str) -> bool:
        lowered = question.lower()

        # Route obviously time-sensitive or live-information questions to web search.
        if any(term in lowered for term in WEB_SEARCH_TERMS):
            return True

        # Questions phrased around recommendations and "best" queries often need current web info.
        if re.search(r"\b(best|recommend|compare|cheapest|near me)\b", lowered):
            return True

        return False

    def answer_question(self, *, question: str, capture: CaptureResult) -> AgentResult:
        # Minimal routing: use live web search for current-info questions, otherwise use vision.
        if self.needs_web_search(question):
            answer = ask_groq_web_search(api_key=self.api_key, question=question)
            return AgentResult(answer=answer, model="groq/compound-mini")

        user_prompt = (
            f"User question: {question}\n"
            f"Screenshot timestamp: {capture.captured_at}\n"
            "Answer naturally and prioritize being helpful."
        )
        answer = ask_groq_vision(
            api_key=self.api_key,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            screenshot_png_bytes=capture.png_bytes,
        )
        return AgentResult(answer=answer, model=self.model)

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
