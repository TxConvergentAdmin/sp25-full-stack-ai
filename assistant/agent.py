from __future__ import annotations

from dataclasses import dataclass
import re

from assistant.capture import CaptureResult
from assistant.search import ask_groq_web_search
from assistant.speech import DEFAULT_TRANSCRIPTION_MODEL, transcribe_audio
from assistant.vision import ask_groq_vision

# Stage 3+: Import helpers when you need them
# from assistant.helpers import ask_groq_chat  # For LLM-based classification
# from assistant.helpers import retrieve_context  # For RAG
# from assistant.helpers import ask_with_tools, execute_tool  # For MCP


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

        # TODO Stage 4: Add conversation history
        # self.conversation_history: list[dict] = []
        # self.max_history = 10  # Keep last 10 exchanges

    def needs_web_search(self, question: str) -> bool:
        """Determine if a question needs web search (vs vision).

        TODO Stage 3: Replace this keyword-based routing with LLM classification.
        Use ask_groq_chat() to classify questions more accurately:

            def classify_question(self, question: str) -> str:
                response = ask_groq_chat(
                    api_key=self.api_key,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "Classify this question as 'vision' (needs to see the screen) "
                            "or 'search' (needs live web data). Reply with one word only."
                        )},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                return "search" if "search" in response.lower() else "vision"
        """
        lowered = question.lower()

        # Route obviously time-sensitive or live-information questions to web search.
        if any(term in lowered for term in WEB_SEARCH_TERMS):
            return True

        # Questions phrased around recommendations and "best" queries often need current web info.
        if re.search(r"\b(best|recommend|compare|cheapest|near me)\b", lowered):
            return True

        return False

    def answer_question(self, *, question: str, capture: CaptureResult) -> AgentResult:
        """Answer a question using the appropriate tool (vision, search, docs, or action).

        TODO Stage 3: Replace needs_web_search() with classify_question() for smarter routing.
        TODO Stage 5: Add "docs" route for RAG queries about the user's documents.
        TODO Stage 6: Add "action" route for MCP tool calls (reminders, notes, etc.).

        Example routing flow for Stage 5+:
            route = self.classify_question(question)  # Returns "vision", "search", "docs", or "action"

            if route == "search":
                # ... web search path ...
            elif route == "docs":
                # Retrieve relevant chunks from user's documents
                context_chunks = retrieve_context(question, top_k=3)
                context = "\\n---\\n".join(context_chunks)
                augmented_prompt = f"Context from user's docs:\\n{context}\\n\\nQuestion: {question}"
                # ... vision call with augmented prompt ...
            elif route == "action":
                # Use MCP tools
                response = ask_with_tools(api_key=self.api_key, model=self.model, question=question)
                if response.tool_call:
                    result = execute_tool(response.tool_call)
                    return AgentResult(answer=f"Done! {result}", model=self.model)
                return AgentResult(answer=response.text, model=self.model)
            else:
                # ... vision path ...
        """
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
            # TODO Stage 4: Pass conversation history
            # history=self.conversation_history,
        )

        # TODO Stage 4: Update conversation history after each exchange
        # self.conversation_history.append({"role": "user", "content": question})
        # self.conversation_history.append({"role": "assistant", "content": answer})
        # # Trim to max history
        # if len(self.conversation_history) > self.max_history * 2:
        #     self.conversation_history = self.conversation_history[-(self.max_history * 2):]

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
