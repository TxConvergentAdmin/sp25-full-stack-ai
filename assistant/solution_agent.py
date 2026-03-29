"""SOLUTION KEY: Complete agent implementation with all 6 stages.

This file shows what agent.py should look like after completing all workshop stages.
DO NOT show this to students until they've attempted each stage!

Stages implemented:
- Stage 1: Vision (basic multimodal)
- Stage 2: Web search + keyword routing
- Stage 3: LLM-based classification
- Stage 4: Conversation memory
- Stage 5: RAG over documents
- Stage 6: MCP tool integration
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.capture import CaptureResult
from assistant.search import ask_groq_web_search
from assistant.speech import DEFAULT_TRANSCRIPTION_MODEL, transcribe_audio
from assistant.vision import ask_groq_vision
from assistant.helpers import (
    ask_groq_chat,
    retrieve_context,
    ask_with_tools,
    execute_tool,
)


DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = (
    "You are a personal assistant. Give direct, concise answers. "
    "Use the screenshot when it is helpful, but do not say things like "
    "'this is not visible in the screenshot' or talk about screenshot limitations "
    "unless the user is explicitly asking about something on screen and the answer truly depends on it. "
    "If the question is general, answer it normally without mentioning the screenshot. "
    "Do not claim to have done a web search or accessed live data unless that actually happened."
)

CLASSIFIER_SYSTEM_PROMPT = (
    "Classify this question into one of these categories:\n"
    "- 'vision': Questions about what's visible on screen (UI, colors, layout, code, text on screen)\n"
    "- 'search': Questions needing live/current information (weather, news, prices, trending)\n"
    "- 'docs': Questions about the user's personal documents, notes, or files\n"
    "- 'action': Requests to DO something (create reminder, save note, set alarm)\n\n"
    "Reply with exactly one word: vision, search, docs, or action."
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
    """Complete agent implementation with all workshop stages.

    This agent can:
    - See and understand screenshots (Stage 1)
    - Search the web for live information (Stage 2)
    - Intelligently route questions using LLM classification (Stage 3)
    - Remember conversation context (Stage 4)
    - Answer from user's documents via RAG (Stage 5)
    - Take actions via MCP tools (Stage 6)
    """

    def __init__(self, *, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> None:
        self.api_key = api_key
        self.model = model

        # Stage 4: Conversation memory
        self.conversation_history: list[dict] = []
        self.max_history = 10  # Keep last 10 exchanges

    def classify_question(self, question: str) -> str:
        """Stage 3: Use LLM to classify the question intent.

        Returns one of: "vision", "search", "docs", "action"
        """
        response = ask_groq_chat(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        response_lower = response.lower().strip()

        # Parse the classification
        if "action" in response_lower:
            return "action"
        elif "docs" in response_lower:
            return "docs"
        elif "search" in response_lower:
            return "search"
        else:
            return "vision"

    def answer_question(self, *, question: str, capture: CaptureResult) -> AgentResult:
        """Main entry point: route and answer the question."""

        # Stage 3: LLM-based classification
        route = self.classify_question(question)

        if route == "action":
            # Stage 6: MCP tool integration
            return self._handle_action(question)

        elif route == "search":
            # Stage 2: Web search
            answer = ask_groq_web_search(api_key=self.api_key, question=question)
            self._update_history(question, answer)
            return AgentResult(answer=answer, model="groq/compound-mini")

        elif route == "docs":
            # Stage 5: RAG over user documents
            return self._handle_docs(question, capture)

        else:
            # Stage 1: Vision (default)
            return self._handle_vision(question, capture)

    def _handle_vision(self, question: str, capture: CaptureResult) -> AgentResult:
        """Stage 1 + 4: Answer using vision with conversation history."""
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
            history=list(self.conversation_history),  # Stage 4: Pass a snapshot of history
        )

        self._update_history(question, answer)
        return AgentResult(answer=answer, model=self.model)

    def _handle_docs(self, question: str, capture: CaptureResult) -> AgentResult:
        """Stage 5: RAG - retrieve from user documents and answer."""
        # Retrieve relevant chunks
        context_chunks = retrieve_context(question, top_k=3)

        if context_chunks:
            context = "\n---\n".join(context_chunks)
            user_prompt = (
                f"Use this context from the user's documents to help answer:\n\n"
                f"{context}\n\n"
                f"Question: {question}"
            )
        else:
            # No documents indexed, fall back to vision
            user_prompt = (
                f"User question: {question}\n"
                f"(No personal documents found to reference.)\n"
                "Answer based on what you can see on screen or your general knowledge."
            )

        answer = ask_groq_vision(
            api_key=self.api_key,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            screenshot_png_bytes=capture.png_bytes,
            history=list(self.conversation_history),
        )

        self._update_history(question, answer)
        return AgentResult(answer=answer, model=self.model)

    def _handle_action(self, question: str) -> AgentResult:
        """Stage 6: MCP tool integration - take actions for the user."""
        response = ask_with_tools(
            api_key=self.api_key,
            model=self.model,
            question=question,
        )

        if response.tool_call:
            # Execute the tool
            result = execute_tool(response.tool_call)
            answer = f"Done! {result}"
        else:
            # LLM decided not to use a tool
            answer = response.text or "I'm not sure how to help with that action."

        self._update_history(question, answer)
        return AgentResult(answer=answer, model=self.model)

    def _update_history(self, question: str, answer: str) -> None:
        """Stage 4: Update conversation history."""
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Trim to max history (each exchange = 2 messages)
        max_messages = self.max_history * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def clear_history(self) -> None:
        """Clear conversation history. Useful for starting fresh."""
        self.conversation_history = []

    def answer_audio_question(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        capture: CaptureResult,
    ) -> VoiceResult:
        """Handle voice input: transcribe and answer."""
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
