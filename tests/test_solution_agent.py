from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from assistant.capture import CaptureResult
from assistant.solution_agent import (
    DEFAULT_GROQ_MODEL,
    ScreenAssistantAgent,
)


def make_capture() -> CaptureResult:
    return CaptureResult(
        captured_at="2026-03-28T12:00:00+00:00",
        width=1440,
        height=900,
        png_bytes=b"not-a-real-png",
    )


class SolutionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ScreenAssistantAgent(api_key="test-key")
        self.capture = make_capture()

    @patch("assistant.solution_agent.ask_groq_chat", return_value="action")
    def test_classify_question_supports_all_routes(self, mock_chat) -> None:
        self.assertEqual(self.agent.classify_question("save this note"), "action")
        mock_chat.return_value = "docs"
        self.assertEqual(self.agent.classify_question("what's in my notes"), "docs")
        mock_chat.return_value = "search"
        self.assertEqual(self.agent.classify_question("weather today"), "search")
        mock_chat.return_value = "vision"
        self.assertEqual(self.agent.classify_question("what is on screen"), "vision")
        mock_chat.return_value = "something unexpected"
        self.assertEqual(self.agent.classify_question("fallback"), "vision")

    @patch("assistant.solution_agent.ask_groq_web_search", return_value="Current weather is sunny.")
    @patch("assistant.solution_agent.ask_groq_chat", return_value="search")
    def test_search_route_returns_web_model_and_updates_history(
        self,
        mock_chat,
        mock_search,
    ) -> None:
        result = self.agent.answer_question(question="What's the weather today?", capture=self.capture)

        self.assertEqual(result.answer, "Current weather is sunny.")
        self.assertEqual(result.model, "groq/compound-mini")
        mock_search.assert_called_once_with(api_key="test-key", question="What's the weather today?")
        self.assertEqual(
            self.agent.conversation_history,
            [
                {"role": "user", "content": "What's the weather today?"},
                {"role": "assistant", "content": "Current weather is sunny."},
            ],
        )

    @patch("assistant.solution_agent.ask_groq_vision", return_value="Your notes mention the deadline.")
    @patch("assistant.solution_agent.retrieve_context", return_value=["Chunk A", "Chunk B"])
    @patch("assistant.solution_agent.ask_groq_chat", return_value="docs")
    def test_docs_route_uses_rag_context(
        self,
        mock_chat,
        mock_retrieve,
        mock_vision,
    ) -> None:
        result = self.agent.answer_question(question="What do my docs say about the deadline?", capture=self.capture)

        self.assertEqual(result.answer, "Your notes mention the deadline.")
        self.assertEqual(result.model, DEFAULT_GROQ_MODEL)
        mock_retrieve.assert_called_once_with("What do my docs say about the deadline?", top_k=3)
        _, kwargs = mock_vision.call_args
        self.assertIn("Chunk A\n---\nChunk B", kwargs["user_prompt"])
        self.assertEqual(kwargs["history"], [])

    @patch("assistant.solution_agent.ask_groq_vision", return_value="I can answer from what is on screen.")
    @patch("assistant.solution_agent.retrieve_context", return_value=[])
    @patch("assistant.solution_agent.ask_groq_chat", return_value="docs")
    def test_docs_route_falls_back_when_no_indexed_docs(
        self,
        mock_chat,
        mock_retrieve,
        mock_vision,
    ) -> None:
        self.agent.answer_question(question="Can you answer from my files?", capture=self.capture)

        _, kwargs = mock_vision.call_args
        self.assertIn("No personal documents found to reference.", kwargs["user_prompt"])

    @patch("assistant.solution_agent.execute_tool", return_value="Reminder created: 'study' at 5pm")
    @patch(
        "assistant.solution_agent.ask_with_tools",
        return_value=SimpleNamespace(
            text=None,
            tool_call=SimpleNamespace(name="create_reminder", arguments={"title": "study", "time": "5pm"}),
        ),
    )
    @patch("assistant.solution_agent.ask_groq_chat", return_value="action")
    def test_action_route_executes_tools(
        self,
        mock_chat,
        mock_ask_with_tools,
        mock_execute_tool,
    ) -> None:
        result = self.agent.answer_question(question="Remind me to study at 5pm", capture=self.capture)

        self.assertEqual(result.answer, "Done! Reminder created: 'study' at 5pm")
        mock_execute_tool.assert_called_once()
        self.assertEqual(
            self.agent.conversation_history[-2:],
            [
                {"role": "user", "content": "Remind me to study at 5pm"},
                {"role": "assistant", "content": "Done! Reminder created: 'study' at 5pm"},
            ],
        )

    @patch("assistant.solution_agent.execute_tool", return_value="Opened app: 'Spotify'")
    @patch(
        "assistant.solution_agent.ask_with_tools",
        return_value=SimpleNamespace(
            text=None,
            tool_call=SimpleNamespace(name="open_app", arguments={"name": "Spotify"}),
        ),
    )
    @patch("assistant.solution_agent.ask_groq_chat", return_value="action")
    def test_action_route_opens_app(
        self,
        mock_chat,
        mock_ask_with_tools,
        mock_execute_tool,
    ) -> None:
        result = self.agent.answer_question(question="Open Spotify", capture=self.capture)

        self.assertEqual(result.answer, "Done! Opened app: 'Spotify'")
        mock_execute_tool.assert_called_once()
        self.assertEqual(
            self.agent.conversation_history[-2:],
            [
                {"role": "user", "content": "Open Spotify"},
                {"role": "assistant", "content": "Done! Opened app: 'Spotify'"},
            ],
        )

    @patch(
        "assistant.solution_agent.ask_with_tools",
        return_value=SimpleNamespace(text="I can't do that yet.", tool_call=None),
    )
    @patch("assistant.solution_agent.ask_groq_chat", return_value="action")
    def test_action_route_can_return_text_without_tool_call(
        self,
        mock_chat,
        mock_ask_with_tools,
    ) -> None:
        result = self.agent.answer_question(question="Do something impossible", capture=self.capture)

        self.assertEqual(result.answer, "I can't do that yet.")

    @patch("assistant.solution_agent.ask_groq_vision", side_effect=["First answer", "Second answer"])
    @patch("assistant.solution_agent.ask_groq_chat", return_value="vision")
    def test_vision_route_reuses_conversation_history(
        self,
        mock_chat,
        mock_vision,
    ) -> None:
        first = self.agent.answer_question(question="What app is this?", capture=self.capture)
        second = self.agent.answer_question(question="What color is the header?", capture=self.capture)

        self.assertEqual(first.answer, "First answer")
        self.assertEqual(second.answer, "Second answer")
        first_call_history = mock_vision.call_args_list[0].kwargs["history"]
        second_call_history = mock_vision.call_args_list[1].kwargs["history"]
        self.assertEqual(first_call_history, [])
        self.assertEqual(
            second_call_history,
            [
                {"role": "user", "content": "What app is this?"},
                {"role": "assistant", "content": "First answer"},
            ],
        )

    def test_history_is_trimmed_to_max_exchanges(self) -> None:
        for index in range(self.agent.max_history + 2):
            self.agent._update_history(f"q{index}", f"a{index}")

        self.assertEqual(len(self.agent.conversation_history), self.agent.max_history * 2)
        self.assertEqual(
            self.agent.conversation_history[0],
            {"role": "user", "content": "q2"},
        )
        self.assertEqual(
            self.agent.conversation_history[-1],
            {"role": "assistant", "content": f"a{self.agent.max_history + 1}"},
        )

    def test_clear_history_resets_memory(self) -> None:
        self.agent._update_history("hello", "hi")
        self.agent.clear_history()
        self.assertEqual(self.agent.conversation_history, [])

    @patch("assistant.solution_agent.transcribe_audio", return_value="Summarize this screen")
    @patch("assistant.solution_agent.ask_groq_vision", return_value="Here is the summary.")
    @patch("assistant.solution_agent.ask_groq_chat", return_value="vision")
    def test_answer_audio_question_transcribes_then_answers(
        self,
        mock_chat,
        mock_vision,
        mock_transcribe,
    ) -> None:
        result = self.agent.answer_audio_question(
            audio_bytes=b"audio",
            filename="voice.wav",
            content_type="audio/wav",
            capture=self.capture,
        )

        self.assertEqual(result.transcript, "Summarize this screen")
        self.assertEqual(result.transcription_model, "whisper-large-v3-turbo")
        self.assertEqual(result.answer, "Here is the summary.")


if __name__ == "__main__":
    unittest.main()
