"""MCP (Model Context Protocol) tool definitions and execution.

This module provides tool schemas and execution for Stage 6 of the workshop.
These are real implementations that actually perform actions (save files, etc.).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx


# Directory paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
REMINDERS_FILE = PROJECT_ROOT / "reminders.json"
NOTES_DIR = PROJECT_ROOT / "notes"

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


# Tool definitions in OpenAI function-calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder for the user. Use this when the user asks to be reminded about something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "What to remind the user about",
                    },
                    "time": {
                        "type": "string",
                        "description": "When to remind (e.g., '5pm', 'tomorrow at 9am', '2024-01-15T17:00:00')",
                    },
                },
                "required": ["title", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note to the user's notes folder. Use this when the user asks to save, write down, or remember information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title for the note (used as filename)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the note",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application on the user's computer. Use this when the user asks to open or launch an app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the app to open (e.g., 'Spotify', 'Chrome', 'Terminal')",
                    },
                },
                "required": ["name"],
            },
        },
    },
]


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    name: str
    arguments: dict


@dataclass
class ToolResponse:
    """Response from ask_with_tools."""

    text: str | None  # Text response if no tool was called
    tool_call: ToolCall | None  # Tool call if the LLM chose to use a tool


def _create_reminder(title: str, time: str) -> str:
    """Create a reminder and save it to reminders.json.

    Args:
        title: What to remind about.
        time: When to remind.

    Returns:
        Confirmation message.
    """
    # Load existing reminders
    reminders = []
    if REMINDERS_FILE.exists():
        try:
            reminders = json.loads(REMINDERS_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            reminders = []

    # Add new reminder
    reminder = {
        "id": len(reminders) + 1,
        "title": title,
        "time": time,
        "created_at": datetime.now().isoformat(),
    }
    reminders.append(reminder)

    # Save back to file
    REMINDERS_FILE.write_text(json.dumps(reminders, indent=2))

    return f"Reminder created: '{title}' at {time}"


def _save_note(title: str, content: str) -> str:
    """Save a note to the notes directory.

    Args:
        title: Title for the note (used as filename).
        content: The note content.

    Returns:
        Confirmation message.
    """
    # Ensure notes directory exists
    NOTES_DIR.mkdir(exist_ok=True)

    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title.strip()[:50]  # Limit length
    if not safe_title:
        safe_title = "note"

    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.txt"
    filepath = NOTES_DIR / filename

    # Write the note
    note_content = f"# {title}\n\nCreated: {datetime.now().isoformat()}\n\n{content}"
    filepath.write_text(note_content)

    return f"Note saved: {filename}"


def _open_app(name: str) -> str:
    """Open an application by name.

    Args:
        name: The name of the app to open.

    Returns:
        Confirmation or error message.
    """
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["open", "-a", name], capture_output=True, timeout=5
            )
        elif sys.platform == "win32":
            result = subprocess.run(
                ["cmd", "/c", "start", "", name], shell=True, capture_output=True, timeout=5
            )
        else:
            result = subprocess.run(
                ["xdg-open", name], capture_output=True, timeout=5
            )

        if result.returncode != 0:
            error = result.stderr.decode().strip()
            return f"Failed to open '{name}': {error or 'unknown error'}"

        return f"Opened app: '{name}'"
    except subprocess.TimeoutExpired:
        return f"Failed to open '{name}': command timed out"
    except Exception as e:
        return f"Failed to open '{name}': {e}"


def execute_tool(tool_call: ToolCall) -> str:
    """Execute a tool call and return the result.

    Args:
        tool_call: The tool call to execute.

    Returns:
        Result message from the tool.

    Raises:
        ValueError: If the tool name is unknown.
    """
    if tool_call.name == "create_reminder":
        return _create_reminder(
            title=tool_call.arguments.get("title", ""),
            time=tool_call.arguments.get("time", ""),
        )
    elif tool_call.name == "save_note":
        return _save_note(
            title=tool_call.arguments.get("title", ""),
            content=tool_call.arguments.get("content", ""),
        )
    elif tool_call.name == "open_app":
        return _open_app(
            name=tool_call.arguments.get("name", ""),
        )
    else:
        raise ValueError(f"Unknown tool: {tool_call.name}")


def ask_with_tools(
    *,
    api_key: str,
    model: str,
    question: str,
    screenshot_png_bytes: bytes | None = None,
    system_prompt: str = "You are a helpful assistant that can take actions for the user.",
) -> ToolResponse:
    """Send a request to Groq with tool definitions and parse the response.

    This function allows the LLM to decide whether to:
    1. Call a tool (create_reminder, save_note, etc.)
    2. Just respond with text

    Args:
        api_key: Groq API key.
        model: Model name.
        question: The user's question or request.
        screenshot_png_bytes: Optional screenshot (not used for tool calls currently).
        system_prompt: System instructions.

    Returns:
        ToolResponse with either text or a tool_call.

    Example:
        >>> response = ask_with_tools(
        ...     api_key="...",
        ...     model="meta-llama/llama-4-scout-17b-16e-instruct",
        ...     question="Remind me to review my notes at 5pm",
        ... )
        >>> if response.tool_call:
        ...     result = execute_tool(response.tool_call)
        ...     print(result)  # "Reminder created: 'review my notes' at 5pm"
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",  # Let the model decide
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

    # Check if the model wants to call a tool
    tool_calls = message.get("tool_calls")
    if tool_calls:
        # Take the first tool call
        tc = tool_calls[0]
        function = tc.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        return ToolResponse(
            text=None,
            tool_call=ToolCall(name=name, arguments=arguments),
        )

    # No tool call, return text response
    content = (message.get("content") or "").strip()
    return ToolResponse(text=content, tool_call=None)
