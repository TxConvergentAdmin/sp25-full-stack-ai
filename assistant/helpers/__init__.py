"""Helper functions for the screen assistant workshop.

These utilities are pre-built so students can focus on the agent logic.
"""

from assistant.helpers.chat import ask_groq_chat
from assistant.helpers.rag import index_documents, retrieve_context, clear_index
from assistant.helpers.mcp_tools import (
    TOOLS,
    ToolCall,
    ToolResponse,
    ask_with_tools,
    execute_tool,
)

__all__ = [
    # Stage 3: Chat/classification
    "ask_groq_chat",
    # Stage 5: RAG
    "index_documents",
    "retrieve_context",
    "clear_index",
    # Stage 6: MCP tools
    "TOOLS",
    "ToolCall",
    "ToolResponse",
    "ask_with_tools",
    "execute_tool",
]
