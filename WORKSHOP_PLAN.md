# Workshop: Build an AI Screen Assistant

**Duration:** ~45 minutes + open hacking time
**Audience:** CS students with Python experience
**Focus:** AI integration patterns & agent routing logic

---

## Overview

In this workshop, you'll learn how a local desktop assistant uses AI to answer questions about what's on your screen. The app captures a screenshot, decides whether to analyze the image or search the web, and returns a concise answer — all in a few seconds.

By the end, you'll understand how to wire up LLM APIs for vision and search, how to build a simple agent that routes between tools, and how to extend the system with your own capabilities.

---

## Pre-Workshop Setup (5 min)

Have participants do this before the session starts (or in the first 5 minutes):

1. Clone the repo and `cd` into it
2. Create a `.env` file with a Groq API key:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
   > Free keys at [console.groq.com](https://console.groq.com)
3. Install dependencies:
   ```bash
   pip install fastapi uvicorn pywebview mss pillow httpx python-dotenv python-multipart sounddevice numpy
   ```
4. Test it works: `python main.py` — a floating overlay should appear

---

## Part 1: The AI Layer (15 min)

**Goal:** Understand how the app talks to an LLM with both text and images.

### 1.1 — Vision: Asking an LLM About a Screenshot

Open `assistant/vision.py` and walk through it together.

**Key concepts to teach:**

- **Multimodal messages** — LLMs can accept both text and images in a single request. The message payload looks like:
  ```python
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": [
          {"type": "text", "text": user_prompt},
          {"type": "image_url", "image_url": {"url": data_url}}
      ]}
  ]
  ```
- **Image preprocessing matters** — Screenshots are huge. The code resizes to 1600×1600 max and compresses JPEG quality in a loop until it fits under 4MB. This is a real-world constraint worth highlighting.
- **The API is OpenAI-compatible** — Groq (and many other providers) use the same `/v1/chat/completions` endpoint format. Point out that switching providers is mostly just changing the base URL and model name.
- **Temperature = 0.2** — Low temperature for factual, consistent answers. Good time to briefly explain what temperature does.

**Live demo:** Run the app, ask "What app is open on my screen?" and show the screenshot → answer flow.

### 1.2 — Web Search: When Vision Isn't Enough

Open `assistant/search.py`.

**Key concepts to teach:**

- **Tool use / function calling** — The `compound-mini` model has built-in web search. The code enables it via:
  ```python
  "compound_custom": {"tools": {"enabled_tools": ["web_search"]}}
  ```
  The model decides *on its own* when and what to search. This is a simple but powerful example of tool use.
- **Different models for different jobs** — Vision uses a large multimodal model (Llama 4 Scout 17B). Search uses a smaller, tool-augmented model. Picking the right model for the task is a core engineering decision.

**Live demo:** Ask "What's the weather today?" and show it routes to web search instead of analyzing the screenshot.

---

## Part 2: The Agent Layer (15 min)

**Goal:** Understand the routing logic that makes this an *agent* rather than just an API wrapper.

### 2.1 — What Makes This an "Agent"?

Open `assistant/agent.py`. This is the brain of the app.

**Key concept:** An agent is code that *decides what to do* based on the input, rather than always doing the same thing. Here, the agent chooses between two tools: vision analysis and web search.

Walk through the `ScreenAssistantAgent` class:

```
User question
    ↓
needs_web_search(question)?
    ├── YES → ask_groq_web_search()   (live data from the web)
    └── NO  → capture screen → ask_groq_vision()  (analyze what's on screen)
```

### 2.2 — The Routing Decision

Look at `needs_web_search()` together:

```python
def needs_web_search(self, question: str) -> bool:
    q = question.lower()
    web_keywords = ["today", "latest", "current", "news", "weather", ...]
    if any(kw in q for kw in web_keywords):
        return True
    web_patterns = [r"\bbest\b", r"\brecommend", r"\bcheapest\b", ...]
    return any(re.search(p, q) for p in web_patterns)
```

**Discussion points:**

- **Rule-based vs. LLM-based routing** — This uses keywords and regex. It's fast and free (no API call). But it's brittle — "What's the best font on my screen?" would incorrectly route to web search. Ask students: how would you improve this?
- **The alternative: use an LLM to route** — You could send the question to a cheap/fast model and ask "Does this need web search or screen analysis?" More accurate, but adds latency and cost. This is a real tradeoff engineers make.
- **Agent = decision + action** — The routing logic is the "decision" part. The vision/search modules are the "action" part. Even simple agents follow this pattern.

### 2.3 — The Full Flow

Walk through `answer_question()`:

```python
async def answer_question(self, question, capture):
    if self.needs_web_search(question):
        answer = ask_groq_web_search(self.api_key, question)
        return AgentResult(answer=answer, model="compound-mini")

    png_bytes = capture.png_bytes
    answer = ask_groq_vision(self.api_key, self.model,
                             self.system_prompt, question, png_bytes)
    return AgentResult(answer=answer, model=self.model)
```

**Point out:** The caller doesn't need to know *how* the answer was generated. The agent abstracts the routing. This is a clean pattern that scales — you can add more tools without changing the interface.

---

## Part 3: How It All Connects (5 min)

Quick walkthrough of `main.py` to show the glue:

- **FastAPI server** serves the UI and handles `/ask` and `/voice` endpoints
- **PyWebView** creates the floating desktop overlay
- The `/ask` endpoint captures a screenshot, passes it + the question to the agent, returns the answer
- The UI (`assistant/overlay/index.html`) is a simple HTML/JS frontend that calls these endpoints

This part is quick — the point is just to show how the AI/agent pieces plug into a real app.

---

## Recap (2 min)

Three things to remember:

1. **LLMs are APIs** — Vision, search, and transcription are all just HTTP calls with different payloads. The OpenAI-compatible format is the lingua franca.
2. **Agents decide, then act** — Even a simple if/else router counts. The pattern is: inspect input → choose a tool → call it → return the result.
3. **Engineering is about tradeoffs** — Rule-based vs. LLM routing. Big model vs. small model. Accuracy vs. speed vs. cost.

---

## Extension Paths

These are independent tracks participants can follow after the workshop. Each one adds a new capability to the assistant.

---

### Path A: Smarter Routing with an LLM

**Difficulty:** ⭐ Easy | **Time:** 30 min | **Concepts:** Prompt engineering, classification

**What you'll do:** Replace the keyword-based `needs_web_search()` with an LLM call that classifies the question.

**Steps:**
1. In `agent.py`, create a new method `classify_question()` that sends the user's question to a fast model (e.g., `llama-3.1-8b-instant` on Groq)
2. Use a system prompt like:
   ```
   Classify this question as either "vision" (needs to look at the screen)
   or "search" (needs live web data). Reply with one word only.
   ```
3. Replace the `needs_web_search()` call in `answer_question()` with your classifier
4. Test with edge cases: "What's the best font on my screen?" should now route to vision

**Stretch:** Add a third category — "general knowledge" — that answers from the LLM's training data without a screenshot or web search (saves time and tokens).

---

### Path B: Add Memory / Conversation History

**Difficulty:** ⭐⭐ Medium | **Time:** 45 min | **Concepts:** State management, context windows

**What you'll do:** Make the assistant remember previous Q&A pairs so you can have a conversation.

**Steps:**
1. Add a `conversation_history: list` to `ScreenAssistantAgent`
2. After each Q&A, append `{"role": "user", "content": question}` and `{"role": "assistant", "content": answer}`
3. In `vision.py`, modify `ask_groq_vision()` to accept and include history in the messages array
4. Cap history at the last 5 exchanges to stay within token limits
5. Add a "Clear" button in the overlay UI

**Stretch:** Only include history when the question seems like a follow-up (e.g., starts with "what about", "and", "also"). Use an LLM or simple heuristics to decide.

---

### Path C: Add a New Tool — File/Code Analysis

**Difficulty:** ⭐⭐ Medium | **Time:** 1 hour | **Concepts:** Tool use, agent architecture

**What you'll do:** Add a third tool that reads and analyzes files from your computer.

**Steps:**
1. Create `assistant/files.py` with a function `analyze_file(api_key, file_path, question)`
2. It should read the file contents and send them to an LLM with the user's question
3. In `agent.py`, add file-related keywords to routing (e.g., "this file", "my code", "the document")
4. Add a new route — or modify the routing logic to detect when a question is about a file
5. For bonus points, let the user drag-and-drop a file onto the overlay

**Stretch:** Support multiple file types — use different prompts for code files vs. PDFs vs. images.

---

### Path D: RAG — Give It Your Documents

**Difficulty:** ⭐⭐⭐ Hard | **Time:** 1.5 hours | **Concepts:** Embeddings, vector search, RAG

**What you'll do:** Let the assistant answer questions from a folder of your PDFs/notes using Retrieval-Augmented Generation.

**Steps:**
1. Install: `pip install chromadb sentence-transformers pymupdf`
2. Create `assistant/rag.py`:
   - Write an `index_documents(folder_path)` function that reads PDFs, chunks text, and stores embeddings in ChromaDB
   - Write a `retrieve(query, top_k=3)` function that finds the most relevant chunks
3. In `agent.py`, add a `needs_document_search()` check (or extend the LLM classifier from Path A)
4. When triggered, retrieve relevant chunks and include them in the LLM prompt as context
5. Add a startup step in `main.py` to index a `~/documents` folder

**Stretch:** Show which document and page the answer came from. Add re-ranking for better retrieval quality.

---

### Path E: Multi-Step Agent with Planning

**Difficulty:** ⭐⭐⭐ Hard | **Time:** 1.5 hours | **Concepts:** Agentic loops, planning, tool chaining

**What you'll do:** Turn the single-step agent into one that can break down complex questions and use multiple tools in sequence.

**Steps:**
1. Define your tools as a list of descriptions:
   ```python
   tools = [
       {"name": "screen_vision", "description": "Analyze what's on screen"},
       {"name": "web_search", "description": "Search the web for current info"},
       {"name": "file_read", "description": "Read and analyze a local file"},
   ]
   ```
2. Create a `plan()` method that sends the question + tool descriptions to an LLM and asks it to output a plan (list of steps with tool names)
3. Create an `execute_plan()` method that runs each step, feeding results into the next
4. Handle the case where the LLM's plan doesn't make sense (validation, retries)

**Example:** "Compare what's on my screen to the latest version of this library's docs" → Step 1: screen_vision → Step 2: web_search → Step 3: synthesize

**Stretch:** Add a `reflect()` step where the agent reviews its own answer and decides if it needs to try again.

---

### Path F: Voice Output (Text-to-Speech)

**Difficulty:** ⭐ Easy | **Time:** 30 min | **Concepts:** TTS APIs, async audio

**What you'll do:** Make the assistant speak its answers out loud.

**Steps:**
1. Create `assistant/tts.py` using Groq's or another provider's TTS endpoint
2. After getting an answer in `main.py`, send it to TTS and play the audio
3. Use `sounddevice` (already installed) to play the WAV/PCM output
4. Add a toggle in the UI to enable/disable voice output

**Stretch:** Stream the audio so the voice starts before the full answer is generated.

---

## Quick Reference: Key Files

| File | What It Does | Lines |
|------|-------------|-------|
| `assistant/agent.py` | Agent routing — decides vision vs. search | ~116 |
| `assistant/vision.py` | Sends screenshot + question to Groq vision API | ~75 |
| `assistant/search.py` | Web search via Groq compound-mini | ~51 |
| `assistant/speech.py` | Audio transcription via Whisper | ~37 |
| `assistant/capture.py` | Takes screenshots using `mss` | ~47 |
| `main.py` | FastAPI server + PyWebView overlay | ~280 |
| `assistant/overlay/index.html` | The floating UI | ~339 |
