# Workshop: Build an AI Screen Assistant

**Duration:** ~60–75 min (live build) + open hacking time
**Audience:** CS students with Python experience
**Format:** Progressive live build — each stage reveals a problem, then you fix it
**Goal:** Build a personal assistant that sees your screen, searches the web, remembers context, knows your documents, and takes actions

---

## The Arc

```
Stage 1: "It can't see"          → Wire up vision          → now it sees your screen
Stage 2: "It hallucinates"       → Add web search + routing → now it uses the right tool
Stage 3: "Routing is dumb"       → LLM-based classification → now it understands intent
Stage 4: "It forgets everything" → Conversation memory      → now follow-ups work
Stage 5: "It doesn't know me"    → RAG pipeline             → now it reads your docs
Stage 6: "It can't do anything"  → MCP tool integration     → now it takes actions
```

---

## What's Pre-Built vs. What Students Build

**Already working (helpers — don't touch):**

| File | What it does |
|------|-------------|
| `main.py` | FastAPI server + PyWebView overlay, all endpoints, window management |
| `assistant/overlay/index.html` | The floating dark-mode UI |
| `assistant/capture.py` | Screenshot capture service |
| `assistant/recorder.py` | Audio recording |
| `assistant/speech.py` | Whisper transcription |
| `assistant/vision.py` | Groq vision API call (already supports `history` param) |
| `assistant/search.py` | Groq web search API call |
| `assistant/helpers/chat.py` | Generic `ask_groq_chat()` — used for LLM classification |
| `assistant/helpers/rag.py` | `index_documents()` + `retrieve_context()` — ChromaDB + sentence-transformers |
| `assistant/helpers/mcp_tools.py` | `ask_with_tools()` + `execute_tool()` — tool definitions and dispatcher |

**Students build:** `assistant/agent.py` — the brain. They uncomment TODOs and add logic stage by stage.

---

## Pre-Workshop Setup (5 min)

1. Clone the repo and `cd` into it
2. Create `.env` with a Groq API key (free at [console.groq.com](https://console.groq.com)):
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
3. Install dependencies:
   ```bash
   pip install fastapi uvicorn pywebview mss pillow httpx python-dotenv python-multipart sounddevice numpy chromadb sentence-transformers pymupdf
   ```
4. Run `python main.py` — overlay appears, but asking a question fails

---

## Stage 1: Vision — "It Can't See" (~8 min)

**The problem:** Run the app. Ask "What's on my screen?" It crashes or returns a placeholder — the app captures a screenshot but has no idea what to do with it.

**The fix:** In `agent.py`, wire up `answer_question()` to call the pre-built `ask_groq_vision()` helper with the screenshot bytes and the user's question.

**What to teach:** Multimodal messages (text + image in one API call), image preprocessing (the compression loop in `_prepare_image_data_url` that gets screenshots under 4MB), the OpenAI-compatible API format (Groq, OpenAI, and others all use the same `/v1/chat/completions` endpoint), and temperature (0.2 = factual and consistent).

**Demo:** Ask "What app is open?" — it sees the screen and answers correctly.

**Code — students fill in `answer_question()`:**
```python
def answer_question(self, *, question: str, capture: CaptureResult) -> AgentResult:
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
```

---

## Stage 2: Web Search + Routing — "It Hallucinates" (~8 min)

**The problem:** Ask "What's the weather in Austin today?" It stares at the screenshot and makes something up. Vision models have zero access to live data.

**The fix:** Add `ask_groq_web_search()` as a second tool and write `needs_web_search()` — a simple keyword/regex check that routes time-sensitive questions to search instead of vision.

**What to teach:** This is what makes it an *agent* — it decides which tool to use instead of always doing the same thing. Point out that `compound-mini` has built-in web search via `compound_custom.tools`, and that different models are better for different jobs (large multimodal model for vision, small tool-augmented model for search).

**Demo:** Ask "What's the weather today?" — routes to search, returns live data. Ask "What color is this button?" — routes to vision.

**Code — students add routing:**
```python
def needs_web_search(self, question: str) -> bool:
    q = question.lower()
    if any(term in q for term in WEB_SEARCH_TERMS):
        return True
    if re.search(r"\b(best|recommend|compare|cheapest|near me)\b", q):
        return True
    return False

def answer_question(self, *, question: str, capture: CaptureResult) -> AgentResult:
    if self.needs_web_search(question):
        answer = ask_groq_web_search(api_key=self.api_key, question=question)
        return AgentResult(answer=answer, model="groq/compound-mini")
    # ... existing vision code ...
```

---

## Stage 3: Smarter Routing — "Routing Is Dumb" (~8 min)

**The problem:** Ask "What's the best font on my screen?" It routes to web search because "best" is a keyword — but the user is asking about something *on screen*. Keywords are fast and free but miss nuance.

**The fix:** Uncomment the `ask_groq_chat` import from `assistant.helpers` and write a `classify_question()` method that sends the question to the same model with a one-line classification prompt: "Classify as 'vision' or 'search'. One word only." Replace `needs_web_search()` with this classifier.

**What to teach:** Rule-based vs. LLM-based routing (the core tradeoff: speed/cost vs. accuracy), prompt engineering for classification (constrained output, low temperature), and the concept of LLM-as-judge — using a model to make decisions, not just generate text.

**Demo:** "What's the best font on my screen?" now correctly routes to vision. "What's the best restaurant near me?" still routes to search.

**Code — students write the classifier:**
```python
from assistant.helpers import ask_groq_chat

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
```

---

## Stage 4: Conversation Memory — "It Forgets Everything" (~10 min)

**The problem:** Ask "What color is the button in the top right?" — answers correctly. Then ask "Make it darker" — it has no idea what "it" refers to. Every question is completely independent.

**The fix:** Uncomment the `conversation_history` list in `__init__`. After each Q&A, append user and assistant messages. Pass the history to `ask_groq_vision()` (which already accepts a `history` parameter). Cap at the last 10 exchanges so you don't overflow the context window.

**What to teach:** LLMs are stateless — every request starts from scratch. "Memory" is an illusion created by appending history to every call (this is exactly what ChatGPT does). Context windows have a token limit, which is why we cap history. Show what happens if you don't: the context overflows and older messages get truncated or the request fails.

**Demo:** "What color is the button?" → Answers. "Make it darker" → Now understands the reference.

**Code — students uncomment and wire up:**
```python
# In __init__:
self.conversation_history: list[dict] = []
self.max_history = 10

# In answer_question(), after getting the answer:
answer = ask_groq_vision(..., history=self.conversation_history)

self.conversation_history.append({"role": "user", "content": question})
self.conversation_history.append({"role": "assistant", "content": answer})
if len(self.conversation_history) > self.max_history * 2:
    self.conversation_history = self.conversation_history[-(self.max_history * 2):]
```

---

## Stage 5: RAG — "It Doesn't Know Me" (~12 min)

**The problem:** Ask "What did Professor Smith say about recursion in lecture 3?" It has no idea. It can see your screen and search the web, but knows nothing about *your* documents, notes, or files.

**The fix:** Import `index_documents` and `retrieve_context` from `assistant.helpers`. Call `index_documents()` at startup to scan a docs folder. Update the classifier to return a third category: `"docs"`. When the route is `"docs"`, call `retrieve_context(question, top_k=3)` to get relevant chunks, then inject them into the prompt as context before calling the vision model.

**What to teach:** Why RAG exists (LLMs know public knowledge, not *your* stuff). Walk through the pipeline using the helper code: documents → chunks (the `_chunk_text` function splits by character count with overlap) → embeddings via `all-MiniLM-L6-v2` (text as vectors — similar meaning = close in space) → stored in ChromaDB → retrieved by cosine similarity → stuffed into the prompt as context. The LLM doesn't "know" your docs — it's reading them on the fly.

**Demo:** Drop a few class note PDFs or text files into a `docs/` folder. Ask a question about them — retrieves the relevant section and answers.

**Code — students add the docs route:**
```python
from assistant.helpers import index_documents, retrieve_context

# In answer_question(), extend the classifier prompt to include "docs":
# "Classify as 'vision', 'search', or 'docs' (user's personal documents)."

# Add the docs route:
elif route == "docs":
    context_chunks = retrieve_context(question, top_k=3)
    context = "\n---\n".join(context_chunks)
    augmented_prompt = (
        f"Use this context from the user's documents:\n\n{context}\n\n"
        f"Question: {question}"
    )
    answer = ask_groq_vision(
        api_key=self.api_key,
        model=self.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=augmented_prompt,
        screenshot_png_bytes=capture.png_bytes,
        history=self.conversation_history,
    )
```

```python
# In main.py startup, add:
from assistant.helpers import index_documents
docs_folder = Path(__file__).parent / "docs"
if docs_folder.exists():
    count = index_documents(str(docs_folder))
    print(f"Indexed {count} document chunks for RAG")
```

---

## Stage 6: MCP — "It Can't Do Anything" (~10 min)

**The problem:** The assistant is now smart — sees, searches, remembers, knows your docs. But ask "Remind me to review my notes at 5pm" or "Save this as a note" — it can only *answer* questions, never *take actions*.

**The fix:** Import `ask_with_tools` and `execute_tool` from `assistant.helpers`. Extend the classifier to return `"action"` for imperative requests. When the route is `"action"`, send the question + tool definitions to the LLM. If it returns a tool call, parse the structured arguments and execute the tool. The helpers already define two tools: `create_reminder` (saves to `reminders.json`) and `save_note` (writes to a `notes/` folder).

**What to teach:** What MCP is (a standard protocol for connecting LLMs to tools — like USB for AI). How function calling works: the LLM sees tool descriptions, chooses which one to call, and outputs structured JSON arguments; your code executes the tool and returns the result. Show the tool definition format in `mcp_tools.py`. This is the core pattern behind every AI agent — decide → call tool → observe result → respond.

**Demo:** "Remind me to review my notes at 5pm" → LLM picks `create_reminder`, outputs `{"title": "review my notes", "time": "5pm"}`, tool writes to `reminders.json`, user sees confirmation.

**Code — students add the action route:**
```python
from assistant.helpers import ask_with_tools, execute_tool

# Add "action" to the classifier prompt:
# "Classify as 'vision', 'search', 'docs', or 'action' (user wants to DO something)."

# Add the action route:
elif route == "action":
    response = ask_with_tools(
        api_key=self.api_key,
        model=self.model,
        question=question,
    )
    if response.tool_call:
        result = execute_tool(response.tool_call)
        return AgentResult(answer=f"Done! {result}", model=self.model)
    return AgentResult(answer=response.text or "I wasn't sure what action to take.", model=self.model)
```

---

## Recap (2 min)

Six concepts, one per stage:

| Stage | Concept | One-liner |
|-------|---------|-----------|
| 1 | Multimodal AI | LLMs can see images, not just read text |
| 2 | Agent routing | Deciding *which* tool to use is the core of an agent |
| 3 | LLM-as-judge | Use a model to make decisions, not just generate content |
| 4 | Context management | Memory is manual — you control what the model sees |
| 5 | RAG | Give LLMs access to private knowledge through retrieval |
| 6 | Tool use / MCP | Let LLMs take actions, not just answer questions |

---

## Extension Paths

Independent tracks for after the workshop. Pick one and hack on it.

---

### Action Modes

**Difficulty:** ⭐ | **Concepts:** Intent classification, prompt templates

Add quick-action buttons: **Summarize**, **Explain**, **Find bug**, **Draft reply**, **What should I click?** Each mode uses a specialized prompt template instead of the generic one. Detect the mode from the UI (button clicks) or from the question itself. Teaches how the same model produces wildly different outputs depending on the prompt, and how production assistants handle diverse intents.

---

### Web + Screen Combined Mode

**Difficulty:** ⭐⭐ | **Concepts:** Multi-tool fusion, context synthesis

For questions like "Is this a good deal?" use *both* the screenshot and web search. Capture what's on screen, extract the claim or product, search the web for context, then synthesize both sources into one answer. Teaches multi-source reasoning and how real agents chain tools together — the next step beyond single-tool routing.

---

### Draft Replies

**Difficulty:** ⭐⭐ | **Concepts:** Prompt engineering, tone control

Detect when the screen shows an email, Slack message, or chat. Auto-generate a reply draft based on the conversation visible on screen. Teaches advanced prompt engineering (tone matching, format constraints, context extraction from screenshots) and practical AI-assisted writing.

---

### Task Extraction

**Difficulty:** ⭐⭐ | **Concepts:** Structured output, entity extraction

Analyze meeting notes, docs, or tickets visible on screen. Extract TODOs, deadlines, and action items as structured JSON. Teaches how to get the model to output structured data (not just prose) and how to validate and parse LLM outputs reliably. Pairs well with the MCP tools from Stage 6.

---

### OCR Text Extraction

**Difficulty:** ⭐ | **Concepts:** Preprocessing pipelines

Run OCR (Tesseract or EasyOCR) on the screenshot *before* sending it to the vision model. Include the extracted text alongside the image. This dramatically improves accuracy for code, dashboards, spreadsheets, and error messages — vision models understand layout but often misread small text. Teaches how preprocessing improves output quality.

---

### Screen Region Selection

**Difficulty:** ⭐⭐ | **Concepts:** Input refinement, accuracy tradeoffs

Let the user drag-select a region instead of sending the full screenshot. Smaller, focused images = more accurate answers and fewer tokens. Requires UI changes (drag overlay) and cropping logic. Teaches how input quality directly affects output quality — a principle that applies everywhere in AI.

---

### Persistent Note Memory

**Difficulty:** ⭐⭐ | **Concepts:** Long-term state, preference learning

Save user preferences (tone, work context, recurring tasks) to a JSON file. Load them into the system prompt on startup. Over time the assistant "learns" you. Teaches the difference between conversation memory (Stage 4 — short-term, in-context) and persistent memory (long-term, stored externally).

---

### Local Screenshot History

**Difficulty:** ⭐⭐ | **Concepts:** Temporal context, multi-image reasoning

Keep the last 5–10 screenshots in a ring buffer. Enable questions like "What changed?" or "Go back to what I was looking at." Teaches temporal reasoning and managing multiple images in a context window — useful for monitoring, debugging, and workflow tracking.

---

### Voice Output (TTS)

**Difficulty:** ⭐ | **Concepts:** TTS APIs, streaming audio

Add text-to-speech so the assistant speaks answers out loud. Use Groq's or another provider's TTS endpoint. `sounddevice` is already installed for playback. Completes the voice loop (speak → transcribe → think → speak back). Easy win that makes the app feel polished.

---

### Read-This-Page Mode

**Difficulty:** ⭐ | **Concepts:** Zero-input UX, prompt design

One-click button that captures the screen and generates a summary — no typing required. Just a specialized prompt ("Summarize everything visible on this screen") and a new UI button. Simple but teaches how UX decisions and prompt design work together.

---

### Clipboard Awareness

**Difficulty:** ⭐ | **Concepts:** Multi-source context

Read clipboard text and combine it with the screenshot. "Explain this error" works much better when the error text comes from the clipboard (exact) rather than OCR'd from a screenshot (approximate). Small feature, big accuracy boost for developer workflows.

---

### Calendar & Reminders (via MCP)

**Difficulty:** ⭐⭐ | **Concepts:** MCP in practice, real API integration

Extend the MCP tools from Stage 6 with real calendar integration (Google Calendar API or a local store). The `create_reminder` tool already saves to JSON — upgrade it to actually schedule notifications or create calendar events. Teaches end-to-end MCP tool implementation.

---

### App-Specific Helpers

**Difficulty:** ⭐⭐ | **Concepts:** Dynamic prompting, context detection

Detect which app is in focus (Gmail, VS Code, Slack, etc.) and swap in a specialized system prompt. For Gmail: prioritize email summarization. For VS Code: prioritize code explanation and bug finding. Teaches dynamic prompt selection and context-aware behavior.

---

### Hotkeys

**Difficulty:** ⭐ | **Concepts:** System integration, UX

Add global keyboard shortcuts (via `pynput`) to summon the overlay, trigger voice recording, or run a quick summarize. Not AI-focused but makes the tool actually usable day-to-day.

---

## Quick Reference

### Key Files

| File | Role | Students edit? |
|------|------|:-:|
| `assistant/agent.py` | The brain — routing, memory, RAG, MCP | Yes — this is the main file |
| `assistant/vision.py` | Vision API call (supports `history` param) | No |
| `assistant/search.py` | Web search API call | No |
| `assistant/helpers/chat.py` | Generic chat call (for classifier) | No — use as-is |
| `assistant/helpers/rag.py` | Chunking + embedding + retrieval | No — call from agent |
| `assistant/helpers/mcp_tools.py` | Tool definitions + executor | No — call from agent |
| `assistant/capture.py` | Screenshot service | No |
| `assistant/speech.py` | Whisper transcription | No |
| `assistant/recorder.py` | Audio recording | No |
| `main.py` | Server + overlay + endpoints | Minor (add RAG startup) |
| `assistant/overlay/index.html` | Floating UI | No (unless extending UI) |

### Models Used

| Task | Model | Why |
|------|-------|-----|
| Vision | `meta-llama/llama-4-scout-17b-16e-instruct` | Large multimodal, strong image understanding |
| Web search | `groq/compound-mini` | Small, has built-in web search tool |
| Classification | `meta-llama/llama-4-scout-17b-16e-instruct` | Fast + accurate for short tasks |
| Transcription | `whisper-large-v3-turbo` | Fast speech-to-text |
| Embeddings (RAG) | `all-MiniLM-L6-v2` (local) | Lightweight, no API needed |

### Stage Summary

| # | Problem shown | Fix applied | Concept |
|---|--------------|-------------|---------|
| 1 | Can't see the screen | Wire up `ask_groq_vision()` | Multimodal LLMs |
| 2 | Hallucinates live data | Add `ask_groq_web_search()` + keyword routing | Agent = decide + act |
| 3 | Routing misclassifies | `classify_question()` via `ask_groq_chat()` | LLM-as-judge |
| 4 | No follow-up context | `conversation_history` list + pass to vision | Context windows |
| 5 | Doesn't know your docs | `index_documents()` + `retrieve_context()` | RAG pipeline |
| 6 | Can't take actions | `ask_with_tools()` + `execute_tool()` | Tool use / MCP |
