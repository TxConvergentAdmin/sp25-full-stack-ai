# AI Screen Assistant Workshop

This project is a workshop app that turns a desktop overlay into a simple AI assistant. It can:

- capture your screen
- answer questions about what is visible
- route current-events questions to web search
- accept voice input
- serve as a starter project for memory, RAG, and tool-calling extensions

## 1. Install Python First

You need Python 3.10 or newer.

### macOS

Option A: Download Python from the official site

1. Go to https://www.python.org/downloads/
2. Download the latest Python 3 release
3. Run the installer
4. Verify:

```bash
python3 --version
```

Option B: Install with Homebrew

```bash
brew install python
python3 --version
```

### Windows

1. Go to https://www.python.org/downloads/
2. Download Python 3
3. During installation, check `Add Python to PATH`
4. Verify in PowerShell:

```powershell
python --version
```

If `python` does not work, try:

```powershell
py --version
```

### Linux

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

## 2. Clone and Enter the Project

```bash
git clone https://github.com/TxConvergentAdmin/sp25-full-stack-ai.git
cd sp25-full-stack-ai
```

## 3. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 4. Install Project Dependencies

Install the packages used by the base app:

```bash
pip install fastapi uvicorn pywebview mss pillow httpx python-dotenv python-multipart sounddevice numpy
```

Optional packages for workshop extensions:

```bash
pip install chromadb sentence-transformers pymupdf
```

## 5. Add Your API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can get a key from https://console.groq.com

## 6. Run the App

```bash
python main.py
```

This starts:

- a FastAPI backend at `http://127.0.0.1:8123`
- a floating desktop overlay window powered by PyWebView

## 7. macOS Permissions

If you are on macOS, you may need to allow:

- Screen Recording for your Python app or terminal
- Microphone access for voice input

If screen capture fails, check:

`System Settings -> Privacy & Security -> Screen Recording`

## 8. What the App Does

At a high level, the app works like this:

1. The UI sends a question to the backend.
2. The backend captures the current screen.
3. The agent decides how to answer:
   - vision for screen-based questions
   - web search for live/current questions
   - voice flow when audio is recorded
4. The answer is returned to the overlay.

## 9. Code Structure

### Entry Point

- [main.py](/Users/spotta/Desktop/Projects/workshop-cluely/main.py)
  Starts FastAPI, launches the floating overlay, handles `/ask` and `/voice`, and wires together capture, recording, and the agent.

### Core Assistant Modules

- [assistant/agent.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/agent.py)
  The starter workshop agent. It currently handles the base routing flow and is the one used by `main.py`.

- [assistant/solution_agent.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/solution_agent.py)
  A completed reference implementation with all workshop stages, including classification, memory, RAG, and tool actions.

- [assistant/capture.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/capture.py)
  Captures the current screen and returns PNG bytes plus metadata like width, height, and timestamp.

- [assistant/vision.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/vision.py)
  Sends a screenshot and prompt to Groq's vision-capable chat API.

- [assistant/search.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/search.py)
  Sends current-information questions to Groq's web-enabled model.

- [assistant/speech.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/speech.py)
  Transcribes recorded audio using Groq's transcription API.

- [assistant/recorder.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/recorder.py)
  Records microphone audio locally and returns WAV bytes.

- [assistant/overlay/index.html](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/overlay/index.html)
  The simple frontend for the floating overlay.

### Helper Modules For Later Workshop Stages

- [assistant/helpers/chat.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/helpers/chat.py)
  Lightweight text-only Groq chat helper for things like question classification.

- [assistant/helpers/rag.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/helpers/rag.py)
  Document indexing and retrieval helpers for RAG over PDFs and notes.

- [assistant/helpers/mcp_tools.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/helpers/mcp_tools.py)
  Tool definitions and execution logic for reminder creation and note saving.

- [assistant/helpers/__init__.py](/Users/spotta/Desktop/Projects/workshop-cluely/assistant/helpers/__init__.py)
  Re-exports helper functions used in later workshop stages.

### Workshop Materials

- [WORKSHOP_PLAN.md](/Users/spotta/Desktop/Projects/workshop-cluely/WORKSHOP_PLAN.md)
  The teaching outline and stage-by-stage workshop plan.

- [AI_CONTEXT.md](/Users/spotta/Desktop/Projects/workshop-cluely/AI_CONTEXT.md)
  Supporting workshop context and notes.

### Output / Workspace Folders

- `notes/`
  Used by the tool-calling helper to save notes created by the assistant.

- `reminders.json`
  Created when the reminder tool is used.

- `.chroma_db/`
  Created if you use the RAG extension.

## 10. Current Functionality vs Workshop Extensions

### What works in the starter app now

- screen capture
- vision-based answers
- keyword-based routing to web search
- voice recording and transcription
- floating desktop overlay UI

### What the full solution supports

The completed solution in `assistant/solution_agent.py` adds:

- LLM-based question classification
- conversation memory
- document retrieval with RAG
- tool calls for actions like reminders and notes

Note:

`main.py` currently imports `ScreenAssistantAgent` from `assistant/agent.py`, which is the workshop starter version. If you want the running app to use the completed implementation, switch the import to `assistant.solution_agent`.

## 11. Example Questions To Try

Vision:

- "What app is open on my screen?"
- "What color is this header?"
- "Summarize what you see."

Search:

- "What's the weather today?"
- "What is the latest OpenAI news?"

Voice:

- Start recording from the overlay and ask a screen-related question out loud.

## 12. Run Tests

The repo includes tests for the completed solution agent:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test*.py' -v
```

## 13. Common Troubleshooting

### `Missing GROQ_API_KEY in .env`

Create the `.env` file in the project root and add your API key.

### Screen capture errors on macOS

Grant Screen Recording permission to Python, your terminal, or the app used to launch Python.

### Microphone recording does not work

Grant microphone permission and make sure your default input device is available.

### `ModuleNotFoundError`

Make sure your virtual environment is activated and the dependencies were installed with `pip install ...`.

## 14. Workshop Progression

This repository is designed to teach a progression:

1. Vision
2. Web search
3. Smarter routing
4. Memory
5. RAG
6. Tool use

The starter code lives in `assistant/agent.py`, and the reference answer lives in `assistant/solution_agent.py`.
