# AI Context

- Project goal: a lightweight local Python screen assistant with a floating overlay.
- Current state: local Python app with a FastAPI server plus a PyWebView window using an always-on-top, frameless, glassy overlay.
- Current UI: input-first overlay with an Ask button, hidden answer area until a response appears, drag support, and a close control.
- Current backend: `GET /` serves the overlay HTML and `POST /ask` captures a fresh screenshot, then `assistant/agent.py` routes either to Groq vision or to Groq web search for live/current-info questions.
- Entry point: `main.py`. Intended start command is `python main.py`; in this environment it has been launched with `python3 main.py`.
- Main files: `main.py` handles startup and HTTP routing. `assistant/capture.py` handles screen capture. `assistant/agent.py` orchestrates prompts and routing. `assistant/vision.py` handles image questions. `assistant/search.py` handles live web-search questions. `assistant/overlay/index.html` contains the overlay UI.

## Planned Core Functionality

- Optionally index uploaded PDFs and retrieve relevant context.
- Return an answer using screen context and optional document context.
- Show the answer in the floating overlay window.

## Dependencies

- Runtime/UI: `fastapi`, `uvicorn`, `pywebview`
- Screen + image handling: `mss`, `pillow`
- Model/API: Groq vision and Groq web search via `httpx`; available dependency list still includes `anthropic`
- Retrieval/docs: `chromadb`, `sentence-transformers`, `pymupdf`, `python-multipart`

## Implementation Notes

- Groq API key is loaded from root `.env` as `GROQ_API_KEY`.
- Time-sensitive questions like deals, weather, news, and latest/current info are routed to web search instead of pure screenshot analysis.
- Keep the overlay local-only: no auth, no database, no multi-user concerns.
- Future modules are expected to follow the original split: capture, vision, rag, and overlay UI.
