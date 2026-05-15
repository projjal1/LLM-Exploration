# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Experimental playground for learning AI/LLM application patterns. Each top-level folder is an independent, self-contained mini-project — there is no shared package, no monorepo tooling, and no root entrypoint. Treat every subfolder as its own runnable example.

Most examples use **Ollama** as a local LLM backend (so they assume `ollama` is running locally) and pick between `llama3:8b` (general tasks) and `mistral:7b` (tool-calling / structured output). The README documents this split.

## Running examples

Each project's entrypoint is `app.py` in its folder. Examples are run directly, not as modules:

```
python <subfolder>/app.py
```

Some examples have their own `requirements.txt` (`chess-simulation/`, `foundation-llm-training/`, `rag-pipeline/document-search/`); most do not and rely on packages being installed globally in the active venv. There is no root `requirements.txt` and no test suite.

### External dependencies the code assumes are already running

- **Ollama daemon** with `llama3:8b` and/or `mistral:7b` pulled — required by nearly every `app.py`.
- **MongoDB on `localhost:27017`** — `mcp-flight-booking/db_ops.py` connects unconditionally at import time, so importing `mcp-server.py` without Mongo running will fail.
- **Tavily API key** — `agentic-workflows/essay-refinement-prompt-chaining/` and `agentic-workflows/weather-document-summarizer/` read `TAVILY_API_KEY` from a local `.env` via `python-dotenv`. `*.env` is gitignored; do not commit one.
- **MCP server reachable at `http://127.0.0.1:8000/mcp`** — `mcp-flight-booking/mcp-client.py` requires `mcp-server.py` to be running first (`python mcp-server.py`, then in another shell `python mcp-client.py`).

### Special run instructions

- `chess-simulation/app.py` launches an Eel desktop app (HTML/JS UI in `web/`) — `eel.start('index.html', ...)` opens a browser window; not a headless script.
- `foundation-llm-training/training-script.py` fine-tunes `distilgpt2` on `my_data.txt` and writes `./finetuned_model/`. `prompt-testing.py` then loads that directory and runs inference. The pipeline currently hardcodes `device="mps"` (Apple Silicon) — change to `"cuda"` or `"cpu"` when running elsewhere.
- `rag-pipeline/document-search/app.py` builds a Chroma vector store in `./chroma_langchain_db/` from `nke-10k-2023.pdf` on first run and reuses it after. Deleting that directory forces re-indexing.

## Architecture patterns to recognize

The examples cluster around three recurring patterns. When extending an existing folder, follow the pattern already in use there:

1. **LangGraph state machines** (`agentic-workflows/*`, `langchain-basics/langchain-sql/`, `langchain-basics/langchain-context-chat/`)
   - A `TypedDict` `State` class defines the graph's shared state.
   - Each node is a plain function `def node(state: State) -> dict` returning a partial state update.
   - `StateGraph(State)` is wired with `add_node`, `add_edge`, and `add_conditional_edges`, then `.compile()` returns the runnable chain.
   - Structured LLM output uses Pydantic `BaseModel` + `llm.with_structured_output(Schema)`.
   - Two sub-patterns to know:
     - *Prompt chaining with a gate* — `joke-refinement-prompt-chaining` and `essay-refinement-prompt-chaining`: generate → conditional check → improve → polish.
     - *Generator/evaluator loop* — `generator-evaluator-workflows`: the evaluator routes back to the generator with feedback until approved.

2. **LangChain agents with tools** (`agentic-workflows/weather-document-summarizer/`, `agentic-workflows/essay-refinement-prompt-chaining/`)
   - Built with `create_tool_calling_agent(llm, tools, prompt)` + `AgentExecutor`.
   - Prompt is pulled from LangChain Hub: `hub.pull("hwchase17/openai-functions-agent")`.
   - Tools used: `TavilySearchResults` for web search, `create_retriever_tool` over a FAISS vector store for RAG-as-a-tool.

3. **MCP client/server** (`mcp-flight-booking/`)
   - `mcp-server.py` exposes tools via `FastMCP` over `streamable-http`, backed by MongoDB queries in `db_ops.py`.
   - `mcp-client.py` connects via `streamablehttp_client`, loads tools with `load_mcp_tools`, and wraps them in a `create_react_agent` driven by `ChatOllama`.

## Conventions baked into the existing code

- `init_chat_model("<model>", model_provider="ollama")` is the standard way models are instantiated — prefer this over `Ollama(...)` / `ChatOllama(...)` when adding to existing files (the older constructors appear only in `chess-simulation/` and `rag-pipeline/`).
- `warnings.filterwarnings("ignore", category=DeprecationWarning)` is intentionally set in several files to quiet noisy LangChain deprecation messages — keep it.
- Secrets are loaded with `load_dotenv()` and read via `os.environ.get(...)`. Do not hardcode API keys; recent commits explicitly moved away from that.
- `mcp-flight-booking/mcp-client.py` contains a hardcoded macOS path (`abs_path = "/Users/projjalgop/..."`) — it is currently unused but a leftover. Do not propagate this style.
