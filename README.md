# LangChain & Ollama AI Exploration

An experimental playground for building local AI applications and agentic workflows using **LangChain**, **LangGraph**, and **Ollama**, along with RAG, MCP tool calling, and lightweight LLM fine-tuning.

## Summary

Each folder contains self-contained examples organized by topic:

- **`langchain-basics/`** — Core LangChain & Ollama patterns including translation, stateful chat memory, structured extraction, sentiment classification, SQL and Neo4j Cypher Q&A, web summarization, and an Eel desktop chess simulation.
- **`langgraph/`** — State-machine workflows using prompt chaining and generator/evaluator loops for email, essay, and joke refinement.
- **`agentic-workflows/`** — Autonomous tool-calling agents and guardrails: customer order redressal, a DevOps incident router with domain specialists, and PII redaction/masking middleware.
- **`rag-pipeline/`** — Retrieval-Augmented Generation pipelines: PDF document search with ChromaDB and hybrid GraphRAG incident analysis combining Neo4j and ChromaDB.
- **`tool-calls/`** — External tool and protocol integrations, including a FastMCP flight-booking server/client backed by MongoDB and a Tavily + FAISS search agent.
- **`foundation-llm-training/`** — Fine-tuning and inference testing on `distilgpt2` using Hugging Face `transformers`.

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally
- Dependencies installed via `pip install -r requirements.txt` (some subfolders also provide their own `requirements.txt`)
- Example-specific services/keys where applicable: Neo4j, MongoDB, or `TAVILY_API_KEY` in a `.env` file

### Model Selection

Configured for a MacBook Pro (16 GB RAM) using lightweight local models under 6–7 GB:

- `llama3:8b` — General tasks, prompting, and basic RAG
- `mistral:7b` — Tool calling, structured output, and refinement chains
- `qwen3:8b` / `qwen2.5:7b-instruct` — Agent routing, Cypher/GraphRAG, and chess simulation
- `nomic-embed-text` — Local embeddings for GraphRAG

Run `ollama pull <model-name>` to download a model, then run any example from its folder (for example, `python app.py`).
