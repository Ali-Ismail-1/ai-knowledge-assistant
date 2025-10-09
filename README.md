# AI Knowledge Assistant

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, LangChain, and ChromaDB. This system provides intelligent question-answering over your documents with built-in guardrails, configurable LLM providers, and a clean API architecture.

---

## Features

- **🚀 FastAPI Backend** – High-performance async API with automatic OpenAPI docs
- **🧠 Multi-Provider LLM Support** – OpenAI and Ollama providers with easy extensibility
- **📚 Document Ingestion** – Load `.txt`, `.md`, and `.pdf` files into a vector store
- **🔍 Semantic Search** – Powered by ChromaDB and HuggingFace embeddings
- **🛡️ Guardrails** – Content filtering and safety prompts
- **💬 Streamlit UI** – Interactive chat interface for end users
- **⚙️ Configuration-Driven** – Environment-based settings with Pydantic
- **🧪 Tested** – pytest suite with coverage for core functionality
- **📊 Observability** – Structured logging and feedback collection

---

## Architecture

```
app/
├── api/                    # FastAPI routes and schemas
│   └── v1/                 # Versioned API endpoints
├── core/                   # Configuration, logging, dependencies
├── services/               # Business logic (RAG, LLM, vectorstore)
├── guardrails/             # Safety filters and prompts
├── monitoring/             # Logging and feedback
└── ui/                     # Streamlit frontend

data/
├── docs/                   # Source documents for ingestion
└── chroma/                 # Persisted vector database

scripts/                    # Maintenance and ingestion scripts
tests/                      # pytest test suite
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **(Optional)** Ollama installed locally, or an OpenAI API key

### 1. Clone and Install

```bash
git clone https://github.com/ali-ismail-1/ai-knowledge-assistant.git
cd ai-knowledge-assistant

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# LLM Provider: "openai" or "ollama"
LLM_PROVIDER=ollama

# OpenAI (if using)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Ollama (if using)
OLLAMA_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434

# Embeddings
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store
PERSIST_DIRECTORY=./data/chroma
DOC_DIR=./data/docs

# Retriever
RETRIEVER_K=5

# Environment
IS_PRODUCTION=false
ALLOW_RESET=true
```

### 3. Ingest Documents

Place your `.txt`, `.md`, or `.pdf` files in `data/docs/`, then run:

```bash
python scripts/build_vectorstore.py
```

This will:
- Recursively scan `data/docs/`
- Split documents into chunks
- Compute embeddings
- Persist to ChromaDB at `data/chroma/`

### 4. Start the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: **http://localhost:8000/docs**

### 5. Launch the Streamlit UI

```bash
streamlit run app/ui/streamlit_app.py
```

Chat interface available at: **http://localhost:8501**

---

## Usage

### API Endpoints

#### `POST /api/v1/chat`

Send a question and receive an answer with context.

**Request:**
```json
{
  "question": "What is the return policy?",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "answer": "The return policy allows returns within 30 days...",
  "sources": [
    {"content": "...", "metadata": {"source": "policies.pdf"}}
  ],
  "session_id": "user-123"
}
```

#### `GET /health`

Health check endpoint.

---

## Configuration

All settings are managed via `app/core/config.py` and can be overridden with environment variables or a `.env` file.

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider: `openai` or `ollama` |
| `EMBEDDINGS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RETRIEVER_K` | `5` | Number of chunks to retrieve per query |
| `PERSIST_DIRECTORY` | `./data/chroma` | ChromaDB persistence path |
| `DOC_DIR` | `./data/docs` | Source documents directory |
| `DEVICE` | Auto-detected | `cuda` or `cpu` for embeddings |

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_rag.py -v
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Type check (if using mypy)
mypy app/
```

### Project Commands (Makefile)

```bash
make install       # Install all dependencies
make test          # Run tests with coverage
make format        # Format with black
make lint          # Lint with ruff
make run           # Start FastAPI server
make ui            # Launch Streamlit app
make clean         # Remove cache and build artifacts
```

---

## Extending the System

### Adding a New LLM Provider

1. Update `app/services/llm_service.py`:
```python
elif settings.llm_provider == "anthropic":
    return ChatAnthropic(model=settings.anthropic_model, ...)
```

2. Add provider settings to `app/core/config.py`

3. Update `.env` template

### Adding Custom Guardrails

1. Define new filters in `app/guardrails/filters.py`
2. Apply in `app/services/rag_service.py` before/after LLM calls

### Using LangGraph Orchestration

Set `USE_LANGGRAPH=true` in your `.env` and implement graph logic in `app/services/graph_service.py`.

---

## Deployment

### Docker

```bash
docker build -t ai-knowledge-assistant .
docker run -p 8000:8000 --env-file .env ai-knowledge-assistant
```

### Production Checklist

- [ ] Set `IS_PRODUCTION=true`
- [ ] Use production-grade ASGI server (gunicorn with uvicorn workers)
- [ ] Configure HTTPS/TLS
- [ ] Set up monitoring and alerting
- [ ] Use managed vector store (e.g., hosted ChromaDB)
- [ ] Implement rate limiting
- [ ] Review and harden guardrails
- [ ] Set `ALLOW_RESET=false` to disable dangerous endpoints

---

## Troubleshooting

### GPU Support for Embeddings

If you have a CUDA-capable GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

The system will auto-detect and use GPU when available.

### Ollama Connection Issues

Ensure Ollama is running:

```bash
ollama serve
ollama pull llama3:8b
```

Check connectivity: `curl http://localhost:11434/api/tags`

### Empty or Slow Responses

- Verify documents were ingested: check `data/chroma/` for files
- Increase `RETRIEVER_K` for more context
- Check LLM provider logs for rate limits or errors

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes with clear messages
4. Add tests for new functionality
5. Run `make format lint test` before submitting
6. Open a Pull Request

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- **LangChain** – LLM orchestration framework
- **ChromaDB** – Vector database
- **FastAPI** – Modern Python web framework
- **Streamlit** – Rapid UI prototyping
- **HuggingFace** – Embedding models

---

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.
