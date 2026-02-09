# RAG Document Assistant

A conversational AI assistant that answers questions based on uploaded documents using Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start (Docker)

```bash
# 1. Clone the repository
git clone https://github.com/ksingh-coder/doc-chat-assistant.git
cd doc-chat-assistant

# 2. Create .env file with your Groq API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start both backend and frontend
docker-compose up -d

# 4. Access the application
# - UI: http://localhost:8501
# - API: http://localhost:8000/docs
```

That's it! Both services are running in Docker containers with **CPU-only PyTorch** and **UV package manager** for lightning-fast builds (1-2s rebuilds vs minutes with pip).

## Features

- 📄 Support for multiple document formats (PDF, TXT, Markdown)
- 🔍 Semantic search using FAISS vector database
- 🤖 High-quality answers using Groq LLM (openai/gpt-oss-120b)
- 🚀 Fast and efficient embeddings with BGE-Large
- 🎨 Interactive Streamlit UI for easy document upload and querying
- � Document management with upload, list, and delete functionality
- �🔌 RESTful API with FastAPI
- 🐳 Docker deployment ready with UV (10-100x faster builds)
- 📊 Comprehensive logging
- 🔧 Modular and maintainable code structure
- ⚡ CPU-only PyTorch for universal compatibility
- 🔒 SELinux-compatible volume mounts

## Tech Stack

- **Python**: 3.12+
- **Backend**: FastAPI
- **Vector DB**: FAISS (local)
- **Embeddings**: BGE-Large (HuggingFace)
- **LLM**: Groq API (openai/gpt-oss-120b)
- **Framework**: LangChain
- **Deployment**: Docker with UV package manager
- **PyTorch**: 2.10.0+cpu (CPU-only, no CUDA)

## Project Structure

```
rag_kundan/
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   └── logging_config.py  # Logging setup
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── document_processor.py  # Document ingestion
│   │   ├── vectorstore.py         # FAISS management
│   │   └── rag_pipeline.py        # RAG logic
│   └── main.py                # FastAPI application
├── data/
│   ├── documents/             # Uploaded documents (mounted volume in Docker)
│   └── vectorstore/           # FAISS index & metadata
├── streamlit_app.py          # Streamlit UI
├── run.py                    # App launcher
├── requirements.txt          # Python dependencies (GPU support)
├── requirements_cpu.txt      # Python dependencies (CPU only)
├── Dockerfile                # Backend container
├── Dockerfile.streamlit      # Frontend container
├── docker-compose.yml        # Multi-container orchestration
└── .env                      # Environment variables
```


## Setup

### 1. Clone and Setup Environment

**Requirements**: Python 3.12 or higher

```bash
# Create virtual environment (Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# For CPU-only deployment (smaller, no GPU support)
pip install -r requirements_cpu.txt```

### 2. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your API key from: https://console.groq.com/
```

### 3. Run Locally

**Backend API:**
```bash
# Start the FastAPI server
python run.py

# Or with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend UI (in a new terminal):**
```bash
# Start the Streamlit UI
streamlit run streamlit_app.py

# Or specify port
streamlit run streamlit_app.py --server.port 8501
```

**Access the application:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- **Streamlit UI: http://localhost:8501** ⭐

> **Quick Test**: Once running, upload `sample_document.md` via the UI and ask "What is machine learning?"
> See [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for detailed UI usage instructions.

### 4. Run with Docker

Docker Compose will start both the backend API and frontend UI together.

```bash
# Build and start both services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f rag-api
docker-compose logs -f rag-ui

# Stop all services
docker-compose down
```

**Access the application:**
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

**Docker Features:**
- ⚡ **UV Package Manager**: Installs dependencies 10-100x faster than pip
- 💻 **CPU-Only PyTorch**: Works on any machine (no GPU required)
- 🔒 **SELinux Compatible**: Volume mounts work on RHEL/Fedora/CentOS
- 📦 **Compact Images**: ~2GB each (vs ~8GB with GPU support)
- 🚀 **Fast Rebuilds**: Code changes rebuild in 1-2 seconds
- 📝 **Container Logs**: Logs stored inside containers (not mounted)


## API Usage

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Upload Document

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

### Query Documents

```bash
# Or with just the question (uses defaults)
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

### Check Document Count

```bash
curl http://localhost:8000/api/v1/documents/count
```

### List All Documents

```bash
curl http://localhost:8000/api/v1/documents/list
```

### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/sample_document.md"
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Streamlit UI

The Streamlit UI provides an intuitive interface for interacting with the RAG system:

### Features:
- 📤 **Upload Documents**: Drag and drop PDF, TXT, or Markdown files
- 💬 **Query Interface**: Ask questions in natural language
- � **Manage Documents**: View all uploaded documents and delete them
- 📊 **Statistics**: See document count vs chunk count
- 🔧 **Adjustable Parameters**: Control retrieval count, temperature, and max tokens
- 📚 **Source Display**: View the source documents used to generate answers

### Using the UI:

1. **Upload Documents Tab**:
   - Select a file (PDF, TXT, or MD format)
   - Click "Upload" to process and add to the knowledge base
   - View confirmation with the number of chunks created

2. **Query Documents Tab**:
   - Enter your question in the text area
   - Adjust parameters in the sidebar (optional)
   - Click "Search" to get an answer
   - View the answer and expandable source documents

3. **Sidebar Settings**:
   - Monitor document and chunk counts separately
   - See total documents uploaded and total chunks created
   - Adjust number of sources to retrieve (1-10)
   - Enable custom LLM parameters for fine-tuning

4. **Manage Documents Tab**:
   - View all uploaded documents with their chunk counts
   - Delete documents you no longer need
   - See statistics: total documents, chunks, and average chunks per document
   - Deleted documents are removed from the vector store immediately

## Configuration

Key settings in `.env`:

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL`: LLM model to use (default: openai/gpt-oss-120b)
- `EMBEDDING_MODEL_NAME`: HuggingFace embedding model
- `EMBEDDING_DEVICE`: Auto-detects GPU/CPU (can override with 'cuda' or 'cpu')
- `CHUNK_SIZE`: Text chunk size for processing
- `CHUNK_OVERLAP`: Overlap between chunks
- `RETRIEVAL_K`: Number of documents to retrieve
- `TEMPERATURE`: LLM temperature (optional, default: 0.7)
- `MAX_TOKENS`: Max response tokens (optional, default: 1024)

## Dependencies

The project provides two requirements files:

- **`requirements.txt`**: Full dependency freeze from `uv pip freeze` (for reference)
- **`requirements_cpu.txt`**: CPU-only dependencies used by Docker

**CPU-Only Setup (Docker):**
- PyTorch installed from CPU index: `https://download.pytorch.org/whl/cpu`
- Results in PyTorch 2.10.0+cpu (no CUDA/GPU packages)
- Image size: ~2GB vs ~8GB with GPU support
- Works on any machine without GPU requirements

**UV Package Manager:**
- Docker uses UV for 10-100x faster dependency installation
- First build: ~5-10 minutes (downloads dependencies)
- Rebuilds with cache: 1-2 seconds
- Code-only changes: <20 seconds

## Development

### Running Tests

```bash
# Add tests in tests/ directory
pytest
```

### Code Structure

- **Document Processing**: Handles file uploads, parsing, and chunking
- **Vector Store**: Manages FAISS index and similarity search
- **RAG Pipeline**: Orchestrates retrieval and generation
- **API Routes**: FastAPI endpoints for user interaction

## Logging

**Local Development:**
- Logs stored in `logs/server.log` with rotation (max 10MB per file, 5 backups)
- Also printed to console

**Docker:**
- Logs stored inside containers (not mounted to host)
- View with: `docker logs rag-api` or `docker logs rag-ui`
- Real-time: `docker logs -f rag-api`

## Troubleshooting

### Docker Issues

**Permission denied errors (SELinux systems):**
The docker-compose.yml includes `:z` flag for SELinux compatibility. If you still see permission errors:
```bash
# Check if SELinux is enabled
getenforce

# If enabled, ensure volume mount has :z flag (already in docker-compose.yml)
volumes:
  - ./data:/app/data:z

# Restart containers
docker-compose down && docker-compose up -d
```

**Services won't start:**
```bash
# Check logs for errors
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose up --build
```

**Upload fails with 500 error:**
- Usually a permission issue with `/app/data/documents`
- Fixed by SELinux `:z` flag (already configured)
- Verify: `docker exec rag-api touch /app/data/documents/test.txt`

**Can't connect to API from UI:**
- Ensure both containers are on the same network
- Check `docker-compose ps` to verify both services are running
- Verify API is healthy: `curl http://localhost:8000/api/v1/health`

**Slow first startup:**
- First run downloads the BGE-Large model (~1.34GB)
- Subsequent starts are much faster
- Model is cached in the container

**Build is slow:**
- First build with UV: 5-10 minutes (downloading deps)
- Subsequent builds: 1-2 seconds (UV cache)
- If rebuilding from scratch, UV will re-download

**Verify CPU-only PyTorch:**
```bash
docker exec rag-api python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Should show: PyTorch: 2.10.0+cpu, CUDA: False
```

**Out of memory:**
- Docker Desktop: Increase memory limit in settings
- Linux: Check available system memory
- Consider using a smaller embedding model

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Groq API Errors**: Verify your API key in `.env`

3. **Memory Issues**: Reduce `CHUNK_SIZE` or use smaller embedding models

4. **FAISS Not Loading**: Delete `data/vectorstore/` and re-upload documents
