# Bato-AI 🤖

An intelligent AI-powered learning roadmap generator that creates personalized, structured learning paths based on official documentation.

## 🌟 Features

- **Personalized Roadmaps**: Generate custom learning paths based on your goals, experience level, and tech stack
- **Documentation-Driven**: Uses official framework documentation (Next.js, React, Python) via RAG (Retrieval-Augmented Generation)
- **Multi-Model LLM**: Hybrid approach using different models for query analysis, generation, and validation
- **Conversational Interface**: Natural language interaction with context-aware query understanding
- **Smart Caching**: Intelligent caching system for faster responses
- **Production-Ready**: Comprehensive error handling, observability, and health checks

## 🏗️ Architecture

```
Bato-AI/
├── app/                    # Main application code
│   ├── core/              # Core utilities (LLM, config, parsers)
│   ├── ingestion/         # Document ingestion pipeline
│   ├── retrieval/         # Vector search and query analysis
│   ├── services/          # Business logic (roadmap generation)
│   ├── schemas/           # Pydantic models
│   └── main.py            # FastAPI application
├── docs/                  # Framework documentation (Next.js, React, Python)
├── prompts/               # LLM prompt templates
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── frontend/              # Frontend application (separate)
└── backend/               # Backend services (separate)
```

### Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Vector Database**: Qdrant
- **LLM**: Hugging Face Inference API (Qwen models)
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Document Processing**: LangChain, BeautifulSoup

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- Hugging Face API token ([get one here](https://huggingface.co/settings/tokens))

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Bato-Ai
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env and add your HUGGINGFACE_API_TOKEN
   ```

5. **Start Qdrant**

   ```bash
   # On Windows:
   .\start_qdrant.ps1

   # On Linux/Mac:
   docker run -d -p 6333:6333 -p 6334:6334 \
     -v $(pwd)/qdrant_storage:/qdrant/storage \
     --name bato-qdrant qdrant/qdrant
   ```

6. **Download documentation** (optional, for first-time setup)

   ```bash
   python scripts/download_docs.py --list  # See available frameworks
   python scripts/download_docs.py nextjs react python  # Download specific frameworks
   ```

7. **Ingest documentation** (optional, for first-time setup)

   ```bash
   python -m app.ingestion.ingest_qdrant
   ```

8. **Start the server**
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## 📖 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Generate Roadmap (Chat Interface)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to learn React, I am a beginner"
  }'
```

### Direct Roadmap Generation

```bash
curl -X POST http://localhost:8000/api/v1/roadmap/generate \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Learn React",
    "intent": "learn",
    "proficiency": "beginner",
    "tech_stack": ["React", "JavaScript"]
  }'
```

### Get Metrics

```bash
curl http://localhost:8000/metrics
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Configuration
HUGGINGFACE_API_TOKEN=your_token_here
QUERY_ANALYSIS_MODEL=Qwen/Qwen2.5-7B-Instruct
GENERATION_MODEL=Qwen/Qwen2.5-72B-Instruct
VALIDATION_MODEL=Qwen/Qwen2.5-32B-Instruct

# Performance Tuning
LLM_TIMEOUT=120              # Timeout in seconds
LLM_MAX_TOKENS=3800          # Max tokens per generation
RETRIEVAL_MAX_CANDIDATES=40  # Max documents to retrieve

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=framework_docs
```

### Performance Optimization

For **faster generation** (< 60 seconds):

- Use `GENERATION_MODEL=Qwen/Qwen2.5-32B-Instruct` (faster than 72B)
- Set `LLM_TIMEOUT=90`
- Set `LLM_MAX_TOKENS=2500`
- Set `RETRIEVAL_MAX_CANDIDATES=30`

For **best quality** (may take longer):

- Use `GENERATION_MODEL=Qwen/Qwen2.5-72B-Instruct`
- Set `LLM_TIMEOUT=180`
- Set `LLM_MAX_TOKENS=3800`
- Set `RETRIEVAL_MAX_CANDIDATES=50`

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test
python tests/repro_roadmap_service.py

# Run with coverage
pytest --cov=app tests/
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

This will start:

- Bato-AI backend (port 8000)
- Qdrant vector database (port 6333)

### Building Docker Image

```bash
docker build -t bato-ai .
docker run -p 8000:8000 --env-file .env bato-ai
```

## 📊 Monitoring

### Health Checks

- **Basic**: `GET /health` - Quick health status
- **Deep**: `GET /health/deep` - Comprehensive component check
- **Metrics**: `GET /metrics` - Service metrics and statistics

### Metrics Tracked

- Request count
- Roadmap generation time (average)
- Cache hit rate
- Error rate
- LLM token usage

## 🛠️ Development

### Project Structure

- `app/core/`: Core utilities (LLM clients, configuration, parsers)
- `app/ingestion/`: Document loading and chunking
- `app/retrieval/`: Vector search and query analysis
- `app/services/`: Business logic (roadmap generation)
- `prompts/`: LLM prompt templates
- `scripts/`: Utility scripts for docs download and maintenance

### Adding New Frameworks

1. Edit `frameworks.yaml`:

   ```yaml
   django:
     name: "Django"
     key: "django"
     base_url: "https://docs.djangoproject.com"
     docs_path: "docs/django"
     git_repo: "https://github.com/django/django.git"
     git_branch: "main"
     git_sparse_paths: ["docs"]
   ```

2. Download documentation:

   ```bash
   python scripts/download_docs.py django
   ```

3. Ingest into Qdrant:
   ```bash
   python -m app.ingestion.ingest_qdrant
   ```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint
ruff check app/ tests/

# Type checking
mypy app/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of a final year project.

## 🙏 Acknowledgments

- Official documentation from Next.js, React, and Python communities
- Hugging Face for LLM inference
- Qdrant for vector search
- LangChain for document processing

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Note**: This is the AI-related code for the Bato project. Frontend and backend services are in separate directories.
