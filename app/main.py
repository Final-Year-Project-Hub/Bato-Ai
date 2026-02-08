"""
Simplified FastAPI application for college project.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Union, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.core.multi_llm import MultiModelLLMManager, MultiModelConfig
from app.ingestion.embedder import BaseEmbedder
from app.ingestion.ingest_qdrant import ingest_framework_docs
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.retriever import QdrantRetriever, RetrievalConfig
from app.retrieval.token_budget import TokenBudgetPlanner
from app.schemas import RoadmapRequest, Roadmap, ChatRequest, ClarificationRequest, TopicDetail
from app.services.roadmap_service import RoadmapService
from app.routers import documents

# Get settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedder: Optional[BaseEmbedder] = None
        self.retriever: Optional[QdrantRetriever] = None
        self.analyzer: Optional[QueryAnalyzer] = None
        self.roadmap_service: Optional[RoadmapService] = None
        self.multi_model_manager: Optional[MultiModelLLMManager] = None
        self.token_planner: Optional[TokenBudgetPlanner] = None
        self.initialized: bool = False
        self.startup_time: Optional[float] = None

state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with graceful startup/shutdown."""
    startup_start = time.time()
    logger.info("🚀 Starting Bato-AI API...")
    
    try:
        # 1. Initialize Qdrant
        logger.info("Connecting to Qdrant...")
        state.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.get_qdrant_key(),
            timeout=settings.QDRANT_TIMEOUT,
            prefer_grpc=settings.QDRANT_PREFER_GRPC
        )
        
        collections = await asyncio.to_thread(state.qdrant_client.get_collections)
        logger.info(f"✅ Qdrant connected: {len(collections.collections)} collections")
        
        # 2. Initialize Embedder
        logger.info("Loading embedder...")
        from app.ingestion.embedder import APIEmbedder, HuggingFaceEmbedder, EmbedderConfig
        
        embedder_config = EmbedderConfig(
            model=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            provider=settings.EMBEDDING_PROVIDER,
            api_token=settings.get_hf_token(),
        )
        
        if settings.EMBEDDING_PROVIDER == "api":
            state.embedder = APIEmbedder(embedder_config)
        else:
            state.embedder = HuggingFaceEmbedder(embedder_config)
            
        logger.info(f"✅ Embedder ready: {state.embedder.config.model}")
        
        # 3. Initialize LLMs
        logger.info("Initializing LLM models...")
        multi_model_config = MultiModelConfig(
            api_token=settings.HUGGINGFACE_API_TOKEN.get_secret_value(),
            query_analysis_model=settings.QUERY_ANALYSIS_MODEL,
            generation_model=settings.GENERATION_MODEL,
            generation_temperature=settings.LLM_TEMPERATURE,
            generation_max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT
        )
        state.multi_model_manager = MultiModelLLMManager(multi_model_config)
        logger.info("✅ Multi-model LLM initialized")
        
        # 4. Initialize Services
        logger.info("Initializing core services...")
        state.token_planner = TokenBudgetPlanner(
            model_max_tokens=settings.TOKEN_BUDGET_MAX,
            conservative=settings.TOKEN_BUDGET_CONSERVATIVE
        )
        
        retrieval_config = RetrievalConfig(
            timeout_seconds=30
        )
        state.retriever = QdrantRetriever(
            client=state.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedder=state.embedder,
            config=retrieval_config
        )
        
        state.analyzer = QueryAnalyzer(
            llm=state.multi_model_manager.get_query_analyzer_llm(),
            token_planner=state.token_planner
        )
        
        state.roadmap_service = RoadmapService(
            retriever=state.retriever,
            analyzer=state.analyzer,
            llm_manager=state.multi_model_manager,
            token_planner=state.token_planner,
            framework_patterns=settings.FRAMEWORK_URL_PATTERNS
        )
        
        # Mark as initialized
        state.initialized = True
        state.startup_time = time.time() - startup_start
        logger.info(f"✅ Bato-AI API ready in {state.startup_time:.2f}s")
        
        yield
        
        # Shutdown
        logger.info("🛑 Shutting down Bato-AI API...")
        state.initialized = False
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Bato-AI API",
    description="AI-powered learning roadmap generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ============================================================================
# Middleware
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# ============================================================================
# Request Logging Middleware
# ============================================================================

@app.middleware("http")
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.0f}ms")
        return response
    except RuntimeError as e:
        # Starlette raises RuntimeError("No response returned.") on client disconnect
        if "No response returned" in str(e):
            duration = (time.time() - start_time) * 1000
            logger.warning(f"{request.method} {request.url.path} - Client Disconnected - {duration:.0f}ms")
            # Return empty response to satisfy ASGI
            return JSONResponse(status_code=499, content={"detail": "Client Closed Request"})
        raise e


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Welcome endpoint."""
    return {"message": "Welcome to Bato-AI - Your AI Learning Roadmap Assistant"}


@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok" if state.initialized else "starting",
        "version": "1.0.0",
        "uptime_seconds": time.time() - state.startup_time if state.startup_time else 0
    }


# ============================================================================
# Include Routers
# ============================================================================

app.include_router(documents.router)


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v1/chat", response_model=Union[Roadmap, ClarificationRequest])
async def chat_roadmap(body: ChatRequest, request: Request):
    """
    Conversational roadmap generation endpoint.
    Returns either a Roadmap or ClarificationRequest.
    """
    # Validate
    if not body.message or len(body.message.strip()) < 3:
        raise HTTPException(status_code=400, detail="Message must be at least 3 characters")
    
    if len(body.message) > 5000:
        raise HTTPException(status_code=400, detail="Message too long (max 5000 characters)")
    
    # Check service
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Convert user_context to dict if provided
        user_context_dict = (
            body.user_context.model_dump(exclude_none=True)
            if body.user_context
            else None
        )
        
        response = await state.roadmap_service.process_chat(
            user_message=body.message,
            conversation_history=body.conversation_history,
            user_context=user_context_dict,
            strict_mode=body.strict_mode
        )
        return response
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout - please try again")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """
    Streaming chat endpoint (Server-Sent Events).
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    return StreamingResponse(
        state.roadmap_service.process_chat_stream(
            user_message=body.message,
            conversation_history=body.conversation_history,
            user_context=body.user_context.model_dump(exclude_none=True) if body.user_context else None,
            strict_mode=body.strict_mode
        ),
        media_type="text/event-stream"
    )


@app.post("/api/v1/roadmap/generate", response_model=Roadmap)
async def generate_roadmap(request: RoadmapRequest):
    """Direct roadmap generation (no conversation)."""
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        roadmap = await state.roadmap_service.generate_roadmap(
            goal=request.goal,
            intent=request.intent,
            proficiency=request.proficiency,
            strict_mode=request.strict_mode
        )
        return roadmap
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/topic/{phase_number}/{topic_title}", response_model=TopicDetail)
async def get_topic_detail(
    phase_number: int,
    topic_title: str,
    phase_title: str = Query(..., description="Title of the phase this topic belongs to"),
    goal: str = Query(..., description="Original roadmap goal (e.g., 'Learn React')")
):
    """
    Get detailed "deep-dive" content for a specific topic.
    Generates comprehensive guide with code snippets and examples.
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    try:
        detail = await state.roadmap_service.get_topic_detail(
            goal=goal,
            phase_number=phase_number,
            phase_title=phase_title,
            topic_title=topic_title
        )
        return detail
    
    except Exception as e:
        logger.error(f"Topic detail error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/topic/stream/{phase_number}/{topic_title}")
async def stream_topic_detail(
    phase_number: int,
    topic_title: str,
    phase_title: str = Query(..., description="Title of the phase this topic belongs to"),
    goal: str = Query(..., description="Original roadmap goal")
):
    """
    Stream detailed topic content as Server-Sent Events (SSE).
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    return StreamingResponse(
        state.roadmap_service.stream_topic_detail(
            goal=goal,
            phase_number=phase_number,
            phase_title=phase_title,
            topic_title=topic_title
        ),
        media_type="text/event-stream"
    )


@app.post("/api/v1/quiz/generate")
async def generate_quiz(request: dict):
    """
    Generate a quiz for a specific topic.
    Expects: goal, phase_title, topic_title, topic_content
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Extract required fields
        goal = request.get("goal")
        phase_title = request.get("phase_title")
        topic_title = request.get("topic_title")
        topic_content = request.get("topic_content")
        
        if not all([goal, phase_title, topic_title, topic_content]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: goal, phase_title, topic_title, topic_content"
            )
        
        # Generate quiz
        quiz = await state.roadmap_service.generate_quiz(
            goal=goal,
            phase_title=phase_title,
            topic_title=topic_title,
            topic_content=topic_content
        )
        
        return quiz
    
    except Exception as e:
        logger.error(f"Quiz generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    framework: str = "all"
):
    """Trigger documentation ingestion."""
    targets = {}
    if framework == "all":
        targets = settings.FRAMEWORKS
    elif framework in settings.FRAMEWORKS:
        targets = {framework: settings.FRAMEWORKS[framework]}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {framework}")
    
    for key, config in targets.items():
        background_tasks.add_task(
            ingest_framework_docs,
            framework=key,
            docs_path=config["path"],
            collection_name=config["collection"],
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_provider=settings.EMBEDDING_PROVIDER,
            embedding_device=settings.EMBEDDING_DEVICE,
            qdrant_url=settings.QDRANT_URL,
            recreate=settings.RECREATE_COLLECTION
        )
    
    return {
        "status": "accepted",
        "frameworks": list(targets.keys()),
        "message": "Ingestion started in background"
    }


# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Always reload for development
        log_level="info"
    )
