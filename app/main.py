"""
Production FastAPI application with observability and security.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Union, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.core.multi_llm import MultiModelLLMManager, MultiModelConfig
from app.ingestion.embedder import BaseEmbedder
from app.ingestion.ingest_qdrant import ingest_framework_docs
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.retriever import QdrantRetriever, RetrievalConfig
from app.retrieval.token_budget import TokenBudgetPlanner
from app.schemas import RoadmapRequest, Roadmap, ChatRequest, ClarificationRequest
from app.services.roadmap_service import RoadmapService

# Get settings
settings = get_settings()

# Setup centralized logging
setup_logging(
    environment=settings.ENV,
    log_level=settings.LOG_LEVEL,
    log_dir=Path("logs") if settings.ENV == "production" else None
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Security
# ============================================================================

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key if authentication is enabled."""
    if settings.is_production:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        # Constant time comparison to prevent timing attacks
        import secrets
        if not secrets.compare_digest(api_key, settings.ADMIN_API_KEY.get_secret_value()):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
            
    return api_key


# ============================================================================
# Rate Limiting
# ============================================================================

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_REQUESTS}/minute"]
) if settings.RATE_LIMIT_ENABLED else None


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
    
    def is_healthy(self) -> bool:
        """Check if all components are initialized."""
        return (
            self.initialized and
            self.qdrant_client is not None and
            self.embedder is not None and
            self.retriever is not None and
            self.analyzer is not None and
            self.roadmap_service is not None
        )

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
        
        # Verify connection
        collections = await asyncio.to_thread(
            state.qdrant_client.get_collections
        )
        logger.info(f"✅ Qdrant connected: {len(collections.collections)} collections")
        
        # 2. Initialize Embedder
        logger.info("Loading embedder...")
        
        # FAILSAFE: Force API provider if in production and on CPU to prevent OOM
        # This handles cases where render.yaml env vars might not have synced yet
        embedding_provider = settings.EMBEDDING_PROVIDER
        if settings.is_production and settings.EMBEDDING_DEVICE == "cpu":
            if embedding_provider != "api":
                logger.warning("⚠️ Production CPU detected: Forcing API embedder to prevent OOM")
                embedding_provider = "api"

        # Use factory to support API fallback based on env vars
        from app.ingestion.embedder import create_embedder, EmbedderConfig
        
        # We manually create config to override the provider if needed
        # (create_embedder reads settings by default if we just passed args, 
        # but better to be explicit here to ensure our override works)
        
        # Actually create_embedder factory reads provider from settings.
        # So we should modify the settings object or pass explicit config.
        # create_embedder doesn't accept provider as arg in my previous implementation? 
        # Let's check embedder.py... Yes it does not take provider arg in create_embedder!
        # It reads from settings.EMBEDDING_PROVIDER.
        
        # So I must patch settings OR instantiate APIEmbedder directly.
        # Let's instantiate explicitly based on our logic.
        
        from app.ingestion.embedder import APIEmbedder, HuggingFaceEmbedder, EmbedderConfig
        
        embedder_config = EmbedderConfig(
            model=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            provider=embedding_provider,
            api_token=settings.get_hf_token(),
            cache_enabled=True
        )
        
        if embedding_provider == "api":
            state.embedder = APIEmbedder(embedder_config)
        else:
            state.embedder = HuggingFaceEmbedder(embedder_config)
            
        logger.info(f"✅ Embedder ready: {state.embedder.config.model} ({state.embedder.config.provider})")
        
        # 3. Initialize LLMs
        logger.info("Initializing LLM models...")
        
        multi_model_config = MultiModelConfig(
            api_token=settings.HUGGINGFACE_API_TOKEN.get_secret_value(),
            query_analysis_model=settings.QUERY_ANALYSIS_MODEL,
            generation_model=settings.GENERATION_MODEL,
            generation_temperature=settings.LLM_TEMPERATURE,
            generation_max_tokens=settings.LLM_MAX_TOKENS,
            validation_model=settings.VALIDATION_MODEL,
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
            cache_enabled=True,
            cache_ttl_seconds=3600,
            time_decay_enabled=True
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
        
        # Start background tasks
        if settings.is_production:
            # Cleanup task
            asyncio.create_task(_background_cleanup())
        
        yield
        
        # Shutdown
        logger.info("🛑 Shutting down Bato-AI API...")
        state.initialized = False
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


async def _background_cleanup():
    """Background cleanup task."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        
        try:
            # Cleanup expired sessions
            if state.analyzer:
                state.analyzer.session_store.cleanup_expired()
            
            # Log metrics
            if state.roadmap_service:
                metrics = state.roadmap_service.get_metrics()
                logger.info(f"Metrics: {metrics['requests']} requests, "
                           f"{metrics['roadmaps_generated']} roadmaps")
        
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Bato-AI API",
    description="AI-powered learning roadmap generation",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json" if not settings.is_production else None,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
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
    expose_headers=["X-Request-ID"]
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted host (production only)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.onrender.com", "localhost", "127.0.0.1"]
    )

# Rate limiting
if limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging."""
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None
        }
    )
    
    # Don't expose internal errors in production
    if settings.is_production:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)}
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler with structured logging."""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code
        }
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# ============================================================================
# Middleware - Request Logging
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    
    # Add request ID
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    
    response = await call_next(request)
    
    # Add request ID to response
    response.headers["X-Request-ID"] = request_id
    
    # Log
    duration = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {duration:.0f}ms",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration
        }
    )
    
    return response


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Welcome point."""
    return {"message": "Welcome to our AI system"}


@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok" if state.is_healthy() else "starting",
        "version": "1.0.0",
        "uptime_seconds": time.time() - state.startup_time if state.startup_time else 0
    }


@app.get("/health/deep")
async def deep_health_check():
    """Comprehensive health check."""
    health = {
        "status": "healthy",
        "components": {},
        "uptime_seconds": time.time() - state.startup_time if state.startup_time else 0
    }
    
    # Check Qdrant
    try:
        if state.qdrant_client:
            await asyncio.to_thread(state.qdrant_client.get_collections)
            health["components"]["qdrant"] = "healthy"
        else:
            health["components"]["qdrant"] = "not_initialized"
            health["status"] = "degraded"
    except Exception as e:
        health["components"]["qdrant"] = f"error: {str(e)}"
        health["status"] = "unhealthy"
    
    # Check LLM
    health["components"]["llm"] = (
        "healthy" if state.multi_model_manager else "not_initialized"
    )
    
    # Check Embedder
    health["components"]["embedder"] = (
        "healthy" if state.embedder else "not_initialized"
    )
    
    # Check Service
    health["components"]["roadmap_service"] = (
        "healthy" if state.roadmap_service else "not_initialized"
    )
    
    return health


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return state.roadmap_service.get_metrics()


# ============================================================================
# API Endpoints
# ============================================================================

@app.post(
    "/api/v1/chat",
    response_model=Union[Roadmap, ClarificationRequest],
    dependencies=[Depends(verify_api_key)] if settings.is_production else []
)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/minute") if limiter else lambda x: x
async def chat_roadmap(
    body: ChatRequest,
    request: Request
):
    """
    Conversational roadmap generation endpoint.
    
    Returns either a Roadmap or ClarificationRequest.
    """
    # Validate
    if not body.message or len(body.message.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Message must be at least 3 characters"
        )
    
    if len(body.message) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Message too long (max 5000 characters)"
        )
    
    # Check service
    if not state.roadmap_service:
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )
    
    try:
        # Convert user_context to dict if provided
        user_context_dict = None
        if body.user_context:
            user_context_dict = body.user_context.dict(exclude_none=True)
        
        response = await state.roadmap_service.process_chat(
            user_message=body.message,
            conversation_history=body.conversation_history,
            user_context=user_context_dict,  # NEW
            strict_mode=body.strict_mode # NEW
        )
        return response
    
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timeout - please try again"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error" if settings.is_production else str(e)
        )


from fastapi.responses import StreamingResponse

@app.post("/api/v1/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request
):
    """
    Streaming chat endpoint (Server-Sent Events).
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    return StreamingResponse(
        state.roadmap_service.process_chat_stream(
            user_message=body.message,
            conversation_history=body.conversation_history,
            user_context=body.user_context,
            strict_mode=body.strict_mode
        ),
        media_type="text/event-stream"
    )


@app.post(
    "/api/v1/roadmap/generate",
    response_model=Roadmap,
    dependencies=[Depends(verify_api_key)] if settings.is_production else []
)
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
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    framework: str = "all",
    api_key: str = Depends(verify_api_key) if settings.is_production else None
):
    """Trigger documentation ingestion."""
    targets = {}
    if framework == "all":
        targets = settings.FRAMEWORKS
    elif framework in settings.FRAMEWORKS:
        targets = {framework: settings.FRAMEWORKS[framework]}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown framework: {framework}"
        )
    
    # Failsafe: Force API provider if in production and on CPU
    embedding_provider = settings.EMBEDDING_PROVIDER
    if settings.is_production and settings.EMBEDDING_DEVICE == "cpu":
        if embedding_provider != "api":
            embedding_provider = "api"

    for key, config in targets.items():
        background_tasks.add_task(
            ingest_framework_docs,
            framework=key,
            docs_path=config["path"],
            collection_name=config["collection"],
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_provider=embedding_provider,
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
# Startup Event
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS if settings.is_production else 1,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )