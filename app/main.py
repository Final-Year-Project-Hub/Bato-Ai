from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List

from app.core.config import settings
from app.schemas import RoadmapRequest, Roadmap
from app.ingestion.embedder import OfflineEmbedder
from qdrant_client import QdrantClient

from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.services.roadmap_service import RoadmapService

# Global state
class AppState:
    qdrant_client: QdrantClient = None
    embedder: OfflineEmbedder = None
    retriever: QdrantRetriever = None
    analyzer: QueryAnalyzer = None
    roadmap_service: RoadmapService = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Bato-Ai API...")
    state.qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    state.embedder = OfflineEmbedder()
    
    # Initialize Core Services
    state.retriever = QdrantRetriever(
        client=state.qdrant_client,
        collection=settings.QDRANT_COLLECTION,
        embedder=state.embedder
    )
    state.analyzer = QueryAnalyzer()
    state.roadmap_service = RoadmapService(
        retriever=state.retriever,
        analyzer=state.analyzer
    )
    
    yield
    # Shutdown
    print("🛑 Shutting down Bato-Ai API...")

app = FastAPI(
    title=settings.APP_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.post(f"{settings.API_V1_STR}/roadmap/generate", response_model=Roadmap)
async def generate_roadmap(request: RoadmapRequest):
    """
    Generate a learning roadmap based on user request.
    Uses DeepSeek and Qdrant RAG.
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
        
    try:
        roadmap = await state.roadmap_service.generate_roadmap(request)
        return roadmap
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from typing import Union
from app.schemas import ChatRequest, ClarificationRequest

@app.post(f"{settings.API_V1_STR}/chat", response_model=Union[Roadmap, ClarificationRequest])
async def chat_roadmap(request: ChatRequest):
    """
    Conversational endpoint.
    Returns either a Roadmap (if info is complete) or ClarificationRequest (if missing info).
    """
    if not state.roadmap_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
        
    try:
        response = await state.roadmap_service.process_chat(request)
        return response
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.API_V1_STR}/ingest")
async def trigger_ingestion():
    """
    Trigger knowledge base ingestion.
    """
    # In a real app, this should run in a background task
    from app.ingestion.ingest_qdrant import main as run_ingestion
    try:
        run_ingestion()
        return {"status": "Ingestion completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
