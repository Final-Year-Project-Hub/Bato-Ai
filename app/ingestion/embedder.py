"""
Production-optimized embedder without caching.
Simplified to focus on embedding generation only.
"""

import logging
import time
from typing import List, Optional
from dataclasses import dataclass, replace
import numpy as np
import httpx  # Lightweight HTTP client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)  # Immutable for hashability
class EmbedderConfig:
    """Immutable configuration for embedder."""
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cuda"
    provider: str = "local"  # 'local' or 'api'
    api_token: Optional[str] = None  # Required for 'api' provider
    batch_size: int = 64
    normalize_embeddings: bool = True
    
    # Connection pooling for HF API
    max_concurrent_requests: int = 5
    request_timeout: int = 30  # Start fast, increase on retry


class BaseEmbedder:
    """Base interface for embedders."""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()
        self.embedding_dim = 384  # Default fallback
        self._stats = {"requests": 0, "errors": 0}
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embeddings = self.embed_documents([text])
        if embeddings:
            return embeddings[0]
        else:
            # Fallback for empty result
            return [0.0] * self.embedding_dim
        
    def get_stats(self) -> dict:
        return {
            **self._stats,
            "embedding_dim": self.embedding_dim,
            "model": self.config.model,
            "device": self.config.device,
            "provider": self.config.provider
        }


class APIEmbedder(BaseEmbedder):
    """
    Lightweight embedder using Hugging Face Inference API.
    Zero memory footprint for the model itself.
    """
    
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        if not self.config.api_token:
            raise ValueError("API token required for APIEmbedder")
            
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.config.model}"
        self.headers = {"Authorization": f"Bearer {self.config.api_token}"}
        
        # Determine dimension (hardcoded for common models to avoid initial network call)
        if "bge-small" in self.config.model:
            self.embedding_dim = 384
        elif "bge-base" in self.config.model:
            self.embedding_dim = 768
        elif "bge-large" in self.config.model:
            self.embedding_dim = 1024
            
        logger.info(f"✅ API Embedder initialized: {self.config.model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding via API with Diagnostic Logging (No Retries)."""
        if not texts:
            return []
        
        start = time.time()
        self._stats["requests"] += len(texts)
        
        logger.info(f"🔌 API Request: Embedding {texts} texts via {self.config.model}")
        
        try:
            # HF API has payload limits, might need chunking if huge batch
            response = httpx.post(
                self.api_url, 
                headers=self.headers, 
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=self.config.request_timeout
            )
            
            # Log specific details for debugging
            if response.status_code != 200:
                logger.error(f"❌ API Status: {response.status_code}")
                logger.error(f"❌ API Headers: {dict(response.headers)}")
                logger.error(f"❌ API Response: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            # Check for specific HF errors in 200 OK responses
            if isinstance(data, dict) and "error" in data:
                logger.error(f"❌ API returned error payload: {data}")
                raise ValueError(data["error"])

            # HF API returns list of lists (vectors) or list of list of lists (if 3D)
            embeddings = data
            
            # Check for 1D single vector response
            if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], float):
                embeddings = [embeddings]

            # Validation
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                if len(embeddings) > 0 and isinstance(embeddings[0], list):
                    duration = time.time() - start
                    logger.info(f"✅ Embedding success: {len(texts)} docs in {duration:.2f}s")
                    return embeddings
                elif len(embeddings) == 0:
                    return []
            
            logger.error(f"❌ Unexpected API response format: {type(embeddings)}")
            logger.error(f"❌ Raw Data Sample: {str(data)[:200]}")
            return [[0.0] * self.embedding_dim] * len(texts)
                    
        except httpx.TimeoutException:
            duration = time.time() - start
            logger.error(f"❌ API Timeout after {duration:.2f}s. Server did not respond in time.")
            self._stats["errors"] += len(texts)
            return [[0.0] * self.embedding_dim] * len(texts)
            
        except Exception as e:
            logger.error(f"❌ API Call failed: {str(e)}")
            self._stats["errors"] += len(texts)
            return [[0.0] * self.embedding_dim] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        return res[0] if res else [0.0] * self.embedding_dim


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Local embedder using sentence-transformers (Heavy memory usage).
    """
    
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        
        # Lazy import to avoid loading heavy libraries unless actually used
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("langchain-huggingface not installed. Use 'api' provider instead.")

        try:
            self.client = HuggingFaceEmbeddings(
                model_name=self.config.model,
                model_kwargs={"device": self.config.device},
                encode_kwargs={
                    "normalize_embeddings": self.config.normalize_embeddings,
                    "batch_size": self.config.batch_size
                }
            )
            logger.info(f"✅ Local Embedder loaded: {self.config.model}")
        except Exception as e:
            logger.warning(f"GPU unavailable, falling back to CPU: {e}")
            self.config = replace(self.config, device="cpu")
            self.client = HuggingFaceEmbeddings(
                model_name=self.config.model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": self.config.normalize_embeddings,
                    "batch_size": self.config.batch_size
                }
            )
        
        self.embedding_dim = self._detect_dim()

    def _detect_dim(self) -> int:
        try:
            sample = self.client.embed_documents(["test"])
            return len(sample[0])
        except:
            return 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents without caching."""
        if not texts:
            return []
        
        self._stats["requests"] += len(texts)
        
        try:
            embeddings = self.client.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            self._stats["errors"] += len(texts)
            return [[0.0] * self.embedding_dim] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query without caching."""
        self._stats["requests"] += 1
        
        try:
            embedding = self.client.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Local query embedding failed: {e}")
            self._stats["errors"] += 1
            return [0.0] * self.embedding_dim


# Factory function
def create_embedder(
    model: str = "BAAI/bge-small-en-v1.5",
    device: str = "cuda",
) -> BaseEmbedder:
    """
    Factory to create embedder based on Global Settings.
    """
    # Import inside to avoid circular deps if any, though config is safe
    from app.core.config import get_settings
    settings = get_settings()
    
    # Priority: Function Args > Settings
    # Actually for provider, we trust settings unless we want to force
    # FORCE LOCAL FOR NOW per user request
    # provider = "local"  # settings.EMBEDDING_PROVIDER
    provider = settings.EMBEDDING_PROVIDER
    config = EmbedderConfig(
        model=model,
        device=device,
        provider=provider,
        api_token=settings.get_hf_token(),
    )
    
    if provider == "api":
        return APIEmbedder(config)
    else:
        return HuggingFaceEmbedder(config)
