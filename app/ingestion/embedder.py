"""
Production-optimized embedder with async support, connection pooling, and API fallback.
"""

import asyncio
import hashlib
import logging
from typing import List, Optional, Dict, Union
from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
import httpx # Lightweight HTTP client

# Removed top-level import to prevent eager loading of heavy libraries
# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)  # Immutable for hashability
class EmbedderConfig:
    """Immutable configuration for embedder."""
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cuda"
    provider: str = "local" # 'local' or 'api'
    api_token: Optional[str] = None # Required for 'api' provider
    batch_size: int = 64
    cache_enabled: bool = True
    cache_size: int = 10000
    normalize_embeddings: bool = True
    
    # Connection pooling for HF API
    max_concurrent_requests: int = 5
    request_timeout: int = 30


class BaseEmbedder:
    """Base interface for embedders."""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()
        self.embedding_dim = 384 # Default fallback
        self._cache: Dict[str, np.ndarray] = {}
        self._stats = {"hits": 0, "misses": 0, "errors": 0, "total_time": 0.0}
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError
        
    def get_stats(self) -> Dict:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total) if total > 0 else 0.0
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "hit_rate": hit_rate,
            "embedding_dim": self.embedding_dim,
            "model": self.config.model,
            "device": self.config.device,
            "provider": self.config.provider
        }

    @staticmethod
    @lru_cache(maxsize=100000)
    def _get_cache_key(text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _check_cache(self, text: str) -> Optional[List[float]]:
        if not self.config.cache_enabled:
            return None
        key = self._get_cache_key(text)
        if key in self._cache:
            self._stats["hits"] += 1
            return self._cache[key].tolist()
        return None

    def _add_to_cache(self, text: str, embedding: List[float]):
        if not self.config.cache_enabled:
            return
        key = self._get_cache_key(text)
        if len(self._cache) >= self.config.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = np.array(embedding, dtype=np.float32)

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Embedding cache cleared")


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
        """Batch embedding via API."""
        if not texts: return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
             cached = self._check_cache(text)
             if cached:
                 results.append((i, cached))
             else:
                 uncached_texts.append(text)
                 uncached_indices.append(i)
                 self._stats["misses"] += 1
        
        if uncached_texts:
            try:
                # HF API has payload limits, might need chunking if huge batch
                response = httpx.post(
                    self.api_url, 
                    headers=self.headers, 
                    json={"inputs": uncached_texts, "options": {"wait_for_model": True}},
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                data = response.json()
                
                # HF API returns list of lists (vectors) or list of list of lists (if 3D)
                # Ensure we have flat list of vectors
                embeddings = data
                
                # Check for 1D single vector response (common when input list size is 1)
                if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], float):
                    embeddings = [embeddings]

                # Validation
                if isinstance(embeddings, list) and len(embeddings) == len(uncached_texts):
                    if len(embeddings) > 0 and isinstance(embeddings[0], list):
                         pass # Valid
                    elif len(embeddings) == 0:
                         pass # Empty
                    else:
                         logger.error(f"Unexpected API response format: {type(embeddings)}")
                         embeddings = [[0.0] * self.embedding_dim] * len(uncached_texts)

                for idx, embedding in zip(uncached_indices, embeddings):
                    self._add_to_cache(uncached_texts[uncached_indices.index(idx)], embedding)
                    results.append((idx, embedding))
                    
            except Exception as e:
                logger.error(f"API Batch embedding failed: {e}")
                self._stats["errors"] += 1
                for idx in uncached_indices:
                    results.append((idx, [0.0] * self.embedding_dim))
        
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

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
            self.config = dataclass.replace(self.config, device="cpu")
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
        # Same cache logic, just calling self.client
        if not texts: return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
             cached = self._check_cache(text)
             if cached:
                 results.append((i, cached))
             else:
                 uncached_texts.append(text)
                 uncached_indices.append(i)
                 self._stats["misses"] += 1
                 
        if uncached_texts:
            try:
                embeddings = self.client.embed_documents(uncached_texts)
                for idx, embedding in zip(uncached_indices, embeddings):
                    self._add_to_cache(uncached_texts[uncached_indices.index(idx)], embedding)
                    results.append((idx, embedding))
            except Exception as e:
                logger.error(f"Local embedding failed: {e}")
                self._stats["errors"] += 1
                for idx in uncached_indices:
                    results.append((idx, [0.0] * self.embedding_dim))
        
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def embed_query(self, text: str) -> List[float]:
        # Simple wrapper with cache
        cached = self._check_cache(text)
        if cached: return cached
        
        try:
            embedding = self.client.embed_query(text)
            self._add_to_cache(text, embedding)
            self._stats["misses"] += 1
            return embedding
        except Exception as e:
            logger.error(f"Local query embedding failed: {e}")
            return [0.0] * self.embedding_dim


# Factory function
def create_embedder(
    model: str = "BAAI/bge-small-en-v1.5",
    device: str = "cuda",
    cache_enabled: bool = True
) -> BaseEmbedder:
    """
    Factory to create embedder based on Global Settings.
    """
    # Import inside to avoid circular deps if any, though config is safe
    from app.core.config import get_settings
    settings = get_settings()
    
    # Priority: Function Args > Settings
    # Actually for provider, we trust settings unless we want to force
    provider = settings.EMBEDDING_PROVIDER
    
    config = EmbedderConfig(
        model=model,
        device=device,
        provider=provider,
        api_token=settings.get_hf_token(),
        cache_enabled=cache_enabled
    )
    
    if provider == "api":
        return APIEmbedder(config)
    else:
        return HuggingFaceEmbedder(config)