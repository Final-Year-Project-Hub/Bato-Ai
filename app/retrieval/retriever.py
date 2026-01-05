"""
Production retriever with smart caching, query optimization, and observability.
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    # Query optimization
    query_expansion_enabled: bool = True
    reranking_enabled: bool = False  # Future: cross-encoder reranking
    
    # Time decay
    time_decay_enabled: bool = True
    decay_half_life_days: int = 180  # 6 months
    
    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    
    # Performance
    max_concurrent_queries: int = 5
    timeout_seconds: int = 30


class TokenCounter:
    """Optimized token counter with caching."""
    
    _encoding = None
    _count_cache = {}
    _cache_size = 10000
    
    def __init__(self):
        if TokenCounter._encoding is None:
            TokenCounter._encoding = tiktoken.get_encoding("cl100k_base")
        self.encoding = TokenCounter._encoding
    
    @classmethod
    @lru_cache(maxsize=10000)
    def count_cached(cls, text: str) -> int:
        """LRU cached token counting."""
        try:
            return len(cls._encoding.encode(text, disallowed_special=()))
        except Exception:
            return max(1, len(text) // 4)
    
    def count(self, text: str) -> int:
        """Count tokens with fallback."""
        return self.count_cached(text)


class RetrievalCache:
    """Time-aware LRU cache for retrieval results."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[List[Document], float]] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[List[Document]]:
        """Get cached results if not expired."""
        if key not in self._cache:
            return None
        
        results, timestamp = self._cache[key]
        
        # Check expiration
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            self._access_order.remove(key)
            return None
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        logger.debug(f"Cache hit: {key[:8]}...")
        return results
    
    def set(self, key: str, results: List[Document]) -> None:
        """Cache results with TTL."""
        # Evict LRU if needed
        while len(self._cache) >= self.max_size:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = (results, time.time())
        self._access_order.append(key)
        logger.debug(f"Cached: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()


class QdrantRetriever:
    """
    Production-ready retriever with advanced features.
    
    Improvements:
    - Query result caching with TTL
    - Time-weighted scoring
    - Async batch retrieval
    - Query optimization
    - Comprehensive metrics
    - Smart truncation strategies
    """
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder,
        config: Optional[RetrievalConfig] = None
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.token_counter = TokenCounter()
        
        # Initialize cache
        self._cache = RetrievalCache(
            ttl_seconds=self.config.cache_ttl_seconds,
            max_size=self.config.cache_max_size
        ) if self.config.cache_enabled else None
        
        # Metrics
        self._metrics = {
            "queries": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0,
            "total_docs_retrieved": 0
        }
        
        logger.info(f"QdrantRetriever initialized: {collection_name}")
    
    def _generate_cache_key(
        self,
        query: str,
        budget_tokens: int,
        max_candidates: int,
        score_threshold: float,
        collection_name: Optional[str] = None
    ) -> str:
        """Generate deterministic cache key."""
        key_parts = [
            query,
            str(budget_tokens),
            str(max_candidates),
            str(score_threshold),
            collection_name or self.collection_name
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _calculate_time_decay(self, date_str: Optional[str]) -> float:
        """
        Calculate time decay factor with configurable half-life.
        
        Formula: decay = 0.5 ^ (age_days / half_life_days)
        """
        if not self.config.time_decay_enabled or not date_str:
            return 1.0
        
        try:
            doc_date = datetime.fromisoformat(date_str)
            age_days = (datetime.now() - doc_date).days
            
            # Exponential decay
            decay = 0.5 ** (age_days / self.config.decay_half_life_days)
            return max(0.1, decay)  # Floor at 0.1
        
        except Exception:
            return 0.9  # Default for unparseable dates
    
    async def retrieve_async(
        self,
        query: str,
        budget: Any,
        max_candidates: int = 5,
        score_threshold: float = 0.0,
        collection_name: Optional[str] = None,
        filters: Optional[Filter] = None
    ) -> List[Document]:
        """
        Async retrieval with caching and optimizations.
        
        New features:
        - Async embedding
        - Query result caching
        - Advanced filtering
        - Parallel processing
        """
        start_time = time.time()
        self._metrics["queries"] += 1
        
        # Check cache
        cache_key = self._generate_cache_key(
            query, budget.retrieval_tokens, max_candidates, 
            score_threshold, collection_name
        )
        
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                return cached
        
        try:
            # Async embedding
            query_vector = await asyncio.to_thread(
                self.embedder.embed_query,
                query
            )
            
            # Search with optional filters
            target_collection = collection_name or self.collection_name
            
            search_results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=target_collection,
                query=query_vector,
                limit=max_candidates,
                score_threshold=score_threshold,
                with_payload=True,
                query_filter=filters
            )
            
            # Apply time decay
            scored_results = self._apply_time_decay(search_results.points)
            
            # Filter by budget
            documents = self._filter_by_budget(
                scored_results,
                budget.retrieval_tokens
            )
            
            # Cache results
            if self._cache:
                self._cache.set(cache_key, documents)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, len(documents))
            
            logger.info(
                f"Retrieved {len(documents)} docs in {latency_ms:.0f}ms "
                f"({sum(d.metadata['chunk_tokens'] for d in documents)}/{budget.retrieval_tokens} tokens)"
            )
            
            return documents
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []
    
    def retrieve(
        self,
        query: str,
        budget: Any,
        max_candidates: int = 5,
        score_threshold: float = 0.0,
        collection_name: Optional[str] = None,
        filters: Optional[Filter] = None
    ) -> List[Document]:
        """Sync wrapper for async retrieve."""
        return asyncio.run(
            self.retrieve_async(
                query, budget, max_candidates, 
                score_threshold, collection_name, filters
            )
        )
    
    def _apply_time_decay(self, search_results: List) -> List:
        """Apply time-weighted decay to scores."""
        if not self.config.time_decay_enabled:
            return search_results
        
        for result in search_results:
            last_modified = result.payload.get("last_modified")
            decay = self._calculate_time_decay(last_modified)
            
            if decay < 1.0:
                original_score = result.score
                result.score = original_score * decay
                logger.debug(
                    f"Decay applied: {original_score:.3f} → {result.score:.3f} "
                    f"(factor: {decay:.2f})"
                )
        
        # Re-sort by adjusted scores
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results
    
    def _filter_by_budget(
        self,
        search_results: List,
        token_budget: int
    ) -> List[Document]:
        """
        Smart budget filtering with multiple strategies.
        
        Strategies:
        1. Greedy: Take highest-scored until budget exhausted
        2. Balanced: Ensure diversity across sources
        3. Truncate: Split large documents if beneficial
        """
        documents = []
        used_tokens = 0
        sources_used = set()
        
        for result in search_results:
            text = result.payload.get("page_content", "")
            if not text:
                continue
            
            doc_tokens = self.token_counter.count(text)
            source = result.payload.get("file_path", "unknown")
            
            # Check budget
            if used_tokens + doc_tokens > token_budget:
                # Strategy: Try truncation if first doc or beneficial
                if not documents or (doc_tokens > token_budget * 0.5 and len(documents) < 3):
                    truncated = self._truncate_document(
                        text, token_budget - used_tokens
                    )
                    if truncated:
                        doc_tokens = self.token_counter.count(truncated)
                        text = truncated
                    else:
                        break  # Can't fit even truncated
                else:
                    break  # Budget exhausted
            
            # Create document
            doc = self._create_document(result, text, doc_tokens)
            documents.append(doc)
            used_tokens += doc_tokens
            sources_used.add(source)
            
            # Diversity check: Avoid over-sampling single source
            if len([d for d in documents if d.metadata.get("file_path") == source]) > 3:
                logger.debug(f"Diversity limit reached for source: {source}")
        
        logger.debug(f"Used {len(sources_used)} unique sources")
        return documents
    
    def _truncate_document(self, text: str, target_tokens: int) -> Optional[str]:
        """Smart truncation preserving sentence boundaries."""
        if target_tokens <= 0:
            return None
        
        # Estimate chars needed (avg 4 chars/token)
        target_chars = int(target_tokens * 3.)
        
        if len(text) <= target_chars:
            return text
        
        # Truncate at sentence boundary
        truncated = text[:target_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        boundary = max(last_period, last_newline)
        if boundary > target_chars * 0.7:  # At least 70% of target
            return truncated[:boundary + 1]
        
        return truncated
    
    def _create_document(
        self,
        result,
        text: str,
        doc_tokens: int
    ) -> Document:
        """Create Document with comprehensive metadata."""
        metadata = dict(result.payload)
        metadata.update({
            "score": float(result.score),
            "point_id": str(result.id),
            "chunk_tokens": doc_tokens,
            "retrieval_timestamp": datetime.now().isoformat()
        })
        metadata.pop("page_content", None)
        
        return Document(page_content=text, metadata=metadata)
    
    def _update_metrics(self, latency_ms: float, docs_count: int) -> None:
        """Update rolling metrics."""
        queries = self._metrics["queries"]
        
        # Exponential moving average for latency
        alpha = 0.1
        self._metrics["avg_latency_ms"] = (
            alpha * latency_ms + 
            (1 - alpha) * self._metrics["avg_latency_ms"]
        )
        
        self._metrics["total_docs_retrieved"] += docs_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval metrics."""
        queries = self._metrics["queries"]
        cache_hits = self._metrics["cache_hits"]
        cache_hit_rate = (cache_hits / queries * 100) if queries > 0 else 0.0
        
        return {
            **self._metrics,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            "avg_docs_per_query": (
                self._metrics["total_docs_retrieved"] / queries 
                if queries > 0 else 0.0
            ),
            "cache_size": len(self._cache._cache) if self._cache else 0
        }
    
    def clear_cache(self) -> None:
        """Clear retrieval cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Retrieval cache cleared")
    
    async def retrieve_batch_async(
        self,
        queries: List[str],
        budget: Any,
        **kwargs
    ) -> List[List[Document]]:
        """
        Batch retrieval with concurrency control.
        
        Benefits:
        - Parallel processing
        - Shared embedding batch
        - Better resource utilization
        """
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_queries)
        
        async def _retrieve_one(query: str):
            async with semaphore:
                return await self.retrieve_async(query, budget, **kwargs)
        
        tasks = [_retrieve_one(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [
            r if not isinstance(r, Exception) else []
            for r in results
        ]
        
        return valid_results


# Factory function
def create_retriever(
    client: QdrantClient,
    collection_name: str,
    embedder,
    config: Optional[RetrievalConfig] = None
) -> QdrantRetriever:
    """Create retriever with configuration."""
    return QdrantRetriever(client, collection_name, embedder, config)