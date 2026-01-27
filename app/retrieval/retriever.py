"""
Simplified retriever for college project.
Removed: Custom caching, time decay, metrics, batch retrieval.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Simplified configuration for retrieval."""
    timeout_seconds: int = 30


class TokenCounter:
    """Lightweight token counter using heuristic (char_len / 4)."""
    
    def count(self, text: str) -> int:
        """Estimate tokens (approx 4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)


class QdrantRetriever:
    """
    Simplified retriever for college project.
    
    Features:
    - Async retrieval
    - Budget-based filtering
    - Basic document creation
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
        
        logger.info(f"QdrantRetriever initialized: {collection_name}")
    
    async def retrieve_async(
        self,
        query: str,
        budget: Any,
        max_candidates: int = 10,
        score_threshold: float = 0.0,
        collection_name: Optional[str] = None,
        filters: Optional[Filter] = None
    ) -> List[Document]:
        """
        Async retrieval with budget filtering.
        
        Pipeline:
        1. Embed query
        2. Search Qdrant
        3. Filter by token budget
        """
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
            
            # Filter by budget
            documents = self._filter_by_budget(
                search_results.points,
                budget.retrieval_tokens
            )
            
            logger.info(
                f"Retrieved {len(documents)} docs "
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
        max_candidates: int = 10,
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
    
    def _filter_by_budget(
        self,
        search_results: List,
        token_budget: int
    ) -> List[Document]:
        """
        Filter documents by token budget.
        Takes highest-scored documents until budget exhausted.
        """
        documents = []
        used_tokens = 0
        
        for result in search_results:
            text = result.payload.get("page_content", "")
            if not text:
                continue
            
            doc_tokens = self.token_counter.count(text)
            
            # Check budget
            if used_tokens + doc_tokens > token_budget:
                # Try truncation if beneficial
                if not documents or doc_tokens > token_budget * 0.5:
                    truncated = self._truncate_document(
                        text, token_budget - used_tokens
                    )
                    if truncated:
                        doc_tokens = self.token_counter.count(truncated)
                        text = truncated
                    else:
                        break
                else:
                    break
            
            # Create document
            doc = self._create_document(result, text, doc_tokens)
            documents.append(doc)
            used_tokens += doc_tokens
        
        return documents
    
    def _truncate_document(self, text: str, target_tokens: int) -> Optional[str]:
        """Simple truncation at character boundary."""
        if target_tokens <= 0:
            return None
        
        # Estimate chars needed (avg 4 chars/token)
        target_chars = int(target_tokens * 4)
        
        if len(text) <= target_chars:
            return text
        
        return text[:target_chars]
    
    def _create_document(
        self,
        result,
        text: str,
        doc_tokens: int
    ) -> Document:
        """Create Document with metadata."""
        metadata = dict(result.payload)
        metadata.update({
            "score": float(result.score),
            "point_id": str(result.id),
            "chunk_tokens": doc_tokens,
            "retrieval_timestamp": datetime.now().isoformat()
        })
        metadata.pop("page_content", None)
        
        return Document(page_content=text, metadata=metadata)


# Factory function
def create_retriever(
    client: QdrantClient,
    collection_name: str,
    embedder,
    config: Optional[RetrievalConfig] = None
) -> QdrantRetriever:
    """Create retriever with configuration."""
    return QdrantRetriever(client, collection_name, embedder, config)
