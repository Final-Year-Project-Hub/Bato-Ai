# ingest_qdrant.py
"""
Simplified ingestion pipeline for college project.
Removed: Complex statistics, duplicate checking, extensive logging.
"""

import logging
import time
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.ingestion.loaders import LoaderFactory, LoaderConfig, BaseDocsLoader
from app.ingestion.chunker import SemanticChunker, ChunkingConfig
from app.ingestion.embedder import HuggingFaceEmbedder, APIEmbedder, EmbedderConfig

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "framework_docs"
    recreate_collection: bool = False
    batch_size: int = 100
    show_progress: bool = True


class QdrantIngestor:
    """
    Simplified ingestion pipeline for college project.
    
    Pipeline:
    1. Load raw documents
    2. Chunk documents
    3. Embed chunks (batched)
    4. Index in Qdrant
    """
    
    def __init__(
        self,
        ingest_config: IngestionConfig,
        loader_config: Optional[LoaderConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        embedder_config: Optional[EmbedderConfig] = None,
    ):
        self.ingest_config = ingest_config
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant: {ingest_config.qdrant_url}")
        self.qdrant_client = QdrantClient(
            url=ingest_config.qdrant_url,
            api_key=ingest_config.qdrant_api_key,
            prefer_grpc=False
        )
        
        # Initialize components
        self.loader_config = loader_config or LoaderConfig()
        self.chunking_config = chunking_config or ChunkingConfig()
        self.embedder_config = embedder_config or EmbedderConfig()
        
        # Create embedder
        logger.info("Initializing embedder...")
        if self.embedder_config.provider == "api":
            self.embedder = APIEmbedder(self.embedder_config)
        else:
            self.embedder = HuggingFaceEmbedder(self.embedder_config)
        
        # Create chunker
        logger.info("Initializing chunker...")
        self.chunker = SemanticChunker(self.chunking_config)
        
        logger.info("✅ QdrantIngestor initialized")
    
    def _ensure_collection_exists(self) -> None:
        """Create or validate Qdrant collection."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            collection_name = self.ingest_config.collection_name
            
            if collection_name in collection_names:
                if self.ingest_config.recreate_collection:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.qdrant_client.delete_collection(collection_name)
                else:
                    logger.info(f"Using existing collection: {collection_name}")
                    return
            
            # Get embedding dimension
            embedding_dim = self.embedder.embedding_dim
            
            logger.info(
                f"Creating collection: {collection_name} "
                f"(vectors: {embedding_dim}D, distance: COSINE)"
            )
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                ),
            )
            
            # Create payload index for filtering
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema="keyword"
            )
            
            logger.info(f"✅ Collection created: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _prepare_payload(self, chunk) -> Dict[str, Any]:
        """Prepare payload for Qdrant point."""
        metadata = chunk.metadata.copy()
        metadata["page_content"] = chunk.page_content
        
        # Convert complex types to JSON strings
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                metadata[key] = json.dumps(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = str(value)
        
        return metadata
    
    def ingest(self, loader: BaseDocsLoader) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline.
        
        Pipeline:
        1. Setup Qdrant collection
        2. Load raw documents
        3. Chunk documents
        4. Embed chunks (batched)
        5. Index in Qdrant (batched)
        """
        logger.info(
            f"Starting ingestion: {loader.framework_name} v{loader.version} "
            f"→ {self.ingest_config.collection_name}"
        )
        
        start_time = time.time()
        chunks_indexed = 0
        
        try:
            # Step 1: Setup collection
            logger.info("Step 1/4: Setting up Qdrant collection...")
            self._ensure_collection_exists()
            
            # Step 2: Load documents
            logger.info(f"Step 2/4: Loading {loader.framework_name} documentation...")
            raw_documents = loader.load()
            
            if not raw_documents:
                logger.warning("No documents loaded!")
                return {"chunks_indexed": 0, "elapsed_seconds": 0}
            
            logger.info(f"Loaded {len(raw_documents)} documents")
            
            # Step 3: Chunk documents
            logger.info("Step 3/4: Chunking documents...")
            all_chunks = []
            
            for raw_doc in tqdm(
                raw_documents,
                desc="Chunking",
                unit="doc",
                disable=not self.ingest_config.show_progress
            ):
                try:
                    chunks = self.chunker.chunk(raw_doc)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Chunking failed for {raw_doc.metadata.get('file_path')}: {e}")
            
            logger.info(f"Created {len(all_chunks)} chunks")
            
            # Step 4: Embed and index
            logger.info("Step 4/4: Embedding and indexing chunks...")
            
            batch_size = self.ingest_config.batch_size
            num_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(
                range(0, len(all_chunks), batch_size),
                desc="Indexing",
                unit="batch",
                total=num_batches,
                disable=not self.ingest_config.show_progress
            ):
                batch_chunks = all_chunks[batch_idx:batch_idx + batch_size]
                batch_texts = [c.page_content for c in batch_chunks]
                
                try:
                    # Embed batch
                    embeddings = self.embedder.embed_documents(batch_texts)
                    
                    # Create points
                    points = []
                    for idx, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                        # Simple ID generation
                        point_id = abs(hash(chunk.page_content[:100] + str(batch_idx + idx))) % (2**63 - 1)
                        
                        points.append(
                            PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload=self._prepare_payload(chunk)
                            )
                        )
                    
                    # Upsert to Qdrant (handles duplicates automatically)
                    self.qdrant_client.upsert(
                        collection_name=self.ingest_config.collection_name,
                        points=points
                    )
                    
                    chunks_indexed += len(points)
                    
                except Exception as e:
                    logger.error(f"Batch indexing failed: {e}")
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Ingestion complete: {chunks_indexed} chunks in {elapsed:.2f}s")
            
            return {
                "chunks_indexed": chunks_indexed,
                "elapsed_seconds": round(elapsed, 2)
            }
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise


# ============================================================================
# Convenience Functions
# ============================================================================

def ingest_framework_docs(
    framework: str,
    docs_path: str,
    collection_name: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_provider: str = "local",
    embedding_device: str = "cuda",
    qdrant_url: str = "http://localhost:6333",
    recreate: bool = False,
    target_tokens: int = 500,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """
    Convenience function for complete ingestion.
    
    Args:
        framework: Framework name (nextjs, react, etc.)
        docs_path: Path to documentation directory
        collection_name: Qdrant collection name
        embedding_model: HuggingFace model
        embedding_provider: "api" or "local"
        embedding_device: "cuda" or "cpu"
        qdrant_url: Qdrant server URL
        recreate: Recreate collection if exists
        target_tokens: Target tokens per chunk
        max_tokens: Maximum tokens per chunk
    """
    # Create loader
    loader = LoaderFactory.create(
        framework=framework,
        docs_root=Path(docs_path)
    )
    
    from app.core.config import get_settings
    settings = get_settings()
    
    # Create configs
    ingest_config = IngestionConfig(
        qdrant_url=qdrant_url,
        qdrant_api_key=settings.get_qdrant_key(),
        collection_name=collection_name,
        recreate_collection=recreate
    )
    
    embedder_config = EmbedderConfig(
        model=embedding_model,
        provider=embedding_provider,
        device=embedding_device,
        api_token=settings.get_hf_token()
    )
    
    chunking_config = ChunkingConfig(
        target_tokens=target_tokens,
        max_tokens_per_chunk=max_tokens
    )
    
    # Create ingestor
    ingestor = QdrantIngestor(
        ingest_config=ingest_config,
        embedder_config=embedder_config,
        chunking_config=chunking_config
    )
    
    # Run ingestion
    return ingestor.ingest(loader)
