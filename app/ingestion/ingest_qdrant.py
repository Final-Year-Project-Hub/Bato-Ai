# ingest_qdrant.py
"""
Production-ready ingestion pipeline: Load → Chunk → Embed → Index.
Optimized for DeepSeek V3.2 and BAAI/bge-small-en-v1.5.
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import hashlib

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
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "framework_docs"
    recreate_collection: bool = False
    
    # Processing settings
    batch_size: int = 100  # Optimized: was 50, increased for faster indexing
    show_progress: bool = True
    log_stats: bool = True
    
    # Performance settings
    max_workers: int = 4  # For parallel processing (future)
    checkpoint_interval: int = 1000  # Save checkpoint every N chunks


class IngestionStats:
    """Track ingestion statistics with detailed metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Document statistics
        self.documents_loaded = 0
        self.documents_failed = 0
        
        # Chunk statistics
        self.chunks_created = 0
        self.chunks_indexed = 0
        self.chunks_failed = 0
        
        # Token statistics
        self.total_tokens = 0
        self.avg_tokens_per_chunk = 0
        
        # Performance statistics
        self.embedding_time = 0.0
        self.indexing_time = 0.0
    
    def elapsed(self) -> float:
        """Time elapsed since start."""
        return time.time() - self.start_time
    
    def chunks_per_second(self) -> float:
        """Throughput in chunks per second."""
        elapsed = self.elapsed()
        return self.chunks_indexed / elapsed if elapsed > 0 else 0
    
    def update_after_chunking(self, chunks: List) -> None:
        """Update stats after chunking phase."""
        self.chunks_created += len(chunks)
        tokens = sum(c.metadata.get('chunk_size_tokens', 0) for c in chunks)
        self.total_tokens += tokens
        
        if self.chunks_created > 0:
            self.avg_tokens_per_chunk = self.total_tokens // self.chunks_created
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "documents": {
                "loaded": self.documents_loaded,
                "failed": self.documents_failed,
            },
            "chunks": {
                "created": self.chunks_created,
                "indexed": self.chunks_indexed,
                "failed": self.chunks_failed,
            },
            "tokens": {
                "total": self.total_tokens,
                "avg_per_chunk": self.avg_tokens_per_chunk,
            },
            "performance": {
                "elapsed_seconds": round(self.elapsed(), 2),
                "chunks_per_second": round(self.chunks_per_second(), 2),
                "embedding_time_seconds": round(self.embedding_time, 2),
                "indexing_time_seconds": round(self.indexing_time, 2),
            }
        }


class QdrantIngestor:
    """
    Complete ingestion pipeline orchestrator.
    
    Pipeline:
    1. Load raw documents (via loaders.py)
    2. Chunk documents (via chunker.py)
    3. Embed chunks (via embedder.py)
    4. Index in Qdrant
    
    Features:
    - Progress tracking with tqdm
    - Comprehensive statistics
    - Error recovery
    - Batch processing
    - Metadata preservation
    
    Example:
    --------
    from pathlib import Path
    
    # Configure components
    ingest_config = IngestionConfig(
        collection_name="nextjs_docs",
        recreate_collection=True
    )
    
    embedder_config = EmbedderConfig(
        model="BAAI/bge-small-en-v1.5",
        device="cuda"
    )
    
    chunking_config = ChunkingConfig(
        target_tokens=500,
        max_tokens_per_chunk=2000
    )
    
    # Create ingestor
    ingestor = QdrantIngestor(
        ingest_config=ingest_config,
        embedder_config=embedder_config,
        chunking_config=chunking_config
    )
    
    # Load documentation
    from loaders import LoaderFactory
    loader = LoaderFactory.create("nextjs", Path("nextjs-docs"))
    
    # Run ingestion
    stats = ingestor.ingest(loader)
    print(json.dumps(stats, indent=2))
    """
    
    def __init__(
        self,
        ingest_config: IngestionConfig,
        loader_config: Optional[LoaderConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        embedder_config: Optional[EmbedderConfig] = None,
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            ingest_config: Ingestion configuration
            loader_config: Document loader configuration
            chunking_config: Chunking configuration
            embedder_config: Embedder configuration
        """
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
        
        # Statistics
        self.stats = IngestionStats()
        
        logger.info("✅ QdrantIngestor initialized")
    
    def _ensure_collection_exists(self) -> None:
        """
        Create or validate Qdrant collection.
        
        Creates collection with proper vector configuration.
        Optionally recreates if recreate_collection=True.
        """
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
            
            logger.info(f"✅ Collection created: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _generate_point_id(self, content: str, index: int) -> int:
        """
        Generate unique point ID for Qdrant.
        
        Uses MD5 hash of content + index to ensure uniqueness.
        
        Args:
            content: Chunk content
            index: Chunk index
            
        Returns:
            Unique integer ID
        """
        hash_input = f"{content[:100]}{index}".encode('utf-8')
        hash_obj = hashlib.md5(hash_input)
        # Convert to positive int within Qdrant limits
        return abs(int(hash_obj.hexdigest(), 16)) % (2 ** 63 - 1)
    
    def _prepare_payload(self, chunk) -> Dict[str, Any]:
        """
        Prepare payload for Qdrant point.
        
        Converts metadata to Qdrant-compatible format:
        - Lists/dicts → JSON strings
        - All values → str/int/float/bool/None
        
        Args:
            chunk: LangChain Document with metadata
            
        Returns:
            Payload dictionary
        """
        # Copy metadata
        metadata = chunk.metadata.copy()
        
        # Add page_content to payload (needed for retrieval)
        metadata["page_content"] = chunk.page_content
        
        # Convert complex types to JSON strings
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                metadata[key] = json.dumps(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = str(value)
        
        return metadata
    
    def _create_points(
        self,
        chunks: List,
        embeddings: List[List[float]],
        batch_id: int
    ) -> List[PointStruct]:
        """
        Create Qdrant points from chunks and embeddings.
        
        Args:
            chunks: List of Document chunks
            embeddings: List of embedding vectors
            batch_id: Batch identifier for ID generation
            
        Returns:
            List of PointStruct objects
        """
        points = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID
            point_id = self._generate_point_id(
                chunk.page_content,
                batch_id * 1000 + idx
            )
            
            # Prepare payload
            payload = self._prepare_payload(chunk)
            
            # Create point
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        return points
    
    def ingest(
        self,
        loader: BaseDocsLoader
    ) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline.
        
        Pipeline:
        1. Setup Qdrant collection
        2. Load raw documents
        3. Chunk documents
        4. Embed chunks (batched)
        5. Index in Qdrant (batched)
        6. Log statistics
        
        Args:
            loader: Initialized document loader
            
        Returns:
            Dictionary with ingestion statistics
            
        Example:
        --------
        from loaders import NextJsDocsLoader
        from pathlib import Path
        
        loader = NextJsDocsLoader(Path("nextjs-docs"))
        stats = ingestor.ingest(loader)
        
        print(f"Indexed {stats['chunks']['indexed']} chunks")
        """
        logger.info(
            f"Starting ingestion: {loader.framework_name} v{loader.version} "
            f"→ {self.ingest_config.collection_name}"
        )
        
        self.stats.start_time = time.time()
        
        try:
            # Step 1: Setup collection
            logger.info("Step 1/4: Setting up Qdrant collection...")
            self._ensure_collection_exists()
            
            # Step 2: Load documents
            logger.info(f"Step 2/4: Loading {loader.framework_name} documentation...")
            raw_documents = loader.load()
            
            self.stats.documents_loaded = len(raw_documents)
            
            if not raw_documents:
                logger.warning("No documents loaded!")
                return self.stats.summary()
            
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
                    self.stats.update_after_chunking(chunks)
                except Exception as e:
                    logger.error(f"Chunking failed for {raw_doc.metadata.get('file_path')}: {e}")
                    self.stats.documents_failed += 1
            
            logger.info(
                f"Created {len(all_chunks)} chunks "
                f"(avg {self.stats.avg_tokens_per_chunk} tokens)"
            )
            
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
                    embed_start = time.time()
                    embeddings = self.embedder.embed_documents(batch_texts)
                    self.stats.embedding_time += time.time() - embed_start
                    
                    # Create points
                    points = self._create_points(
                        batch_chunks,
                        embeddings,
                        batch_idx // batch_size
                    )
                    
                    # Index in Qdrant
                    index_start = time.time()
                    self.qdrant_client.upsert(
                        collection_name=self.ingest_config.collection_name,
                        points=points
                    )
                    self.stats.indexing_time += time.time() - index_start
                    
                    self.stats.chunks_indexed += len(points)
                    
                except Exception as e:
                    logger.error(f"Batch indexing failed: {e}")
                    self.stats.chunks_failed += len(batch_chunks)
            
            # Step 5: Log statistics
            if self.ingest_config.log_stats:
                self._log_stats()
            
            return self.stats.summary()
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise
    
    def _log_stats(self) -> None:
        """Log comprehensive ingestion statistics."""
        summary = self.stats.summary()
        embedder_stats = self.embedder.get_stats()
        
        logger.info("\n" + "=" * 70)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 70)
        
        # Collection info
        logger.info(f"Collection: {self.ingest_config.collection_name}")
        
        # Document stats
        logger.info(f"\nDocuments:")
        logger.info(f"  Loaded: {summary['documents']['loaded']}")
        logger.info(f"  Failed: {summary['documents']['failed']}")
        
        # Chunk stats
        logger.info(f"\nChunks:")
        logger.info(f"  Created: {summary['chunks']['created']}")
        logger.info(f"  Indexed: {summary['chunks']['indexed']}")
        logger.info(f"  Failed: {summary['chunks']['failed']}")
        
        # Token stats
        logger.info(f"\nTokens:")
        logger.info(f"  Total: {summary['tokens']['total']:,}")
        logger.info(f"  Avg per chunk: {summary['tokens']['avg_per_chunk']}")
        
        # Performance stats
        logger.info(f"\nPerformance:")
        logger.info(f"  Total time: {summary['performance']['elapsed_seconds']:.2f}s")
        logger.info(f"  Throughput: {summary['performance']['chunks_per_second']:.2f} chunks/sec")
        logger.info(f"  Embedding time: {summary['performance']['embedding_time_seconds']:.2f}s")
        logger.info(f"  Indexing time: {summary['performance']['indexing_time_seconds']:.2f}s")
        
        # Embedder stats
        logger.info(f"\nEmbedder:")
        logger.info(f"  Model: {embedder_stats['model']}")
        logger.info(f"  Dimension: {embedder_stats['embedding_dim']}")
        logger.info(f"  Device: {embedder_stats['device']}")
        logger.info(f"  Cache hits: {embedder_stats['hits']}")
        logger.info(f"  Cache hit rate: {embedder_stats['hit_rate']:.2%}")
        
        logger.info("=" * 70 + "\n")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics."""
        try:
            collection = self.qdrant_client.get_collection(
                self.ingest_config.collection_name
            )
            
            return {
                "name": self.ingest_config.collection_name,
                "points_count": collection.points_count,
                "vectors_count": collection.vectors_count,
                "status": collection.status.value if hasattr(collection.status, 'value') else str(collection.status),
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


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
        embedding_model: HuggingFace model (default: bge-small)
        embedding_device: "cuda" or "cpu"
        qdrant_url: Qdrant server URL
        recreate: Recreate collection if exists
        target_tokens: Target tokens per chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Dictionary with ingestion statistics
        
    Example:
    --------
    stats = ingest_framework_docs(
        framework="nextjs",
        docs_path="./nextjs-docs",
        collection_name="nextjs_v14",
        embedding_model="BAAI/bge-small-en-v1.5",
        recreate=True
    )
    
    print(f"Indexed {stats['chunks']['indexed']} chunks")
    """
    # Create loader
    loader = LoaderFactory.create(
        framework=framework,
        docs_root=Path(docs_path)
    )
    
    # Create configs
    ingest_config = IngestionConfig(
        qdrant_url=qdrant_url,
        qdrant_api_key=settings.get_qdrant_key(),
        collection_name=collection_name,
        recreate_collection=recreate
    )
    
    from app.core.config import get_settings
    settings = get_settings()
    
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


# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # Define documentation paths (relative to project root)
#     # Uses unified collection for all frameworks
#     DOCS_CONFIG = {
#         "nextjs": {
#             "path": "docs/nextjs",
#             "collection": "framework_docs"
#         },
#         "react": {
#             "path": "docs/react",
#             "collection": "framework_docs"
#         },
#         "python": {
#             "path": "docs/python",
#             "collection": "framework_docs"
#         }
#     }
    
#     # Track if collection needs recreation (only first framework)
#     first_framework = True
    
#     for framework, config in DOCS_CONFIG.items():
#         print("\n" + "=" * 50)
#         print(f"Processing {framework.title()} Documentation")
#         print("=" * 50)
        
#         docs_path = Path(config["path"])
#         if not docs_path.exists():
#             print(f"⚠️  Documentation path not found: {docs_path}")
#             print(f"    Run 'python scripts/download_docs.py' first.")
#             continue
            
#         try:
#             stats = ingest_framework_docs(
#                 framework=framework,
#                 docs_path=str(docs_path),
#                 collection_name=config["collection"],
#                 recreate=first_framework  # Only recreate for first framework
#             )
#             first_framework = False  # Subsequent frameworks append to collection
#             print(json.dumps(stats, indent=2))
#         except Exception as e:
#             print(f"❌ Failed to ingest {framework} docs: {e}")
#             import traceback
#             traceback.print_exc()

