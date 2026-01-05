
import logging
from app.ingestion.ingest_qdrant import QdrantIngestor, IngestionConfig, EmbedderConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Qdrant collection...")
    
    # Use defaults from config which point to 'framework_docs'
    ingest_config = IngestionConfig() 
    embedder_config = EmbedderConfig()
    
    # Initialize ingestor (loads embedder and connecting to Qdrant)
    ingestor = QdrantIngestor(
        ingest_config=ingest_config,
        embedder_config=embedder_config
    )
    
    # Create collection if missing
    ingestor._ensure_collection_exists()
    
    logger.info("✅ Collection initialization complete.")

if __name__ == "__main__":
    main()
