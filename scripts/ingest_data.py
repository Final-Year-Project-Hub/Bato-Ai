
import asyncio
import os
import logging
from pathlib import Path
from app.ingestion.ingest_qdrant import ingest_framework_docs
from app.core.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Run ingestion for all frameworks defined in settings.
    Reads QDRANT_URL and QDRANT_API_KEY from environment variables.
    """
    settings = get_settings()
    
    # Override with env vars if present (allows running locally against cloud)
    qdrant_url = os.getenv("QDRANT_URL", settings.QDRANT_URL)
    qdrant_api_key = os.getenv("QDRANT_API_KEY", settings.get_qdrant_key())
    
    logger.info("=" * 50)
    logger.info(f"Generated Ingestion Script")
    logger.info(f"Target Qdrant: {qdrant_url}")
    logger.info("=" * 50)

    if "localhost" in qdrant_url and os.getenv("QDRANT_URL"):
        logger.warning("⚠️ You are pointing to localhost but provided QDRANT_URL env var. double check!")

    # Frameworks to ingest
    # You can customize this list or read from settings
    frameworks = {
        "nextjs": "docs/nextjs",
        "react": "docs/react",
        "python": "docs/python",
    }

    first = True
    for name, path_str in frameworks.items():
        doc_path = Path(path_str)
        if not doc_path.exists():
            logger.warning(f"Skipping {name}: path {doc_path} not found")
            continue
            
        logger.info(f"Starting ingestion for {name}...")
        try:
            stats = ingest_framework_docs(
                framework=name,
                docs_path=str(doc_path),
                collection_name="framework_docs", # Unified collection
                qdrant_url=qdrant_url,
                # recreate=first, # Only recreate on the first one
                recreate=False, # Safer to not recreate by default, or ask user? 
                                # Let's set False to append. User can clear manually if needed.
                                # Actually, for a fresh cloud instance, we might want to ensure it's created.
                                # But _ensure_collection_exists in the ingestor handles creation.
                target_tokens=500,
                max_tokens=2000
            )
            logger.info(f"✅ {name} complete. Indexed {stats['chunks']['indexed']} chunks.")
        except Exception as e:
            logger.error(f"❌ Failed to ingest {name}: {e}")
        
        first = False

if __name__ == "__main__":
    main()
