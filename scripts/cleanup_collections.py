import asyncio
import logging
import sys
import os
import argparse

# Fix ModuleNotFoundError
sys.path.append(os.getcwd())

from qdrant_client import QdrantClient
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_collections():
    """
    List all collections and delete those that don't match the current configuration.
    """
    parser = argparse.ArgumentParser(description="Cleanup Qdrant collections")
    parser.add_argument("--force", action="store_true", help="Delete without confirmation")
    args = parser.parse_args()

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    try:
        collections = client.get_collections().collections
        current_collection = settings.QDRANT_COLLECTION_NAME
        
        logger.info(f"Current target collection: {current_collection}")
        logger.info(f"Found {len(collections)} collections in Qdrant:")
        
        for col in collections:
            print(f" - {col.name}")
            
        print("\n")
        
        for col in collections:
            if col.name != current_collection:
                if args.force:
                    logger.info(f"Deleting {col.name} (FORCE)...")
                    client.delete_collection(col.name)
                    logger.info("Deleted.")
                else:
                    confirm = input(f"Delete outdated collection '{col.name}'? (y/n): ")
                    if confirm.lower() == 'y':
                        logger.info(f"Deleting {col.name}...")
                        client.delete_collection(col.name)
                        logger.info("Deleted.")
                    else:
                        logger.info(f"Skipped {col.name}")
            else:
                logger.info(f"Keeping active collection '{col.name}'")

    except Exception as e:
        logger.error(f"Error accessing Qdrant: {e}")

if __name__ == "__main__":
    cleanup_collections()

