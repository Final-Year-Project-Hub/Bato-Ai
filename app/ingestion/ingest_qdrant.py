from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pathlib import Path

from app.ingestion.loaders import NextJsDocsLoader
from app.ingestion.chunker import TokenBasedMDXChunker
from app.ingestion.embedder import OfflineEmbedder
from app.core.config import settings

def main():
    project_root = Path(__file__).resolve().parents[2] # app/ingestion/file -> app/ingestion -> app -> root
    docs_root = project_root / "docs" / "nextjs"
    
    if not docs_root.exists():
        print(f"⚠️ Docs root not found at: {docs_root}")
        # Try finding it relative to where script is run if above fails, or just warn
        # For now assume structure is correct
        return

    loader = NextJsDocsLoader(docs_root)
    raw_docs = loader.load()

    # Use TokenBasedMDXChunker instead of splitter
    chunker = TokenBasedMDXChunker(max_tokens=500, overlap=50)
    chunks = chunker.chunk(raw_docs)
    print(f"🧩 Total chunks: {len(chunks)}")

    if not chunks:
        print("No chunks generated. Exiting.")
        return

    embedder = OfflineEmbedder()
    # Batch embedding
    chunk_texts = [doc.page_content for doc in chunks]
    vectors = embedder.embed(chunk_texts)

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=len(vectors[0]),
                distance=Distance.COSINE
            )
        )

    # Upload in batches to avoid payload too large errors
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_payloads = [doc.metadata for doc in chunks[i:i+batch_size]]
        # We need to add page_content to payload for retrieval!!!
        for j, payload in enumerate(batch_payloads):
            payload["text"] = chunk_texts[i+j] # Embed content in payload for retrieval

        # Using integer IDs for simplicity, or we let Qdrant assign UUIDs
        # client.upsert accepts points directly or upload_collection
        
        client.upload_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors=batch_vectors,
            payload=batch_payloads,
            parallel=2
        )
        print(f"Uploaded batch {i} - {i+batch_size}")

    print("✅ Qdrant ingestion complete")


if __name__ == "__main__":
    main()
