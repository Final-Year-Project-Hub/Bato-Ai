from typing import List
from qdrant_client import QdrantClient
from app.retrieval.query_analyzer import QueryIntent

class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection: str, embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, intent: QueryIntent, token_budget: int):
        # Allow retrieval for all intents now
        # if intent != QueryIntent.LEARN:
        #     return []


        vector = self.embedder.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=5,
            with_payload=True,
        )

        docs = []
        used_tokens = 0

        for point in response.points:
            text = point.payload.get("text", "")
            tokens = len(text.split())

            if used_tokens + tokens > token_budget:
                break

            docs.append(text)
            used_tokens += tokens

        return docs
