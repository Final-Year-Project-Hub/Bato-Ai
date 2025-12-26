from qdrant_client import QdrantClient

from app.ingestion.embedder import OfflineEmbedder
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.token_budget import TokenBudgetPlanner
from app.retrieval.qdrant_retriever import QdrantRetriever


def main():
    print("\n=== RETRIEVER SYSTEM TEST ===\n")

    query = "I want to learn Next.js routing"

    analyzer = QueryAnalyzer()
    intent = analyzer.analyze(query)

    planner = TokenBudgetPlanner()
    budget = planner.plan(intent)

    embedder = OfflineEmbedder()

    client = QdrantClient(path="./qdrant_data")

    retriever = QdrantRetriever(
        client=client,
        collection_name="nextjs_docs",
        embedder=embedder,
    )

    print("Query:", query)
    print("Intent:", intent)
    print("Budget:", budget)

    docs = retriever.retrieve(query, intent, budget)

    print(f"\nRetrieved {len(docs)} docs:\n")
    for d in docs[:3]:
        print("-", d.metadata.get("file_path"))


if __name__ == "__main__":
    main()
