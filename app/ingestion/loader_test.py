from pathlib import Path
from loaders import NextJsDocsLoader

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCS_ROOT = PROJECT_ROOT / "docs" / "nextjs"

print("Docs root path:", DOCS_ROOT)

loader = NextJsDocsLoader(DOCS_ROOT)
docs = loader.load()

print("Sample doc metadata:", docs[0].metadata)
print("Sample content preview:", docs[0].page_content[:200])

# import random

# sample = random.choice(documents)

# print("\n=== SAMPLE DOCUMENT ===")
# print("Metadata:", sample["metadata"])
# print("\nContent Preview:\n")
# print(sample["content"][:800])

# from collections import Counter

# topics = Counter()
# routers = Counter()

# for doc in documents:
#     topics[doc["metadata"]["topic"]] += 1
#     routers[doc["metadata"]["router"]] += 1

# print("\n=== TOPICS DISTRIBUTION ===")
# for topic, count in topics.most_common(10):
#     print(topic, count)

# print("\n=== ROUTER DISTRIBUTION ===")
# print(routers)

# query = "routing"

# matched = [
#     doc for doc in documents
#     if query in doc["metadata"]["topic"].lower()
# ]

# print(f"\nDocs matching topic '{query}': {len(matched)}")

# for d in matched[:3]:
#     print("-", d["metadata"]["path"])

