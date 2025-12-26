from pathlib import Path
from langchain_core.documents import Document
from tqdm import tqdm


class NextJsDocsLoader:
    def __init__(self, docs_root: Path):
        if not docs_root.exists():
            raise FileNotFoundError(f"Docs root not found: {docs_root}")
        self.docs_root = docs_root

    def load(self) -> list[Document]:
        files = list(self.docs_root.rglob("*.mdx"))
        print(f"📄 Found {len(files)} documentation files")

        documents = []

        for file in tqdm(files, desc="Loading Next.js docs"):
            text = file.read_text(encoding="utf-8")

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "nextjs-docs",
                        "file_path": str(file.relative_to(self.docs_root)),
                        "section": file.parts[0] if len(file.parts) > 0 else "root",
                        "topic": file.stem.lower()
                    }
                )
            )

        print(f"✅ Total documents loaded: {len(documents)}")
        return documents
