import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from tqdm import tqdm


class NextJsDocsLoader:
    """
    Loads Next.js official documentation (.md / .mdx)
    into LangChain Document objects.
    """

    SUPPORTED_EXTENSIONS = {".md", ".mdx"}

    def __init__(self, docs_root: str | Path):
        self.docs_root = Path(docs_root).resolve()

        if not self.docs_root.exists():
            raise FileNotFoundError(f"Docs root not found at {self.docs_root}")

    def _is_doc_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"❌ Failed to read {path}: {e}")
            return ""

    def load(self) -> List[Document]:
        documents: List[Document] = []

        all_files = list(self.docs_root.rglob("*"))
        doc_files = [f for f in all_files if f.is_file() and self._is_doc_file(f)]

        print(f"📄 Found {len(doc_files)} documentation files")

        for file_path in tqdm(doc_files, desc="Loading Next.js docs"):
            content = self._read_file(file_path)

            if not content.strip():
                continue

            relative_path = file_path.relative_to(self.docs_root)

            metadata = {
                "source": "nextjs-docs",
                "file_path": str(relative_path),
                "section": (
                    relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else "root"
                ),
                "topic": file_path.stem,
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        print("✅ Total documents loaded:", len(documents))
        return documents
