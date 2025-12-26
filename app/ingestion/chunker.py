from typing import List
import re

import tiktoken
from langchain_core.documents import Document


class TokenBasedMDXChunker:
    """
    Splits MDX / Markdown documents into semantic, token-limited chunks.
    """

    def __init__(self, max_tokens: int = 500, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap

        # GPT-style tokenizer (industry standard)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # -------------------------
    # TOKEN UTILS
    # -------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Sliding window token split with overlap.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.max_tokens - self.overlap

        return chunks

    # -------------------------
    # MARKDOWN UTILS
    # -------------------------

    def _strip_frontmatter(self, text: str) -> str:
        """
        Removes YAML frontmatter (--- ... ---)
        """
        return re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)

    def _split_by_headings(self, text: str):
        """
        Splits markdown by ## and ### headings.
        Keeps heading text.
        """
        pattern = r"(?:^|\n)(##+ .*)"
        parts = re.split(pattern, text)

        sections = []
        current_heading = "Introduction"
        current_level = 1
        buffer = ""

        for part in parts:
            if part.startswith("##"):
                if buffer.strip():
                    sections.append((current_heading, current_level, buffer))

                current_heading = part.strip().lstrip("# ").strip()
                current_level = part.count("#")
                buffer = ""
            else:
                buffer += part

        if buffer.strip():
            sections.append((current_heading, current_level, buffer))

        return sections

    # -------------------------
    # MAIN API
    # -------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs: List[Document] = []

        for doc in documents:
            raw_text = self._strip_frontmatter(doc.page_content)
            sections = self._split_by_headings(raw_text)

            for heading, level, content in sections:
                if not content.strip():
                    continue

                token_count = self._count_tokens(content)

                if token_count <= self.max_tokens:
                    chunk_texts = [content]
                else:
                    chunk_texts = self._split_by_tokens(content)

                for idx, chunk_text in enumerate(chunk_texts):
                    metadata = {
                        **doc.metadata,
                        "heading": heading,
                        "level": level,
                        "chunk_index": idx,
                    }

                    chunked_docs.append(
                        Document(
                            page_content=chunk_text.strip(),
                            metadata=metadata,
                        )
                    )

        return chunked_docs
