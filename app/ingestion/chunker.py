# chunker.py
"""
Semantic chunking with token awareness for DeepSeek V3.2.
Splits documents intelligently while respecting structure and token budgets.
"""

import logging
import re
import tiktoken
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """
    Configuration for semantic chunking with token awareness.
    
    Optimized for DeepSeek V3.2 Special (64K context, recommended 32K).
    """
    # Character-based chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 4000
    
    # Token-aware chunking
    target_tokens: int = 500       # Target tokens per chunk (balanced for quality)
    max_tokens_per_chunk: int = 2000  # Hard limit
    use_token_aware: bool = True
    encoding_name: str = "cl100k_base"  # Standard for most models
    
    # Content preservation
    preserve_code_blocks: bool = True
    preserve_headers: bool = True
    clean_whitespace: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size > self.max_chunk_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be <= "
                f"max_chunk_size ({self.max_chunk_size})"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        if self.target_tokens > self.max_tokens_per_chunk:
            logger.warning(
                f"target_tokens ({self.target_tokens}) > "
                f"max_tokens_per_chunk ({self.max_tokens_per_chunk}), "
                f"adjusting target down"
            )
            self.target_tokens = int(self.max_tokens_per_chunk * 0.7)


class TokenCounter:
    """
    Efficient token counter using tiktoken.
    
    Uses cl100k_base encoding (standard for GPT-4, Claude, DeepSeek, etc.)
    with caching for performance.
    """
    
    _encoding_cache: Dict[str, tiktoken.Encoding] = {}
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter.
        
        Args:
            encoding_name: Tiktoken encoding name (default: cl100k_base)
        """
        self.encoding_name = encoding_name
        self.encoding = self._get_encoding()
    
    def _get_encoding(self) -> tiktoken.Encoding:
        """Get or create cached encoding."""
        if self.encoding_name in self._encoding_cache:
            return self._encoding_cache[self.encoding_name]
        
        try:
            encoding = tiktoken.get_encoding(self.encoding_name)
            self._encoding_cache[self.encoding_name] = encoding
            return encoding
        except Exception as e:
            logger.warning(f"Failed to load encoding {self.encoding_name}: {e}")
            # Fallback to cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")
            self._encoding_cache[self.encoding_name] = encoding
            return encoding
    
    def count(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text or not isinstance(text, str):
            return 0
        
        try:
            return len(self.encoding.encode(text, disallowed_special=()))
        except Exception as e:
            logger.debug(f"Token counting failed: {e}")
            # Rough fallback: ~4 chars per token
            return max(1, len(text) // 4)


class SemanticChunker:
    """
    Intelligent semantic chunker that:
    - Preserves markdown headers as structural anchors
    - Keeps code blocks atomic
    - Respects token budgets
    - Maintains document context
    
    Optimized for:
    - Technical documentation (Next.js, React, Django, etc.)
    - Mixed content (text + code)
    - DeepSeek V3.2 Special token limits
    
    Example:
    --------
    from loaders import RawDocument
    
    config = ChunkingConfig(target_tokens=500, max_tokens_per_chunk=2000)
    chunker = SemanticChunker(config)
    
    raw_doc = RawDocument(
        content="# Introduction\n\nNext.js is...",
        metadata={"source": "nextjs", "file_path": "intro.md"}
    )
    
    chunks = chunker.chunk(raw_doc)
    for chunk in chunks:
        print(f"Tokens: {chunk.metadata['chunk_size_tokens']}")
        print(chunk.page_content[:100])
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize semantic chunker.
        
        Args:
            config: ChunkingConfig instance (uses defaults if None)
        """
        self.config = config or ChunkingConfig()
        self.token_counter = TokenCounter(self.config.encoding_name)
        
        # Header-aware splitting for markdown structure
        if self.config.preserve_headers:
            self.header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "h1"),
                    ("##", "h2"),
                    ("###", "h3"),
                    ("####", "h4"),
                ],
                strip_headers=False
            )
        else:
            self.header_splitter = None
        
        # Recursive splitting for large sections
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n```",      # Code block boundaries
                "\n## ",      # Major headers
                "\n### ",     # Sub-headers
                "\n#### ",    # Sub-sub-headers
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentence breaks
                " ",          # Word breaks
                ""
            ],
            keep_separator=True,
        )
        
        logger.info(
            f"SemanticChunker initialized: "
            f"target={self.config.target_tokens} tokens, "
            f"max={self.config.max_tokens_per_chunk} tokens"
        )
    
    def _clean_content(self, text: str) -> str:
        """Clean content based on configuration."""
        if not text:
            return ""
        
        if self.config.clean_whitespace:
            # Remove excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _is_code_block(self, text: str) -> bool:
        """Check if text contains code blocks."""
        return bool(re.search(r'```', text))
    
    def _extract_code_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Extract code block positions and languages.
        
        Returns:
            List of (start_pos, end_pos, language) tuples
        """
        blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'plaintext'
            start = match.start()
            end = match.end()
            blocks.append((start, end, language))
        
        return blocks
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, Dict[str, str]]]:
        """
        Split text by header hierarchy.
        
        Returns:
            List of (content, header_metadata) tuples
        """
        if not self.header_splitter:
            return [(text, {})]
        
        try:
            chunks = self.header_splitter.split_text(text)
            return [(chunk.page_content, chunk.metadata) for chunk in chunks]
        except Exception as e:
            logger.warning(f"Header splitting failed: {e}")
            return [(text, {})]
    
    def _split_text_respecting_code(self, text: str) -> List[str]:
        """
        Split text while preserving code blocks as atomic units.
        
        Strategy:
        1. Extract all code blocks
        2. Split text between code blocks
        3. Keep code blocks whole (unless too large)
        """
        if not self.config.preserve_code_blocks:
            return self.text_splitter.split_text(text)
        
        code_blocks = self._extract_code_blocks(text)
        
        if not code_blocks:
            # No code blocks, normal splitting
            return self.text_splitter.split_text(text)
        
        chunks = []
        last_end = 0
        
        for start, end, lang in code_blocks:
            # Text before code block
            if start > last_end:
                text_before = text[last_end:start].strip()
                if len(text_before) >= self.config.min_chunk_size:
                    # Split the text portion
                    sub_chunks = self.text_splitter.split_text(text_before)
                    chunks.extend([
                        c for c in sub_chunks 
                        if len(c.strip()) >= self.config.min_chunk_size
                    ])
            
            # Code block as atomic unit
            code_content = text[start:end]
            if len(code_content) >= self.config.min_chunk_size:
                chunks.append(code_content)
            
            last_end = end
        
        # Remaining text after last code block
        if last_end < len(text):
            text_after = text[last_end:].strip()
            if len(text_after) >= self.config.min_chunk_size:
                sub_chunks = self.text_splitter.split_text(text_after)
                chunks.extend([
                    c for c in sub_chunks 
                    if len(c.strip()) >= self.config.min_chunk_size
                ])
        
        return chunks
    
    def _filter_by_token_budget(
        self,
        chunks: List[str]
    ) -> List[Tuple[str, bool]]:
        """
        Filter chunks that exceed token budget.
        
        Returns:
            List of (chunk_text, is_code) tuples
        """
        result = []
        
        for chunk in chunks:
            token_count = self.token_counter.count(chunk)
            is_code = self._is_code_block(chunk)
            
            if token_count <= self.config.max_tokens_per_chunk:
                # Chunk fits within budget
                result.append((chunk, is_code))
            elif is_code:
                # Code block exceeds limit - keep as-is with warning
                logger.warning(
                    f"Code block exceeds token limit "
                    f"({token_count} > {self.config.max_tokens_per_chunk}), "
                    f"keeping as-is"
                )
                result.append((chunk, is_code))
            else:
                # Text chunk too large - split further
                logger.debug(
                    f"Chunk exceeds token limit ({token_count} tokens), "
                    f"splitting further"
                )
                
                # Create smaller splitter
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size // 2,
                    chunk_overlap=self.config.chunk_overlap // 2,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                
                sub_chunks = sub_splitter.split_text(chunk)
                
                for sub_chunk in sub_chunks:
                    sub_tokens = self.token_counter.count(sub_chunk)
                    if (len(sub_chunk.strip()) >= self.config.min_chunk_size and
                        sub_tokens <= self.config.max_tokens_per_chunk):
                        result.append((sub_chunk, False))
        
        return result
    
    def _create_chunk_metadata(
        self,
        base_metadata: Dict,
        chunk_index: int,
        total_chunks: int,
        chunk_text: str,
        header_metadata: Dict,
        is_code: bool
    ) -> Dict:
        """Create comprehensive chunk metadata."""
        tokens = self.token_counter.count(chunk_text)
        
        # Start with base metadata from raw document
        metadata = base_metadata.copy()
        
        # Add chunk-specific metadata
        metadata.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size_chars": len(chunk_text),
            "chunk_size_tokens": tokens,
            "chunk_word_count": len(chunk_text.split()),
            "has_code": is_code,
            "chunk_type": "code" if is_code else "text",
        })
        
        # Merge header metadata (from MarkdownHeaderTextSplitter)
        metadata.update(header_metadata)
        
        return metadata
    
    def chunk(self, raw_document) -> List[Document]:
        """
        Chunk document using semantic awareness.
        
        Pipeline:
        1. Clean content
        2. Split by headers (preserve structure)
        3. Split sections respecting code blocks
        4. Apply token budget filtering
        5. Create Document objects with metadata
        
        Args:
            raw_document: RawDocument from loaders
            
        Returns:
            List of LangChain Document objects with metadata
            
        Example:
        --------
        from loaders import NextJsDocsLoader
        
        loader = NextJsDocsLoader(Path("nextjs-docs"))
        raw_docs = loader.load()
        
        chunker = SemanticChunker()
        
        for raw_doc in raw_docs:
            chunks = chunker.chunk(raw_doc)
            print(f"Split into {len(chunks)} chunks")
        """
        # Step 1: Clean content
        content = self._clean_content(raw_document.content)
        
        if not content:
            logger.warning(f"Empty content for {raw_document.metadata.get('file_path')}")
            return []
        
        # Step 2: Split by headers (maintains document structure)
        header_sections = self._split_by_headers(content)
        
        all_chunks = []
        
        # Step 3: Process each section
        for section_text, header_metadata in header_sections:
            # Split section respecting code blocks
            section_chunks = self._split_text_respecting_code(section_text)
            
            # Apply token budget filtering
            if self.config.use_token_aware:
                section_chunks = self._filter_by_token_budget(section_chunks)
            else:
                section_chunks = [
                    (c, self._is_code_block(c)) 
                    for c in section_chunks
                ]
            
            # Store with header metadata
            all_chunks.extend([
                (chunk, is_code, header_metadata) 
                for chunk, is_code in section_chunks
            ])
        
        # Step 4: Create Document objects
        documents = []
        
        for i, (chunk_text, is_code, header_metadata) in enumerate(all_chunks):
            metadata = self._create_chunk_metadata(
                base_metadata=raw_document.metadata,
                chunk_index=i,
                total_chunks=len(all_chunks),
                chunk_text=chunk_text,
                header_metadata=header_metadata,
                is_code=is_code
            )
            
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
            )
        
        logger.debug(
            f"Chunked {raw_document.metadata.get('file_path', 'unknown')} "
            f"→ {len(documents)} chunks "
            f"(avg {sum(d.metadata['chunk_size_tokens'] for d in documents) // len(documents)} tokens)"
        )
        
        return documents


# Convenience function for quick chunking
def chunk_document(
    content: str,
    metadata: Dict,
    target_tokens: int = 500,
    max_tokens: int = 2000
) -> List[Document]:
    """
    Quick document chunking with defaults.
    
    Args:
        content: Document content
        metadata: Document metadata
        target_tokens: Target tokens per chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of Document chunks
    """
    from loaders import RawDocument
    
    config = ChunkingConfig(
        target_tokens=target_tokens,
        max_tokens_per_chunk=max_tokens
    )
    
    chunker = SemanticChunker(config)
    
    raw_doc = RawDocument(content=content, metadata=metadata)
    
    return chunker.chunk(raw_doc)


# if __name__ == "__main__":
#     # Demo usage
#     logging.basicConfig(level=logging.INFO)
    
#     from loaders import RawDocument
    
#     # Test document with mixed content
#     content = """
# # Next.js Installation

# Next.js is a React framework for production.

# ## Installation

# Install Next.js with npm:

# ```bash
# npx create-next-app@latest my-app
# cd my-app
# npm run dev
# ```

# ## Project Structure

# Your project will have this structure:

# ```
# my-app/
# ├── app/
# ├── public/
# └── package.json
# ```

# That's it! You're ready to start building.
# """
    
#     raw_doc = RawDocument(
#         content=content,
#         metadata={"source": "nextjs", "file_path": "installation.md"}
#     )
    
#     # Chunk with different configs
#     configs = [
#         ChunkingConfig(target_tokens=200, max_tokens_per_chunk=500),
#         ChunkingConfig(target_tokens=500, max_tokens_per_chunk=1000),
#     ]
    
#     for i, config in enumerate(configs, 1):
#         print(f"\n{'='*60}")
#         print(f"Config {i}: target={config.target_tokens}, max={config.max_tokens_per_chunk}")
#         print('='*60)
        
#         chunker = SemanticChunker(config)
#         chunks = chunker.chunk(raw_doc)
        
#         print(f"Created {len(chunks)} chunks:")
#         for j, chunk in enumerate(chunks, 1):
#             tokens = chunk.metadata['chunk_size_tokens']
#             chunk_type = chunk.metadata['chunk_type']
#             preview = chunk.page_content[:80].replace('\n', ' ')
#             print(f"  {j}. [{chunk_type}] {tokens} tokens: {preview}...")