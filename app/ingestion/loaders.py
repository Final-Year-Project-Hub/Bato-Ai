# loaders.py
"""
Production-ready framework documentation loaders.
Optimized for Next.js, React, Django, FastAPI with smart preprocessing.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """
    Raw document before chunking.
    
    Attributes:
        content: Document text content
        metadata: Dictionary with source, file_path, and other info
    """
    content: str
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required metadata fields."""
        required = {'source', 'file_path'}
        if not required.issubset(self.metadata.keys()):
            raise ValueError(f"metadata must contain: {required}")


@dataclass
class LoaderConfig:
    """Configuration for document loading."""
    # File filtering
    skip_hidden_files: bool = True
    skip_excluded_dirs: bool = True
    remove_empty_files: bool = True
    
    # Encoding handling
    primary_encoding: str = "utf-8"
    fallback_encodings: List[str] = field(
        default_factory=lambda: ["latin-1", "cp1252", "iso-8859-1"]
    )
    
    # Content extraction
    extract_frontmatter: bool = True
    extract_code_languages: bool = True
    extract_headers: bool = True
    
    # Size limits
    min_file_size: int = 50          # bytes
    max_file_size: int = 10_000_000  # 10MB
    
    # Excluded directories (common noise)
    excluded_dirs: set = field(
        default_factory=lambda: {
            'node_modules', 'tests', '__pycache__', 'dist', 'build',
            '.next', 'venv', '.git', '.env', 'coverage', '.vscode',
            '__tests__', 'test', '.pytest_cache', '.tox'
        }
    )


class BaseDocsLoader(ABC):
    """
    Base class for all documentation loaders.
    
    Provides common functionality:
    - File discovery and filtering
    - Encoding detection and reading
    - Metadata extraction
    - Progress tracking
    
    Subclass for specific frameworks.
    """
    
    def __init__(
        self,
        docs_root: Path,
        framework_name: str,
        version: str = "latest",
        base_url: Optional[str] = None,
        config: Optional[LoaderConfig] = None
    ):
        """
        Initialize documentation loader.
        
        Args:
            docs_root: Path to documentation root directory
            framework_name: Framework identifier (nextjs, react, etc.)
            version: Documentation version
            base_url: Base URL for online docs
            config: LoaderConfig instance
        """
        if not docs_root.exists():
            raise FileNotFoundError(f"Documentation root not found: {docs_root}")
        
        self.docs_root = Path(docs_root)
        self.framework_name = framework_name
        self.version = version
        self.base_url = base_url
        self.config = config or LoaderConfig()
        self._supported_extensions = self.get_supported_extensions()
        
        logger.info(
            f"Initialized {framework_name} loader: "
            f"root={docs_root}, version={version}"
        )
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g., ['.md', '.mdx'])."""
        pass
    
    def extract_frontmatter(self, text: str) -> Tuple[Dict, str]:
        """
        Extract YAML frontmatter from markdown.
        
        Returns:
            (frontmatter_dict, content_without_frontmatter)
        """
        if not self.config.extract_frontmatter:
            return {}, text
        
        if not text.startswith('---'):
            return {}, text
        
        try:
            parts = text.split('---', 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                content = parts[2].strip()
                return frontmatter, content
        except yaml.YAMLError as e:
            logger.debug(f"Failed to parse frontmatter: {e}")
        
        return {}, text
    
    def extract_headings(self, text: str) -> List[Tuple[int, str]]:
        """
        Extract markdown headings.
        
        Returns:
            List of (level, heading_text) tuples
        """
        if not self.config.extract_headers:
            return []
        
        headings = []
        pattern = r'^(#{1,6})\s+(.+?)(?:\s*{[^}]*})?$'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            level = len(match.group(1))
            heading = match.group(2).strip()
            # Remove markdown links from headings
            heading = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', heading)
            headings.append((level, heading))
        
        return headings
    
    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks with languages.
        
        Returns:
            List of (language, code) tuples
        """
        if not self.config.extract_code_languages:
            return []
        
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'plaintext'
            code = match.group(2).strip()
            code_blocks.append((language, code))
        
        return code_blocks
    
    def build_heading_hierarchy(self, text: str) -> List[str]:
        """
        Build hierarchical path from top-level headings.
        
        Returns:
            List of H1 and H2 headings for context
        """
        headings = self.extract_headings(text)
        if not headings:
            return []
        
        # Return only top 2 levels for hierarchy
        return [h[1] for h in headings if h[0] <= 2]
    
    def construct_url(self, file: Path) -> Optional[str]:
        """
        Construct online documentation URL for file.
        
        Args:
            file: Path to documentation file
            
        Returns:
            Online URL or None if base_url not configured
        """
        if not self.base_url:
            return None
        
        relative_path = file.relative_to(self.docs_root)
        url_path = str(relative_path.with_suffix('')).replace('\\', '/')
        
        # Remove 'index' from end of path
        if url_path.endswith('/index'):
            url_path = url_path[:-6]
        
        return f"{self.base_url.rstrip('/')}/{url_path}"
    
    def extract_metadata(
        self,
        file: Path,
        text: str,
        frontmatter: Dict
    ) -> Dict:
        """
        Extract comprehensive metadata from file.
        
        Returns:
            Dictionary with all metadata fields
        """
        relative_path = file.relative_to(self.docs_root)
        parts = relative_path.parts
        
        # Extract code information
        code_blocks = self.extract_code_blocks(text)
        has_code = len(code_blocks) > 0
        code_languages = list(set(lang for lang, _ in code_blocks)) if code_blocks else []
        
        # File statistics
        file_stat = file.stat()
        last_modified = datetime.fromtimestamp(file_stat.st_mtime)
        
        # Build metadata
        metadata = {
            # Core identifiers
            "source": self.framework_name,
            "version": self.version,
            "file_path": str(relative_path),
            "file_name": file.stem,
            "file_type": file.suffix,
            
            # Structure
            "section": parts[0] if len(parts) > 0 else "root",
            "subsection": parts[1] if len(parts) > 1 else None,
            
            # URLs
            "url": self.construct_url(file),
            
            # Timestamps
            "last_modified": last_modified.isoformat(),
            
            # Content characteristics
            "has_code": has_code,
            "code_languages": code_languages,
            "heading_hierarchy": self.build_heading_hierarchy(text),
            
            # Frontmatter fields
            "title": frontmatter.get("title", file.stem),
            "description": frontmatter.get("description", ""),
            "tags": frontmatter.get("tags", []),
            "keywords": frontmatter.get("keywords", []),
            
            # Statistics
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": text.count('\n') + 1,
        }
        
        return metadata
    
    def should_process_file(self, file: Path) -> bool:
        """
        Determine if file should be processed.
        
        Filters based on:
        - Hidden files
        - Excluded directories
        - File size
        """
        # Skip hidden files
        if self.config.skip_hidden_files and file.name.startswith('.'):
            return False
        
        # Skip excluded directories
        if self.config.skip_excluded_dirs:
            if any(excluded in file.parts for excluded in self.config.excluded_dirs):
                return False
        
        # Check file size
        try:
            file_size = file.stat().st_size
            if (file_size < self.config.min_file_size or 
                file_size > self.config.max_file_size):
                return False
        except OSError:
            return False
        
        return True
    
    def read_file_safe(self, file: Path) -> Optional[str]:
        """
        Read file with encoding fallbacks.
        
        Tries primary encoding, then fallbacks.
        
        Returns:
            File content or None if all encodings fail
        """
        encodings = [self.config.primary_encoding] + self.config.fallback_encodings
        
        for encoding in encodings:
            try:
                return file.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                return None
        
        logger.error(f"Failed to read {file} with any encoding")
        return None
    
    def preprocess_content(self, text: str) -> str:
        """
        Preprocess content (override in subclasses).
        
        Default: strip whitespace only.
        """
        return text.strip()
    
    def load(self) -> List[RawDocument]:
        """
        Load all documentation files as RawDocuments.
        
        Pipeline:
        1. Discover files by extensions
        2. Filter files
        3. Read with encoding fallbacks
        4. Extract frontmatter
        5. Preprocess content
        6. Build metadata
        7. Create RawDocument objects
        
        Returns:
            List of RawDocument instances
        """
        # Discover files
        files = []
        for ext in self._supported_extensions:
            files.extend(self.docs_root.rglob(f"*{ext}"))
        
        # Filter files
        files = [f for f in files if self.should_process_file(f)]
        
        logger.info(
            f"📄 Found {len(files)} {self.framework_name} v{self.version} "
            f"files to process"
        )
        
        # Process files
        raw_documents = []
        skipped = 0
        errors = 0
        
        for file in tqdm(
            files,
            desc=f"Loading {self.framework_name}",
            unit="file"
        ):
            try:
                # Read file
                text = self.read_file_safe(file)
                if not text:
                    errors += 1
                    continue
                
                # Skip empty files
                if self.config.remove_empty_files and not text.strip():
                    skipped += 1
                    continue
                
                # Extract frontmatter
                frontmatter, content = self.extract_frontmatter(text)
                
                # Preprocess
                processed_text = self.preprocess_content(content)
                
                # Extract metadata
                metadata = self.extract_metadata(file, processed_text, frontmatter)
                
                # Create RawDocument
                raw_documents.append(
                    RawDocument(content=processed_text, metadata=metadata)
                )
                
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                errors += 1
        
        logger.info(
            f"✅ {self.framework_name}: {len(raw_documents)} documents loaded, "
            f"{skipped} skipped, {errors} errors"
        )
        
        return raw_documents


# ============================================================================
# Framework-Specific Loaders
# ============================================================================

class NextJsDocsLoader(BaseDocsLoader):
    """Loader for Next.js documentation (MDX format)."""
    
    def __init__(
        self,
        docs_root: Path,
        version: str = "14",
        config: Optional[LoaderConfig] = None
    ):
        super().__init__(
            docs_root=docs_root,
            framework_name="nextjs",
            version=version,
            base_url="https://nextjs.org/docs",
            config=config
        )
    
    def get_supported_extensions(self) -> List[str]:
        return ['.mdx', '.md']
    
    def preprocess_content(self, text: str) -> str:
        """Remove MDX-specific syntax."""
        # Remove import statements
        text = re.sub(r'^import .+$', '', text, flags=re.MULTILINE)
        
        # Remove export statements
        text = re.sub(r'^export .+$', '', text, flags=re.MULTILINE)
        
        # Remove React directives
        text = re.sub(r"'use (client|server)';?", '', text)
        
        # Clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


class ReactDocsLoader(BaseDocsLoader):
    """Loader for React documentation."""
    
    def __init__(
        self,
        docs_root: Path,
        version: str = "18",
        config: Optional[LoaderConfig] = None
    ):
        # Handle src/content subdirectory if it exists (standard in react.dev repo)
        content_dir = docs_root / "src/content"
        if content_dir.exists():
            docs_root = content_dir
            
        super().__init__(
            docs_root=docs_root,
            framework_name="react",
            version=version,
            base_url="https://react.dev",
            config=config
        )
    
    def get_supported_extensions(self) -> List[str]:
        return ['.md', '.mdx']
    
    def preprocess_content(self, text: str) -> str:
        """Remove React doc-specific syntax."""
        text = re.sub(r'^import .+$', '', text, flags=re.MULTILINE)
        text = re.sub(r"'use [^']+';?", '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class PythonDocsLoader(BaseDocsLoader):
    """
    Loader for Python documentation (RST/MD).
    Capable of reading processed markdown or raw RST from CPython docs.
    """
    
    def __init__(
        self,
        docs_root: Path,
        version: str = "3.12",
        config: Optional[LoaderConfig] = None
    ):
        # Handle Doc subdirectory if it exists (standard in cpython repo)
        content_dir = docs_root / "Doc"
        if content_dir.exists():
            docs_root = content_dir
            
        super().__init__(
            docs_root=docs_root,
            framework_name="python",
            version=version,
            base_url="https://docs.python.org/3",
            config=config
        )
    
    def get_supported_extensions(self) -> List[str]:
        # Support both RST (official) and MD (often used in tutorials)
        return ['.rst', '.md', '.txt']
    
    def preprocess_content(self, text: str) -> str:
        """
        Preprocess Python docs.
        If RST, we might want to strip some directives.
        """
        # Simple cleanup for now
        # Remove Sphinx directives like .. toctree:: or .. code-block::
        text = re.sub(r'^\.\.\s+\w+::.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# ============================================================================
# Generic Configuration-Driven Loader
# ============================================================================

class GenericDocsLoader(BaseDocsLoader):
    """
    Universal loader driven by YAML configuration.
    
    Add new frameworks by editing frameworks.yaml - no code changes needed!
    """
    
    def __init__(
        self,
        framework_config: Dict[str, any],
        docs_root: Optional[Path] = None,
        config: Optional[LoaderConfig] = None
    ):
        # CRITICAL: Set framework_config FIRST because parent __init__ calls get_supported_extensions()
        self.framework_config = framework_config
        
        # Determine docs root
        if docs_root is None:
            docs_root = Path(framework_config["docs_path"])
        else:
            docs_root = Path(docs_root)
        
        # Ensure docs_root exists before proceeding
        if not docs_root.exists():
            raise FileNotFoundError(f"Documentation root not found: {docs_root}")
        
        # Handle subdirectory
        subdirectory = framework_config.get("subdirectory")
        if subdirectory:
            subdir_path = docs_root / subdirectory
            if subdir_path.exists():
                docs_root = subdir_path
            else:
                logger.warning(f"Subdirectory not found: {subdir_path}, using {docs_root}")
        
        # Initialize parent with validated path
        super().__init__(
            docs_root=docs_root,
            framework_name=framework_config["key"],
            version=framework_config.get("version", "latest"),
            base_url=framework_config.get("base_url", ""),
            config=config
        )
    
    def get_supported_extensions(self) -> List[str]:
        return self.framework_config.get("extensions", [".md"])
    
    def preprocess_content(self, text: str) -> str:
        preprocessing = self.framework_config.get("preprocessing", {})
        
        # Apply regex patterns
        for pattern in preprocessing.get("remove_patterns", []):
            try:
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
            except re.error as e:
                logger.warning(f"Invalid regex: {e}")
        
        # Clean whitespace
        if preprocessing.get("clean_whitespace", True):
            text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


# ============================================================================
# Loader Factory
# ============================================================================

class LoaderFactory:
    """Configuration-driven factory for creating framework loaders."""
    
    # Legacy hardcoded loaders (deprecated)
    LOADERS = {
        "nextjs": NextJsDocsLoader,
        "react": ReactDocsLoader,
        "python": PythonDocsLoader,
    }
    
    _config_cache: Optional[Dict] = None
    
    @classmethod
    def load_frameworks_config(cls, config_path: str = "frameworks.yaml") -> Dict:
        """Load framework configurations from YAML."""
        if cls._config_cache:
            return cls._config_cache
        
        # Try absolute path execution relative from project root
        project_root = Path(__file__).resolve().parents[2]
        config_file = project_root / config_path
        
        if not config_file.exists():
             # Fallback to CWD just in case
             config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cls._config_cache = config
        logger.info(f"Loaded {len(config['frameworks'])} frameworks")
        return config
    
    @classmethod
    def get_available_frameworks(cls) -> List[str]:
        """Get list of available frameworks."""
        config = cls.load_frameworks_config()
        return list(config["frameworks"].keys())
    
    @classmethod
    def create(
        cls,
        framework: str,
        docs_root: Optional[Path] = None,
        version: str = "latest",
        config: Optional[LoaderConfig] = None,
        use_yaml: bool = True  # Use YAML config for flexibility
    ) -> BaseDocsLoader:
        """
        Create loader for framework.
        
        Args:
            framework: Framework name (nextjs, react, etc.)
            docs_root: Path to documentation root
            version: Documentation version
            config: LoaderConfig instance
            
        Returns:
            Framework-specific loader instance
            
        Example:
        --------
        loader = LoaderFactory.create(
            framework="nextjs",
            docs_root=Path("nextjs-docs"),
            version="14"
        )
        
        raw_docs = loader.load()
        """
        # Try YAML config first (new way)
        if use_yaml:
            try:
                frameworks_config = cls.load_frameworks_config()
                if framework in frameworks_config["frameworks"]:
                    fw_config = frameworks_config["frameworks"][framework]
                    return GenericDocsLoader(fw_config, docs_root, config)
            except (FileNotFoundError, KeyError):
                logger.warning("YAML config not found, falling back to legacy loaders")
        
        # Fallback to legacy hardcoded loaders
        if framework not in cls.LOADERS:
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Available: {list(cls.LOADERS.keys())}"
            )
        
        loader_class = cls.LOADERS[framework]
        return loader_class(docs_root, version=version, config=config)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load Next.js documentation
    docs_path = Path("./nextjs-docs")  # Replace with actual path
    
    if docs_path.exists():
        loader = LoaderFactory.create(
            framework="nextjs",
            docs_root=docs_path,
            version="14"
        )
        
        raw_docs = loader.load()
        
        if raw_docs:
            print(f"\n{'='*70}")
            print("SAMPLE DOCUMENT")
            print('='*70)
            
            doc = raw_docs[0]
            print(f"Source: {doc.metadata['source']}")
            print(f"File: {doc.metadata['file_path']}")
            print(f"URL: {doc.metadata['url']}")
            print(f"Has code: {doc.metadata['has_code']}")
            print(f"Languages: {doc.metadata['code_languages']}")
            print(f"\nContent preview:")
            print(doc.content[:300] + "...")
    else:
        print(f"Documentation path not found: {docs_path}")
        print("Please provide a valid documentation directory")