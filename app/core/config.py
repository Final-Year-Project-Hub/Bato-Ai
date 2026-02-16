"""
Production-ready configuration with secrets management and validation.
Follows 12-factor app principles with environment-based configuration.
"""

import logging
import os
import secrets
from pathlib import Path
from typing import Optional, Dict, List, Any
from functools import lru_cache

from pydantic import Field, field_validator, computed_field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings with strict validation and secrets management.
    
    Features:
    - Secrets are masked in logs/repr
    - Computed fields for derived values
    - Strict validation with helpful errors
    - Type safety with Pydantic v2
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        # Validate on assignment for runtime safety
        validate_assignment=True
    )
    
    # ========================================================================
    # ENVIRONMENT
    # ========================================================================
    ENV: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # ========================================================================
    # SECURITY
    # ========================================================================
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    API_KEY_HEADER: str = Field(default="X-API-Key")
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000","http://localhost:8000"])
    ADMIN_API_KEY: SecretStr = Field(default=SecretStr("super-secret-admin-key"), description="Admin API Key for ingestion")
    
    # ========================================================================
    # HUGGING FACE / LLM (SECRETS)
    # ========================================================================
    HUGGINGFACE_API_TOKEN: SecretStr = Field(..., description="HF API token (required)")
    LLM_PROVIDER: str = Field(default="huggingface", pattern="^(huggingface|openai)$")
    
    # ========================================================================
    # MULTI-MODEL CONFIGURATION
    # ========================================================================


    # Model Selection
    QUERY_ANALYSIS_MODEL: str = Field(default="llama-3.1-8b-instant")
    GENERATION_MODEL: str = Field(default="llama-3.1-8b-instant")

    # GENERATION_MODEL=Qwen/Qwen2.5-7B-Instruct


    
    # ========================================================================
    # LLM GENERATION SETTINGS
    # ========================================================================
    LLM_MAX_TOKENS: int = Field(default=3072, ge=256, le=8192)
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    LLM_TIMEOUT: int = Field(default=300, ge=5, le=600)
    LLM_CACHE_ENABLED: bool = Field(default=True)
    LLM_CACHE_SIZE: int = Field(default=1000, ge=0, le=10000)

    # ========================================================================
    # REDIS CACHE (Required for Production)
    # ========================================================================
    REDIS_ENABLED: bool = Field(default=True, description="Enable Redis caching")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password (if required)")
    REDIS_DB: int = Field(default=0, ge=0, le=15, description="Redis database number")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, ge=1, le=200, description="Max connections in pool")
    REDIS_SOCKET_TIMEOUT: int = Field(default=5, ge=1, le=30, description="Socket timeout in seconds")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(default=5, ge=1, le=30, description="Connect timeout in seconds")
    
    # Cache TTLs (in seconds)
    ROADMAP_CACHE_TTL: int = Field(default=86400, ge=60, description="Roadmap cache TTL (24 hours)")
    LLM_CACHE_TTL: int = Field(default=604800, ge=60, description="LLM cache TTL (7 days)")

    # ========================================================================
    # GROQ API (for Topic Deep-Dive)
    # ========================================================================
    GROQ_API_KEY: SecretStr = Field(..., description="Groq API key for fast inference")
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant", description="Groq model ID")
    GROQ_MAX_TOKENS: int = Field(default=6144, ge=512, le=8192, description="Max tokens for Groq")
    GROQ_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0, description="Groq temperature")
    GROQ_TIMEOUT: int = Field(default=120, ge=10, le=300, description="Groq timeout in seconds")

    # ========================================================================
    # QDRANT
    # ========================================================================
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[SecretStr] = Field(default=None)
    QDRANT_COLLECTION_NAME: str = Field(default="framework_docs")
    RECREATE_COLLECTION: bool = Field(default=False)
    
    # Connection pooling
    QDRANT_TIMEOUT: int = Field(default=30, ge=5, le=120)
    QDRANT_PREFER_GRPC: bool = Field(default=False)
    
    # ========================================================================
    # EMBEDDINGS
    # ========================================================================
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    EMBEDDING_PROVIDER: str = Field(default="api", pattern="^(local|api)$")
    EMBEDDING_DEVICE: str = Field(default="cpu", pattern="^(cuda|cpu|mps)$")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=128)
    
    # ========================================================================
    # DOCUMENT CHUNKING
    # ========================================================================
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=4000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=500)
    TARGET_TOKENS: int = Field(default=500, ge=100, le=2000)
    MAX_TOKENS_PER_CHUNK: int = Field(default=2000, ge=500, le=4000)
    PRESERVE_CODE_BLOCKS: bool = Field(default=True)
    
    # ========================================================================
    # RETRIEVAL SETTINGS
    # ========================================================================
    RETRIEVAL_MAX_CANDIDATES: int = Field(default=50, ge=5, le=200)
    RETRIEVAL_SCORE_THRESHOLD: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Document Restriction
    MIN_DOCS_REQUIRED: int = Field(default=4, ge=0, le=20, description="Minimum documents required for roadmap generation")
    STRICT_DOCUMENT_MODE: bool = Field(default=True, description="Enforce strict document-only responses")
    
    # ========================================================================
    # TOKEN BUDGET
    # ========================================================================
    TOKEN_BUDGET_CONSERVATIVE: bool = Field(default=True)
    TOKEN_BUDGET_MAX: int = Field(default=16384, ge=4096, le=65536)
    
    # ========================================================================
    # INGESTION SETTINGS
    # ========================================================================
    BATCH_SIZE: int = Field(default=50, ge=1, le=200)
    SHOW_PROGRESS: bool = Field(default=True)
    LOG_STATS: bool = Field(default=True)
    
    # ========================================================================
    # API SERVER
    # ========================================================================
    API_HOST: str = Field(default="0.0.0.0", validation_alias="HOST")
    API_PORT: int = Field(default=8000, ge=1024, le=65535, validation_alias="PORT")
    API_WORKERS: int = Field(default=4, ge=1, le=16, validation_alias="WEB_CONCURRENCY")
    API_RELOAD: bool = Field(default=False)  # Hot reload in dev
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=False)  # Disabled for development
    RATE_LIMIT_REQUESTS: int = Field(default=60, ge=1)
    RATE_LIMIT_WINDOW: int = Field(default=60, ge=1)  # seconds
    

    
    # ========================================================================
    # FRAMEWORKS CONFIGURATION
    # ========================================================================
    
    @computed_field
    @property
    def FRAMEWORKS(self) -> Dict[str, Dict[str, str]]:
        """
        Load framework configurations from frameworks.yaml.
        Falls back to hardcoded defaults if YAML not found.
        """
        try:
            import yaml
            from pathlib import Path
            
            config_file = Path("frameworks.yaml")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Convert to simplified format for backward compatibility
                frameworks = {}
                for key, fw_config in config.get("frameworks", {}).items():
                    frameworks[key] = {
                        "path": fw_config.get("docs_path", f"docs/{key}"),
                        "collection": fw_config.get("collection", "framework_docs")
                    }
                return frameworks
        except Exception as e:
            logger.warning(f"Could not load frameworks.yaml: {e}, using defaults")
        
        # Fallback to hardcoded defaults
        return {
            "react": {"path": "docs/react", "collection": "framework_docs"},
            "python": {"path": "docs/python", "collection": "framework_docs"},
            "nextjs": {"path": "docs/nextjs", "collection": "framework_docs"}
        }
    
    @computed_field
    @property
    def FRAMEWORK_URL_PATTERNS(self) -> Dict[str, Dict[str, Any]]:
        """
        Load framework URL patterns from frameworks.yaml.
        Falls back to hardcoded defaults if YAML not found.
        """
        try:
            import yaml
            from pathlib import Path
            
            config_file = Path("frameworks.yaml")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Extract URL patterns from framework configs
                patterns = {}
                for key, fw_config in config.get("frameworks", {}).items():
                    url_config = fw_config.get("url_config", {})
                    patterns[key] = {
                        "base_url": fw_config.get("base_url", ""),
                        "path_prefix": url_config.get("path_prefix", f"{key}/"),
                        "strip_numeric_prefix": url_config.get("strip_numeric_prefix", False),
                        "file_extensions": fw_config.get("extensions", [".md"]),
                        "replace_extensions": url_config.get("replace_extensions", {})
                    }
                return patterns
        except Exception as e:
            logger.warning(f"Could not load framework URL patterns: {e}, using defaults")
        
        # Fallback to hardcoded defaults
        return {
            "nextjs": {
                "base_url": "https://nextjs.org/docs",
                "path_prefix": "nextjs/",
                "strip_numeric_prefix": True,
                "file_extensions": [".mdx", ".md"]
            },
            "react": {
                "base_url": "https://react.dev/learn",
                "path_prefix": "react/",
                "strip_numeric_prefix": False,
                "file_extensions": [".md", ".mdx"]
            },
            "python": {
                "base_url": "https://docs.python.org/3",
                "path_prefix": "python/",
                "strip_numeric_prefix": False,
                "file_extensions": [".rst", ".md", ".txt"],
                "replace_extensions": {".rst": ".html"}
            }
        }
    
    # ========================================================================
    # COMPUTED FIELDS
    # ========================================================================
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENV == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENV == "development"
    
    @computed_field
    @property
    def docs_base_path(self) -> Path:
        """Base path for documentation storage."""
        return Path("docs")
    
    # ========================================================================
    # VALIDATORS
    # ========================================================================
    @field_validator("EMBEDDING_DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate CUDA availability."""
        if v == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return v
    
    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("CHUNK_SIZE", 1000)
        if v >= chunk_size:
            raise ValueError(f"CHUNK_OVERLAP ({v}) must be < CHUNK_SIZE ({chunk_size})")
        return v
    
    @field_validator("HUGGINGFACE_API_TOKEN")
    @classmethod
    def validate_hf_token(cls, v: SecretStr) -> SecretStr:
        """Validate HF token format."""
        token = v.get_secret_value()
        if not token or not token.startswith("hf_"):
            raise ValueError("Invalid HuggingFace API token format (must start with 'hf_')")
        return v
    
    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins."""
        if not v:
            raise ValueError("ALLOWED_ORIGINS cannot be empty")
        return v
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    def get_hf_token(self) -> str:
        """Get HuggingFace API token as plain string."""
        return self.HUGGINGFACE_API_TOKEN.get_secret_value()
    
    def get_qdrant_key(self) -> Optional[str]:
        """Get Qdrant API key as plain string."""
        return self.QDRANT_API_KEY.get_secret_value() if self.QDRANT_API_KEY else None
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL if enabled."""
        if not self.REDIS_ENABLED:
            return None
        return self.REDIS_URL
    
    def __repr__(self) -> str:
        """Safe repr without exposing secrets."""
        return (
            f"Settings(env={self.ENV}, "
            f"llm={self.GENERATION_MODEL}, "
            f"embedding_device={self.EMBEDDING_DEVICE}, "
            f"debug={self.DEBUG})"
        )


# ============================================================================
# SINGLETON PATTERN WITH CACHING
# ============================================================================

@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).
    
    Benefits:
    - Settings loaded once and reused
    - Thread-safe with lru_cache
    - Easy to mock in tests
    
    Usage:
        from app.core.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Backward compatibility
settings = get_settings()


if __name__ == "__main__":
    # Validate configuration
    try:
        config = get_settings()
        print("✅ Configuration validated successfully!")
        print(f"\nEnvironment: {config.ENV}")
        print(f"Debug: {config.DEBUG}")
        print(f"API Port: {config.API_PORT}")
        print(f"Embedding Device: {config.EMBEDDING_DEVICE}")
        print(f"Generation Model: {config.GENERATION_MODEL}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise
