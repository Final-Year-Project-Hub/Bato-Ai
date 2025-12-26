from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Bato-Ai Intelligent Roadmap Generator"
    API_V1_STR: str = "/api/v1"
    
    # Qdrant Config
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "nextjs_docs"
    
    # LLM Config
    LLM_PROVIDER: str = "openai" # "openai" or "huggingface"
    
    # OpenAI / DeepSeek API (Standard)
    DEEPSEEK_API_KEY: Optional[str] = None
    MODEL_NAME: str = "deepseek-chat"
    
    # Hugging Face Inference API
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    HF_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct" # Default to user's requested model
    
    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5" # Default to small for lighter local dev, can switch to bge-m3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
