# multi_llm.py
"""
Multi-model LLM manager for hybrid approach.
Uses different models for different tasks to optimize for speed and quality.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from app.core.llm import BatoLLM, LLMConfig, LLMProvider
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MultiModelConfig:
    """Configuration for multi-model setup."""
    # Query Analysis: Fast model for extracting user intent
    query_analysis_model: str = settings.QUERY_ANALYSIS_MODEL
    query_analysis_temperature: float = 0.1
    query_analysis_max_tokens: int = 1024
    
    # Roadmap Generation: Main model for generating roadmaps
    generation_model: str = settings.GENERATION_MODEL
    generation_temperature: float = 0.2
    generation_max_tokens: int = 3072
    
    # Common settings
    api_token: str = ""
    timeout: int = 180


class MultiModelLLMManager:
    """
    Manages multiple LLM instances for different tasks.
    
    Strategy:
    - DeepSeek V3: Query analysis (accuracy critical)
    - DeepSeek V2.5: Roadmap generation (2-3x faster)
    - DeepSeek V2.5: Validation (speed matters)
    
    Benefits:
    - 40-60% faster than using V3 for everything
    - Maintains high accuracy where it matters
    - Reduces cost
    """
    
    def __init__(
        self, 
        config: Optional[MultiModelConfig] = None, 
        api_token: Optional[str] = None
    ):
        """Initialize multi-model manager."""
        self.config = config or MultiModelConfig()
        
        if api_token:
            self.config.api_token = api_token
        
        # Initialize separate LLM instances
        self.query_analyzer_llm = self._create_llm(
            model_id=self.config.query_analysis_model,
            temperature=self.config.query_analysis_temperature,
            max_tokens=self.config.query_analysis_max_tokens,
            purpose="query_analysis"
        )
        
        self.generator_llm = self._create_llm(
            model_id=self.config.generation_model,
            temperature=self.config.generation_temperature,
            max_tokens=self.config.generation_max_tokens,
            purpose="generation"
        )
        
        logger.info("✅ MultiModelLLMManager initialized")
        logger.info(f"   Query Analysis: {self.config.query_analysis_model}")
        logger.info(f"   Generation: {self.config.generation_model}")
    
    def _create_llm(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
        purpose: str
    ) -> BatoLLM:
        """Create LLM instance with specific configuration."""
        config = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            model_id=model_id,
            api_token=self.config.api_token,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.config.timeout,
            enable_caching=True  # Re-enable caching for AsyncInferenceClient
        )
        
        llm = BatoLLM(config)
        logger.debug(f"Created {purpose} LLM: {model_id}")
        return llm
    
    def get_query_analyzer_llm(self) -> BatoLLM:
        """Get LLM for query analysis (high accuracy)."""
        return self.query_analyzer_llm
    
    def get_generator_llm(self) -> BatoLLM:
        """Get LLM for roadmap generation."""
        return self.generator_llm
    
    async def analyze_query(self, prompt: str) -> str:
        """Analyze query using query analysis model."""
        return await self.query_analyzer_llm.ainvoke(prompt)
    
    async def generate_roadmap(self, prompt: str) -> str:
        """Generate roadmap using generation model."""
        return await self.generator_llm.ainvoke(prompt)
    
    async def answer(
        self,
        query: str,
        documents: list,
        budget: Optional[Any] = None,
        include_sources: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate answer using generator LLM (for compatibility with LLMManager interface).
        Uses DeepSeek V2.5 for fast generation.
        """
        # Build context from documents
        context = self._build_context(documents)
        
        # Build prompt
        prompt = f"{query}\n\nContext:\n{context}"
        
        # Generate answer using generator LLM (V2.5)
        answer_text = await self.generator_llm.ainvoke(prompt, **kwargs)
        
        return {
            "answer": answer_text,
            "sources": self._extract_sources(documents) if include_sources else [],
            "query": query,
            "documents_used": len(documents),
            "tokens_budget": budget.answer_tokens if budget else 0,
        }
    
    def _build_context(self, documents: list) -> str:
        """Build context string from documents."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("file_path", "unknown")
            content = doc.page_content[:4000]
            context_parts.append(f"[{i}] Source: {source}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, documents: list) -> list:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            sources.append({
                "file_path": doc.metadata.get("file_path"),
                "score": doc.metadata.get("score"),
                "url": doc.metadata.get("url"),
            })
        return sources
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all models."""
        return {
            "query_analyzer": {
                "model": self.config.query_analysis_model
            },
            "generator": {
                "model": self.config.generation_model
            }
        }
    
    def clear_all_caches(self) -> None:
        """Clear caches for all models (no-op for OpenAI client)."""
        logger.info("Cache clearing not supported for OpenAI-compatible client")


# Convenience function
def create_multi_model_manager(api_token: str) -> MultiModelLLMManager:
    """Quick multi-model manager creation."""
    config = MultiModelConfig(api_token=api_token)
    return MultiModelLLMManager(config)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    import os
    
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    if not api_token:
        print("⚠️  Set HUGGINGFACE_API_TOKEN environment variable")
        exit(1)
    
    # Create manager
    manager = create_multi_model_manager(api_token)
    
    print("\n" + "=" * 70)
    print("MULTI-MODEL LLM MANAGER")
    print("=" * 70)
    print(f"\nStats: {manager.get_stats()}")
