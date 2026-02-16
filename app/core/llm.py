"""
Simplified LLM wrapper for Hugging Face Inference API with Redis caching.
"""

import asyncio
import hashlib
import logging
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from huggingface_hub import AsyncInferenceClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field, PrivateAttr
from app.core.config import settings
from app.core.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    HUGGINGFACE = "huggingface"


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: LLMProvider = LLMProvider.HUGGINGFACE
    model_id: str = settings.GENERATION_MODEL
    api_token: str = ""
    
    # Generation
    max_tokens: int = 3072
    temperature: float = 0.2
    top_p: float = 0.9
    timeout: int = 360
    
    # Cachinge
    enable_caching: bool = True
    cache_size: int = 1000
    
    # System prompt
    system_prompt: str = (
        "You are a Senior Technical Architect creating learning roadmaps. "
        "Provide structured, actionable guidance with time estimates."
    )


class LLMResponse(BaseModel):
    """Structured LLM response with metadata."""
    content: str = Field(..., description="Generated content")
    tokens_used: int = Field(default=0)
    cache_hit: bool = Field(default=False)
    latency_ms: int = Field(default=0)
    model: str = Field(default="")


class BatoLLM(BaseChatModel):
    """
    Simplified LLM wrapper for Hugging Face Inference API.
    
    Features:
    - Simple retry logic (3 attempts)
    - Response caching
    - Basic metrics
    """
    
    config: LLMConfig = Field(default_factory=LLMConfig)
    _redis: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    _metrics: Dict[str, int] = PrivateAttr(default_factory=lambda: {
        "requests": 0,
        "cache_hits": 0,
        "errors": 0
    })
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        config = config or LLMConfig()
        super().__init__(config=config, **kwargs)
        
        # Initialize Redis cache
        try:
            self._redis = get_redis_cache()
            logger.info("✅ LLM Redis cache initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis cache: {e}")
            raise
        
        # Initialize Hugging Face Inference Client
        self._client = AsyncInferenceClient(
            token=config.api_token,
            model=config.model_id
        )
        logger.info(f"✅ BatoLLM initialized with HuggingFace: {config.model_id}")
    
    @property
    def _llm_type(self) -> str:
        return "bato-llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
    
    def _build_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to prompt string."""
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"System: {msg.content}\n\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"User: {msg.content}\n\n"
            elif isinstance(msg, AIMessage):
                prompt += f"Assistant: {msg.content}\n\n"
            else:
                prompt += f"{msg.content}\n\n"
        return prompt.strip()
    
    def _format_mistral_prompt(self, messages: List[BaseMessage]) -> str:
        """Format messages using Mistral's instruct template."""
        formatted = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Mistral doesn't have explicit system role, prepend to first user message
                formatted += f"{msg.content}\n\n"
            elif isinstance(msg, HumanMessage):
                formatted += f"[INST] {msg.content} [/INST]"
            elif isinstance(msg, AIMessage):
                formatted += f" {msg.content}</s>"
            else:
                formatted += f"[INST] {msg.content} [/INST]"
        return formatted
    
    async def _call_with_retry(
        self,
        func,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ):
        """Simple exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s due to error: {repr(e)}")
                await asyncio.sleep(delay)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate for ChatModel."""
        start_time = time.time()
        self._metrics["requests"] += 1
        
        # Build prompt
        prompt = self._build_prompt(messages)
        
        # Check Redis cache
        cache_key = f"llm:{hashlib.md5(str(prompt).encode()).hexdigest()}"
        if self.config.enable_caching:
            try:
                cached_content = self._redis.get(cache_key)
                if cached_content:
                    self._metrics["cache_hits"] += 1
                    logger.debug(f"✅ Cache hit for prompt hash {cache_key[:12]}")
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=cached_content))])
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        
        # API call with retry
        async def _api_call():
             # Build formatted prompt for text generation
             formatted_prompt = self._format_mistral_prompt(messages)
             logger.info(f"🤖 BatoLLM calling HuggingFace with model: {self.config.model_id}")
             
             # Use text_generation instead of chat_completion
             response = await asyncio.wait_for(
                 self._client.text_generation(
                     prompt=formatted_prompt,
                     max_new_tokens=self.config.max_tokens,
                     temperature=self.config.temperature,
                     top_p=self.config.top_p,
                     stop_sequences=stop or [],
                     return_full_text=False
                 ),
                 timeout=self.config.timeout
             )
             if(response):
                logger.info(f"🤖 BatoLLM response extracted")
             
             return response
        
        try:
            # Execute with retry
            content = await self._call_with_retry(_api_call)
            
            # Cache in Redis
            if self.config.enable_caching:
                try:
                    from app.core.config import get_settings
                    ttl = get_settings().LLM_CACHE_TTL
                    self._redis.set(cache_key, content, ttl=ttl)
                    logger.debug(f"Cached LLM response with TTL={ttl}s")
                except Exception as e:
                    logger.warning(f"Cache set failed: {e}")
            
            # Log metrics
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"✅ LLM call: {latency_ms}ms, {len(content)} chars")
            
            # Return result
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"❌ LLM call failed: {repr(e)}")
            raise
            
    async def astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Async stream for ChatModel."""
        self._metrics["requests"] += 1
        
        # Build formatted prompt for text generation
        formatted_prompt = self._format_mistral_prompt(messages)
        logger.info(f"🤖 BatoLLM streaming from HuggingFace with model: {self.config.model_id}")
        
        try:
            # Use text_generation with streaming
            async for token in self._client.text_generation(
                prompt=formatted_prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop_sequences=stop or [],
                stream=True,
                details=False,
                return_full_text=False
            ):
                if token:
                    yield ChatGeneration(message=AIMessage(content=token))
                    
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"❌ LLM stream failed: {repr(e)}")
            raise
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync generate."""
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
    
    def _convert_messages(self, messages: list) -> list:
        """Convert LangChain messages to API format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, dict):
                converted.append(msg)
            else:
                converted.append({
                    "role": "user",
                    "content": msg.content if hasattr(msg, "content") else str(msg)
                })
        
        return converted
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        try:
            redis_stats = self._redis.get_stats()
            cache_size = redis_stats.get("total_keys", 0)
        except:
            cache_size = 0
        
        return {
            **self._metrics,
            "cache_size": cache_size,
            "cache_hit_rate": (
                self._metrics["cache_hits"] / self._metrics["requests"]
                if self._metrics["requests"] > 0 else 0.0
            ),
            "cache_backend": "redis"
        }
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        try:
            # Clear only LLM keys (prefix: llm:)
            # Note: This is a simplified version. For production, use SCAN with pattern
            logger.warning("Clear cache not fully implemented for Redis. Use Redis CLI: FLUSHDB")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
