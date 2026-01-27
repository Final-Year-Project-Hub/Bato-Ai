"""
Groq LLM wrapper for fast inference with Llama models.
Optimized for topic deep-dive generation.
"""

import logging
from typing import List, Optional, AsyncIterator, Any
from groq import AsyncGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field, PrivateAttr

from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqLLM(BaseChatModel):
    """
    Groq LLM wrapper for Llama 3.1 8B.
    
    Provides fast inference for educational content generation
    with async and streaming support.
    """
    
    model_name: str = Field(default=settings.GROQ_MODEL)
    max_tokens: int = Field(default=settings.GROQ_MAX_TOKENS)
    temperature: float = Field(default=settings.GROQ_TEMPERATURE)
    timeout: int = Field(default=settings.GROQ_TIMEOUT)
    
    _client: Optional[AsyncGroq] = PrivateAttr(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = AsyncGroq(
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            timeout=self.timeout
        )
        logger.info(f"✅ GroqLLM initialized: {self.model_name}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Generate response from Groq API."""
        try:
            # Convert LangChain messages to Groq format
            groq_messages = self._convert_messages(messages)
            
            # Call Groq API
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=groq_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop,
                **kwargs
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Return in LangChain format
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=content))]
            )
        
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    async def astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncIterator[AIMessage]:
        """Stream response from Groq API."""
        try:
            # Convert messages
            groq_messages = self._convert_messages(messages)
            
            # Stream from Groq
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=groq_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop,
                stream=True,
                **kwargs
            )
            
            # Yield chunks
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield AIMessage(content=chunk.choices[0].delta.content)
        
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Groq API format."""
        groq_messages = []
        
        for msg in messages:
            # Determine role
            if isinstance(msg, SystemMessage) or msg.type == "system":
                role = "system"
            elif isinstance(msg, AIMessage) or msg.type == "ai":
                role = "assistant"
            else:  # HumanMessage or default
                role = "user"
            
            groq_messages.append({
                "role": role,
                "content": msg.content
            })
        
        return groq_messages
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "groq"
    
    def _generate(self, *args, **kwargs):
        """Sync generate not implemented - use async only."""
        raise NotImplementedError("Use async methods (_agenerate or astream)")
