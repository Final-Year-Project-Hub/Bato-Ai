"""
Simplified query analyzer for college project.
Removed: SessionStore complexity, quality scoring, excessive metrics.
"""

import asyncio
import hashlib
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel, Field, field_validator, computed_field

from app.retrieval.token_budget import Intent, Depth
from app.core.constants import (
    get_proficiency_pattern,
    get_intent_pattern,
    get_depth_pattern
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class ExtractedQuery(BaseModel):
    """Validated extracted query."""
    
    goal: Optional[str] = Field(None, min_length=3, max_length=500)
    intent: Optional[str] = Field(None, pattern=get_intent_pattern())
    depth: Optional[str] = Field(None, pattern=get_depth_pattern())
    proficiency: Optional[str] = Field(None, pattern=get_proficiency_pattern())
    tech_stack: Optional[List[str]] = Field(None, max_length=10)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    
    # Scope for dynamic roadmap calculation
    scope: str = Field(default="medium", pattern="^(narrow|medium|broad)$")
    
    # Metadata
    extraction_time: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    
    @field_validator('tech_stack')
    @classmethod
    def validate_tech_stack(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate tech stack."""
        if not v:
            return v
        if len(v) > 10:
            raise ValueError("tech_stack cannot have more than 10 items")
        return v
    
    @field_validator('goal')
    @classmethod
    def validate_goal(cls, v: Optional[str]) -> Optional[str]:
        """Validate goal."""
        if v:
            return v.strip()
        return v
    
    @computed_field
    @property
    def required_fields(self) -> List[str]:
        """List of required fields."""
        return ["goal", "intent", "proficiency", "tech_stack"]
    
    @computed_field
    @property
    def missing_fields(self) -> List[str]:
        """Compute missing required fields."""
        missing = []
        if not self.goal:
            missing.append("goal")
        if not self.intent:
            missing.append("intent")
        if not self.proficiency:
            missing.append("proficiency")
        if not self.tech_stack:
            missing.append("tech_stack")
        return missing
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if all required fields are present."""
        return len(self.missing_fields) == 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.goal:
            return "No goal extracted"
        
        parts = [f"Goal: {self.goal[:50]}..."]
        if self.intent:
            parts.append(f"Intent: {self.intent}")
        if self.proficiency:
            parts.append(f"Level: {self.proficiency}")
        if self.tech_stack:
            parts.append(f"Tech: {', '.join(self.tech_stack[:3])}")
        
        return " | ".join(parts)
    
    def to_cache_key(self) -> str:
        """Generate cache key for this query."""
        parts = [
            self.goal or "",
            self.intent or "",
            self.proficiency or "",
            ",".join(sorted(self.tech_stack or []))
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()


# ============================================================================
# Simplified Session Management
# ============================================================================

class SessionStore:
    """Simple session store without TTL/LRU complexity."""
    
    def __init__(self):
        self._store: Dict[str, InMemoryChatMessageHistory] = {}
    
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get or create session."""
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]
    
    def clear_session(self, session_id: str) -> None:
        """Clear specific session."""
        if session_id in self._store:
            del self._store[session_id]
    
    def clear_all(self) -> None:
        """Clear all sessions."""
        self._store.clear()


# ============================================================================
# Query Analyzer
# ============================================================================

def _get_system_message() -> str:
    """Load system message from prompt file."""
    from app.core.prompt_manager import load_prompt
    from app.core.constants import (
        get_proficiency_levels,
        get_intent_values,
        get_depth_values
    )
    
    return load_prompt(
        "query_analyzer",
        PROFICIENCY_LEVELS=', '.join(f'"{p}"' for p in get_proficiency_levels()),
        INTENTS=', '.join(f'"{i}"' for i in get_intent_values()),
        DEPTHS=', '.join(f'"{d}"' for d in get_depth_values())
    )


class QueryAnalyzer:
    """Simplified query analyzer for college project."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        session_id: str = "default",
        token_planner = None,
        session_store: Optional[SessionStore] = None
    ):
        self.llm = llm
        self.session_id = session_id
        self.token_planner = token_planner
        self.session_store = session_store or SessionStore()
        
        # Build chain immediately (no lazy loading)
        self.chain = self._build_chain()
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.session_store.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        logger.info(f"QueryAnalyzer initialized (session: {session_id})")
    
    def _build_chain(self):
        """Build extraction chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", _get_system_message()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return prompt | self.llm
    
    async def analyze_async(self, query: str) -> ExtractedQuery:
        """
        Async query analysis with validation.
        
        Pipeline:
        1. Invoke LLM with history
        2. Parse JSON response
        3. Validate with Pydantic
        """
        try:
            # Invoke chain
            response = await self.chain_with_history.ainvoke(
                {"input": query},
                config={"configurable": {"session_id": self.session_id}}
            )
            
            content = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Query analyzer LLM response: {content[:500]}")
            
            # Parse and validate
            from app.core.parsers import QueryOutputParser
            parser = QueryOutputParser(pydantic_object=ExtractedQuery)
            result = parser.parse(content)
            
            # Create validated model
            extracted = ExtractedQuery(
                **result,
                session_id=self.session_id
            )
            
            # Recalculate confidence if needed
            if extracted.confidence == 0.0:
                required_present = len(extracted.required_fields) - len(extracted.missing_fields)
                extracted.confidence = required_present / len(extracted.required_fields)
            
            # Log
            status = "✅" if extracted.is_complete else "⚠️"
            logger.info(
                f"{status} Query analyzed: {extracted.summary()} "
                f"(confidence: {extracted.confidence:.2f})"
            )
            
            if extracted.missing_fields:
                logger.debug(f"Missing: {', '.join(extracted.missing_fields)}")
            
            return extracted
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return self._fallback_extraction(query)
    
    def analyze(self, query: str) -> ExtractedQuery:
        """Sync wrapper."""
        return asyncio.run(self.analyze_async(query))
    
    def _fallback_extraction(self, query: str) -> ExtractedQuery:
        """Fallback when extraction fails."""
        return ExtractedQuery(
            goal=query[:200] if query else None,
            confidence=0.25,
            session_id=self.session_id
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session_store.clear_session(self.session_id)
    
    def switch_session(self, new_session_id: str) -> None:
        """Switch to different session."""
        self.session_id = new_session_id
        logger.info(f"Switched to session: {new_session_id}")


# Factory
def create_analyzer(
    llm: BaseChatModel,
    session_id: str = "default",
    model_max_tokens: int = 16384
) -> QueryAnalyzer:
    """Create analyzer with defaults."""
    from app.retrieval.token_budget import TokenBudgetPlanner
    planner = TokenBudgetPlanner(model_max_tokens=model_max_tokens)
    return QueryAnalyzer(llm, session_id=session_id, token_planner=planner)
