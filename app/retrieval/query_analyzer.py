"""
Production query analyzer with validation, caching, and observability.
"""

import asyncio
import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
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
    """Validated extracted query with computed fields."""
    
    goal: Optional[str] = Field(None, min_length=3, max_length=500)
    intent: Optional[str] = Field(None, pattern=get_intent_pattern())
    depth: Optional[str] = Field(None, pattern=get_depth_pattern())
    proficiency: Optional[str] = Field(None, pattern=get_proficiency_pattern())
    tech_stack: Optional[List[str]] = Field(None, max_length=10)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    
    # Metadata
    extraction_time: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    
    @field_validator('tech_stack')
    @classmethod
    def validate_tech_stack(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and deduplicate tech stack."""
        if not v:
            return v
        
        # Deduplicate
        seen = set()
        unique = []
        for item in v:
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                unique.append(item)
        
        # Validate each item
        if len(unique) > 10:
            raise ValueError("tech_stack cannot have more than 10 items")
        
        for item in unique:
            if len(item) < 2 or len(item) > 50:
                raise ValueError(f"Invalid tech_stack item: {item}")
        
        return unique
    
    @field_validator('goal')
    @classmethod
    def validate_goal(cls, v: Optional[str]) -> Optional[str]:
        """Validate goal quality."""
        if not v:
            return v
        
        # Check for meaningful content
        if len(v.split()) < 2:
            raise ValueError("Goal must contain at least 2 words")
        
        return v.strip()
    
    @computed_field
    @property
    def required_fields(self) -> List[str]:
        """List of required fields for completeness."""
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

    @field_validator('proficiency')
    @classmethod
    def validate_proficiency_level(cls, v: Optional[str]) -> Optional[str]:
        """Normalize and validate proficiency."""
        if not v:
            return v
            
        from app.core.constants import normalize_proficiency
        normalized = normalize_proficiency(v)
        
        if normalized != v.lower().strip():
            logger.warning(f"Proficiency normalized: '{v}' -> '{normalized}'")
            
        return normalized
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if all required fields are present."""
        return len(self.missing_fields) == 0
    
    @computed_field
    @property
    def quality_score(self) -> float:
        """
        Calculate query quality score (0-1).
        
        Factors:
        - Completeness (40%)
        - Confidence (30%)
        - Goal clarity (20%)
        - Tech stack specificity (10%)
        """
        # Completeness score
        completeness = (len(self.required_fields) - len(self.missing_fields)) / len(self.required_fields)
        
        # Goal clarity (word count heuristic)
        goal_clarity = min(1.0, len(self.goal.split()) / 10) if self.goal else 0.0
        
        # Tech stack specificity
        tech_specificity = min(1.0, len(self.tech_stack) / 3) if self.tech_stack else 0.0
        
        # Weighted sum
        return (
            0.4 * completeness +
            0.3 * self.confidence +
            0.2 * goal_clarity +
            0.1 * tech_specificity
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.goal:
            return "No goal extracted"
        
        parts = [f"Goal: {self.goal[:50]}..."]
        if self.intent:
            parts.append(f"Intent: {self.intent}")
        if self.depth:
            parts.append(f"Depth: {self.depth}")
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
# Session Management
# ============================================================================

class SessionStore:
    """
    Production session store with TTL and cleanup.
    
    Improvements:
    - Session expiration
    - Memory limits
    - Background cleanup
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 1000):
        self._store: Dict[str, Tuple[InMemoryChatMessageHistory, float]] = {}
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._cleanup_task = None
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session with TTL update."""
        now = datetime.now().timestamp()
        
        if session_id in self._store:
            history, _ = self._store[session_id]
            self._store[session_id] = (history, now)  # Update access time
        else:
            # Evict if needed
            if len(self._store) >= self.max_sessions:
                self._evict_lru()
            
            history = InMemoryChatMessageHistory()
            self._store[session_id] = (history, now)
        
        return self._store[session_id][0]
    
    def clear_session(self, session_id: str) -> None:
        """Clear specific session."""
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"Session cleared: {session_id}")
    
    def clear_all(self) -> None:
        """Clear all sessions."""
        self._store.clear()
        logger.info("All sessions cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used session."""
        if not self._store:
            return
        
        lru_session = min(self._store.items(), key=lambda x: x[1][1])
        session_id, _ = lru_session
        del self._store[session_id]
        logger.debug(f"Evicted LRU session: {session_id}")
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = datetime.now().timestamp()
        expired = [
            sid for sid, (_, timestamp) in self._store.items()
            if now - timestamp > self.ttl
        ]
        
        for sid in expired:
            del self._store[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    async def start_cleanup_task(self, interval_seconds: int = 300):
        """Start background cleanup task."""
        while True:
            await asyncio.sleep(interval_seconds)
            self.cleanup_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "active_sessions": len(self._store),
            "max_sessions": self.max_sessions,
            "ttl_seconds": self.ttl
        }


# ============================================================================
# Query Analyzer
# ============================================================================

@lru_cache(maxsize=1)
def _get_system_message() -> str:
    """Cached system message."""
    return """You are an intelligent query analyzer for a technical learning roadmap system.
Extract structured information from user input while maintaining conversation context.

## Extract these fields into valid JSON:

1. "goal": The user's complete learning objective (string or null if input is gibberish/irrelevant)
   - Examples: "learn Python", "build a portfolio website", "master React hooks"
   - If user says "learn python intermediate", goal should be "learn Python at intermediate level"
   - If unclear, use a generic goal like "learn programming" or "build a project"
2. "intent": "build" or "learn" (string or null)
3. "depth": "conceptual", "practical", or "balanced" (string or null)
4. "proficiency": "beginner", "intermediate", or "expert" (string or null)
5. "tech_stack": Technologies mentioned (list of strings or null)
6. "confidence": Score 0-1 based on completeness (float)

## CRITICAL RULES:
1. CONTEXT: Merge with history ONLY if current input is a refinement (e.g., "add typescript", "make it advanced").
2. RESET: If input is a NEW GOAL (e.g. "learn cooking"), UNRELATED, or RANDOM, IGNORE history and start fresh.
3. GIBBERISH: If input is meaningless (e.g. "asdf", "kjsd"), set "goal": null and "confidence": 0.
4. PERSISTENCE: Keep fields (like proficiency) only if they still apply to the new goal.
5. INFERENCE: Make reasonable inferences from context (only if related).
6. CONFIDENCE: Set based on completeness (all fields = 1.0).

## Output Format (JSON only):
{{
  "goal": "learn Python at intermediate level",
  "intent": "learn",
  "depth": "balanced",
  "proficiency": "intermediate",
  "tech_stack": ["Python"],
  "confidence": 0.9
}}

Output ONLY valid JSON, no markdown, no explanation."""


class QueryAnalyzer:
    """
    Production query analyzer with caching and validation.
    
    Improvements:
    - Query result caching
    - Validation with Pydantic
    - Metrics tracking
    - Session management with TTL
    - Background cleanup
    """
    
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
        
        # Lazy initialization
        self._chain = None
        self._chain_with_history = None
        
        # Metrics
        self._metrics = {
            "analyses": 0,
            "cache_hits": 0,
            "validation_errors": 0,
            "avg_confidence": 0.0,
            "avg_quality": 0.0
        }
        
        logger.info(f"QueryAnalyzer initialized (session: {session_id})")
    
    @property
    def chain(self):
        """Lazy chain initialization."""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain
    
    @property
    def chain_with_history(self):
        """Lazy chain with history."""
        if self._chain_with_history is None:
            self._chain_with_history = RunnableWithMessageHistory(
                self.chain,
                self.session_store.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
        return self._chain_with_history
    
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
        4. Update metrics
        """
        self._metrics["analyses"] += 1
        
        try:
            # Invoke chain
            response = await asyncio.to_thread(
                self.chain_with_history.invoke,
                {"input": query},
                config={"configurable": {"session_id": self.session_id}}
            )
            
            
            content = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Query analyzer LLM response: {content[:500]}")
            
            # Parse and validate
            from app.core.parsers import QueryOutputParser
            parser = QueryOutputParser(pydantic_object=ExtractedQuery)
            result = parser.parse(content)
            logger.info(f"Parsed result: {result}")
            
            # Create validated model
            extracted = ExtractedQuery(
                **result,
                session_id=self.session_id
            )
            
            # Recalculate confidence if needed
            if extracted.confidence == 0.0:
                required_present = len(extracted.required_fields) - len(extracted.missing_fields)
                extracted.confidence = required_present / len(extracted.required_fields)
            
            # Update metrics
            self._update_metrics(extracted)
            
            # Log
            status = "✅" if extracted.is_complete else "⚠️"
            logger.info(
                f"{status} Query analyzed: {extracted.summary()} "
                f"(confidence: {extracted.confidence:.2f}, quality: {extracted.quality_score:.2f})"
            )
            
            if extracted.missing_fields:
                logger.debug(f"Missing: {', '.join(extracted.missing_fields)}")
            
            return extracted
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            self._metrics["validation_errors"] += 1
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
    
    def _update_metrics(self, extracted: ExtractedQuery) -> None:
        """Update rolling metrics."""
        analyses = self._metrics["analyses"]
        
        # Exponential moving average
        alpha = 0.1
        self._metrics["avg_confidence"] = (
            alpha * extracted.confidence +
            (1 - alpha) * self._metrics["avg_confidence"]
        )
        self._metrics["avg_quality"] = (
            alpha * extracted.quality_score +
            (1 - alpha) * self._metrics["avg_quality"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get analyzer metrics."""
        return {
            **self._metrics,
            "session_stats": self.session_store.get_stats()
        }
    
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