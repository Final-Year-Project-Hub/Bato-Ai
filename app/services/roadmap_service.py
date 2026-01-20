"""
Production roadmap service with comprehensive error handling and observability.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document

from app.retrieval.token_budget import Intent, Depth, TokenBudgetPlanner
from app.retrieval.query_analyzer import QueryAnalyzer, ExtractedQuery
from app.retrieval.retriever import QdrantRetriever
from app.core.multi_llm import MultiModelLLMManager
from app.core.constants import get_all_suggestions
from app.core.config import get_settings
from app.core.prompt_manager import load_prompt
from app.core.exceptions import InsufficientDocumentationError, JSONParseError
from app.core.cache import get_cache

# Qdrant filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)





# ============================================================================
# Data Models (using existing from original file)
# ============================================================================


# Import correct schemas from app.schemas
from app.schemas import (
    Roadmap,
    Phase,
    Topic,
    Subtopic,
    RoadmapRequest,
    ClarificationRequest,
    ChatRequest,
    QueryIntent,
    InsufficientDocumentationError
)

# Use Roadmap for internal service response as well
RoadmapResponse = Roadmap
ClarificationResponse = ClarificationRequest


# ============================================================================
# Production Roadmap Service
# ============================================================================

class RoadmapService:
    """
    Production roadmap generation service.
    """
    
    def __init__(
        self,
        retriever: QdrantRetriever,
        analyzer: QueryAnalyzer,
        llm_manager: MultiModelLLMManager,
        token_planner: Optional[TokenBudgetPlanner] = None,
        framework_patterns: Optional[Dict[str, Dict]] = None
    ):
        self.retriever = retriever
        self.analyzer = analyzer
        self.llm_manager = llm_manager
        self.token_planner = token_planner
        self.cache = get_cache()  # Multi-level cache
        self.framework_patterns = framework_patterns or {}
        
        logger.info("RoadmapService initialized with caching enabled")
        
        # Request deduplication
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Metrics
        self._metrics = {
            "requests": 0,
            "clarifications": 0,
            "roadmaps_generated": 0,
            "errors": 0,
            "avg_generation_time_s": 0.0,
            "total_phases_generated": 0
        }
        
        logger.info("✅ RoadmapService initialized")
    
    def _build_context_with_urls(self, retrieved_docs: List[Document]) -> str:
        """
        Build context string with URLs for LLM.
        
        Formats each document with metadata headers so the LLM can extract URLs.
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            url = doc.metadata.get('url', '')
            source = doc.metadata.get('source', 'unknown')
            file_path = doc.metadata.get('file_path', '')
            
            # Build structured context with metadata
            context_parts.append(
                f"## Document {i}\n"
                f"Source: {source}\n"
                f"File: {file_path}\n"
                f"URL: {url}\n\n"
                f"{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    async def process_chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None,  # NEW
        strict_mode: Optional[bool] = None  # NEW
    ) -> Union[Roadmap, ClarificationRequest]:
        """
        Process chat with optional user context for personalization.
        
        Args:
            user_message: User's input message
            conversation_history: Previous conversation messages
            user_context: Optional user context (name, preferences, known tech)
        """
        start_time = time.time()
        self._metrics["requests"] += 1
        
        try:
            # Validate
            if not user_message or len(user_message.strip()) < 3:
                raise ValueError("Message too short")
            
            # Extract user info if provided
            user_name = None
            user_id = None
            known_tech = []
            if user_context:
                user_name = user_context.get("user_name")
                user_id = user_context.get("user_id")  # Extract user_id for cache
                known_tech = user_context.get("known_technologies", [])
                logger.info(f"Processing for user: {user_name} (ID: {user_id}), knows: {known_tech}")
            
            # Quick greeting check
            if self._is_greeting(user_message):
                self._metrics["clarifications"] += 1
                greeting_msg = f"Hello{' ' + user_name if user_name else ''}! 👋 I'm here to create your learning roadmap. What would you like to learn or build?"
                return ClarificationRequest(
                    message=greeting_msg,
                    missing_fields=["goal", "intent", "proficiency"]
                )
            
            settings = get_settings()
            
            # OPTIMIZATION: Run query analysis and preliminary search in parallel
            async with timer("parallel_analysis_retrieval"):
                # Start both operations concurrently
                analysis_task = self.analyzer.analyze_async(user_message, conversation_history)
                # Preliminary retrieval with basic query (can be refined later)
                prelim_retrieval_task = self.retriever.retrieve_async(
                    query=user_message,
                    budget=self.token_planner.plan(Intent.LEARN, Depth.BALANCED),
                    max_candidates=3  # Quick preliminary search
                )
                
                # Wait for both to complete
                extracted, prelim_docs = await asyncio.gather(
                    analysis_task,
                    prelim_retrieval_task,
                    return_exceptions=True
                )
                
                # Handle potential errors
                if isinstance(extracted, Exception):
                    logger.error(f"Query analysis failed: {extracted}")
                    raise extracted
                if isinstance(prelim_docs, Exception):
                    logger.warning(f"Preliminary retrieval failed: {prelim_docs}")
                    prelim_docs = []
            
            # Check completeness
            if not extracted.is_complete:
                self._metrics["clarifications"] += 1
                return self._create_clarification(extracted)
            
            # Check cache (with deduplication) - now user-specific
            cache_key = self.cache.generate_cache_key(
                extracted.goal,
                extracted.intent,
                extracted.proficiency,
                extracted.tech_stack,
                user_id  # Include user_id to make cache user-specific
            )
            
            # Request deduplication
            if cache_key in self._pending_requests:
                logger.info(f"Deduplicating request: {cache_key[:8]}...")
                return await self._pending_requests[cache_key]
            
            # Create future for deduplication
            future = asyncio.Future()
            self._pending_requests[cache_key] = future
            
            try:
                # Check cache first
                cached = self.cache.get(cache_key)
                if cached:
                    future.set_result(cached)
                    return cached
                
                # Generate
                roadmap = await self.generate_roadmap(
                    goal=extracted.goal,
                    intent=extracted.intent,
                    proficiency=extracted.proficiency,
                    depth=extracted.depth,
                    tech_stack=extracted.tech_stack,

                    conversation_history=conversation_history,
                    strict_mode=strict_mode
                )
                
                # Cache
                self.cache.set(cache_key, roadmap)
                
                # Update metrics
                generation_time = time.time() - start_time
                self._update_metrics(roadmap, generation_time)
                
                # Set future
                future.set_result(roadmap)
                
                return roadmap
            
            finally:
                # Cleanup
                self._pending_requests.pop(cache_key, None)
        
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Chat processing failed: {e}", exc_info=True)
            raise
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting."""
        greetings = {"hello", "hi", "hey", "hola", "greetings", "sup"}
        clean = message.lower().strip().split()[0].replace("!", "").replace(".", "")
        return clean in greetings
    
    async def _analyze_query_async(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]]
    ) -> ExtractedQuery:
        """Async query analysis with history."""
        # Simply use the analyzer directly, it handles history internally if configured
        # But we need to sync history if provided externally
        if history:
             # Basic history sync if needed, but analyzer tracks its own session
             pass
        
        return await self.analyzer.analyze_async(message)
    
    def _create_clarification(self, extracted: ExtractedQuery) -> ClarificationRequest:
        """Create clarification with smart suggestions."""
        field_prompts = {
            "goal": "What would you like to build or learn?",
            "intent": "Are you building a project or learning concepts?",
            "proficiency": "What's your experience level?",
            "tech_stack": "Which technologies are you interested in?"
        }
        
        prompts = [
            field_prompts[field]
            for field in extracted.missing_fields
            if field in field_prompts
        ]
        
        message = "To create your roadmap, I need:\n\n"
        message += "\n".join(f"{i}. {p}" for i, p in enumerate(prompts, 1))
        
        # Get suggestions from centralized constants
        suggestions = get_all_suggestions()
        
        return ClarificationRequest(
            message=message,
            missing_fields=extracted.missing_fields,
            suggested_values={
                k: v for k, v in suggestions.items()
                if k in extracted.missing_fields
            }
        )
    
    async def generate_roadmap(
        self,
        goal: str,
        intent: Optional[Union[Intent, str]] = None,
        proficiency: str = "beginner",
        depth: Optional[str] = None,
        tech_stack: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        min_phases: int = 5,
        max_phases: int = 9,
        strict_mode: Optional[bool] = None
    ) -> Roadmap:
        """
        Generate a learning roadmap based on user request and documentation.g.
        """
        # Convert intent string to Enum if needed
        if isinstance(intent, str):
            intent_val = intent.lower()
        elif hasattr(intent, 'value'):
            intent_val = intent.value
        else:
            intent_val = "learn"
            
        logger.info(
            f"🚀 Generating roadmap: goal='{goal[:50]}...', "
            f"intent={intent_val}, proficiency={proficiency}"
        )
        
        # 1. Retrieve Knowledge
        query = f"{goal} {intent_val} {' '.join(tech_stack or [])}"
        

        # Build metadata filter if specific tech stack is requested
        search_filter = None
        if tech_stack:
            # Create should conditions for each tech in stack (OR logic)
            # This matches 'source' metadata field from loaders.py
            conditions = []
            
            # Normalize tech keys using framework_patterns if available
            normalized_stack = []
            available_keys = {k.lower(): k for k in (self.framework_patterns or {}).keys()}
            
            for tech in tech_stack:
                # Normalize: remove dots, spaces, special chars
                # "Next.js" -> "nextjs", "React.js" -> "reactjs"
                tech_normalized = tech.lower().replace('.', '').replace('-', '').replace(' ', '')
                normalized_key = tech_normalized  # Default to normalized version
                
                # Check known frameworks (partial match)
                # e.g. "nextjs" matches "nextjs" key
                for key in available_keys:
                    # Remove special chars from key too for comparison
                    key_normalized = key.replace('.', '').replace('-', '').replace(' ', '')
                    if key_normalized in tech_normalized or tech_normalized in key_normalized:
                        normalized_key = available_keys[key]
                        break
                    
                normalized_stack.append(normalized_key)
                
                conditions.append(
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=normalized_key)
                    )
                )
            
            if conditions:
                search_filter = Filter(should=conditions)
                logger.info(f"Applying Qdrant filter: source in {normalized_stack}")
        
        retrieved_docs = await self.retriever.retrieve_async(
            query=query,
            budget=self.token_planner.plan(Intent(intent_val), Depth.BALANCED),
            max_candidates=5,
            filters=search_filter
        )
        
        # STRICT DOCUMENT MODE: Validate retrieval results
        settings = get_settings()
        docs_count = len(retrieved_docs)
        
        logger.info(f"Retrieved {docs_count} documents for query: {query[:50]}...")
        
        # Check if strict mode is enabled
        is_strict = strict_mode if strict_mode is not None else settings.STRICT_DOCUMENT_MODE
        
        if is_strict:
            # Zero documents - reject immediately
            if docs_count == 0:
                error_msg = (
                    f"No documentation found for {', '.join(tech_stack or ['the requested topic'])}. "
                    "Cannot generate roadmap without source material. "
                    "Please try a different technology or check if documentation is available."
                )
                logger.warning(f"Retrieval failed: {error_msg}")
                raise InsufficientDocumentationError(
                    error_msg,
                    tech_stack=tech_stack or [],
                    docs_found=0,
                    min_required=settings.MIN_DOCS_REQUIRED
                )
            
            # Below minimum threshold - warn user
            if docs_count < settings.MIN_DOCS_REQUIRED:
                logger.warning(
                    f"Low documentation coverage: {docs_count}/{settings.MIN_DOCS_REQUIRED} docs. "
                    "Roadmap quality may be reduced."
                )
                # Note: We proceed but will add warning to response metadata
        
        # Extract sources for transparency
        sources_used = list(set([
            doc.metadata.get('file_path', doc.metadata.get('source', 'unknown'))
            for doc in retrieved_docs
        ]))
        
        # Calculate retrieval confidence based on document count and scores
        avg_score = sum(doc.metadata.get('score', 0.5) for doc in retrieved_docs) / max(docs_count, 1)
        retrieval_confidence = min(1.0, (docs_count / settings.MIN_DOCS_REQUIRED) * avg_score)
        
        logger.info(
            f"📊 Retrieval Quality: {docs_count} docs, "
            f"avg_score={avg_score:.3f}, confidence={retrieval_confidence:.3f}"
        )
        
        # Build context with URLs for LLM
        context_str = self._build_context_with_urls(retrieved_docs)
        
        # 2. Load and format prompt template
        system_prompt = load_prompt(
            "roadmap_generator",
            GOAL=goal,
            INTENT=intent_val,
            PROFICIENCY=proficiency,
            TECH_STACK=', '.join(tech_stack or []),
            DOCS_COUNT=docs_count,
            CONTEXT=context_str[:4500],
            MIN_PHASES=min_phases,
            MAX_PHASES=max_phases,
            KEY_TECHNOLOGIES=json.dumps(tech_stack or [])
        )
        
        # 3. Generate with LLM
        logger.info("Calling LLM for roadmap generation...")
        response = await self.llm_manager.get_generator_llm().ainvoke(system_prompt)
        logger.info(f"LLM response type: {type(response)}")
        
        response_text = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM returned {len(response_text)} characters")
        logger.info(f"First 500 chars of LLM response: {response_text[:500]}")


        
        # 4. Parse JSON
        try:
            
            # Extract JSON from response
            json_str = response_text.strip()
            
            # Remove markdown code block markers - simple and robust approach
            # Remove opening markers
            if json_str.startswith("```json"):
                json_str = json_str[7:].lstrip()  # Remove ```json and any whitespace
            elif json_str.startswith("```"):
                json_str = json_str[3:].lstrip()  # Remove ``` and any whitespace
            
            # Remove closing markers
            if json_str.endswith("```"):
                json_str = json_str[:-3].rstrip()  # Remove trailing ``` and whitespace
            
            # Final cleanup - ensure we start with {
            json_str = json_str.strip()
            if not json_str.startswith('{'):
                # Find the first { and extract from there
                start = json_str.find('{')
                if start >= 0:
                    json_str = json_str[start:]
            
            # Check if extracted JSON is empty
            if not json_str or not json_str.strip():
                logger.error("Extracted JSON string is empty")
                raise ValueError("Failed to extract JSON from LLM response")
            
            # Parse JSON with repair attempts
            logger.info(f"Extracted JSON string length: {len(json_str)}")
            logger.info(f"First 200 chars of extracted JSON: {json_str[:200]}")
            
            # Remove trailing commas (common LLM error)
            # Pattern: comma followed by optional whitespace and then } or ]
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            try:
                data = json.loads(json_str.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {e}")
                logger.info("Attempting JSON repair...")
                
                # Attempt 1: Try to fix unterminated strings by finding the last complete object
                try:
                    # Find the last complete closing brace
                    last_brace = json_str.rfind('}')
                    if last_brace > 0:
                        repaired = json_str[:last_brace + 1]
                        logger.info(f"Attempting parse with truncated JSON (length: {len(repaired)})")
                        data = json.loads(repaired)
                        logger.info("✅ JSON repair successful - truncated to last complete object")
                    else:
                        raise e
                except json.JSONDecodeError:
                    # Attempt 2: Find last complete field (before incomplete value)
                    try:
                        # Find the last comma that separates complete fields
                        last_comma = json_str.rfind(',')
                        if last_comma > 0:
                            # Try truncating at the last comma and closing properly
                            repaired = json_str[:last_comma]
                            # Count and close braces/brackets
                            open_braces = repaired.count('{')
                            close_braces = repaired.count('}')
                            open_brackets = repaired.count('[')
                            close_brackets = repaired.count(']')
                            
                            if open_brackets > close_brackets:
                                repaired += ']' * (open_brackets - close_brackets)
                            if open_braces > close_braces:
                                repaired += '}' * (open_braces - close_braces)
                            
                            logger.info(f"Attempting parse with truncated JSON at last comma (length: {len(repaired)})")
                            data = json.loads(repaired)
                            logger.info("✅ JSON repair successful - truncated at last complete field")
                        else:
                            raise e
                    except json.JSONDecodeError:
                        # Attempt 3: Try to close any unclosed strings and objects
                        try:
                            repaired = json_str.rstrip()
                            # Count braces to see if we need to close the object
                            open_braces = repaired.count('{')
                            close_braces = repaired.count('}')
                            if open_braces > close_braces:
                                # Add missing closing braces
                                repaired += '}' * (open_braces - close_braces)
                                logger.info(f"Added {open_braces - close_braces} closing braces")
                                data = json.loads(repaired)
                                logger.info("✅ JSON repair successful - added missing braces")
                            else:
                                raise e
                        except json.JSONDecodeError:
                            logger.error(f"All JSON repair attempts failed")
                            logger.error(f"Raw response (first 1000 chars): {response_text[:1000]}")
                            logger.error(f"Extracted JSON (first 500 chars): {json_str[:500]}")
                            logger.error(f"Extracted JSON (last 500 chars): {json_str[-500:]}")
                            raise
            
            # Recalculate total hours just in case
            total_hours = sum(p.get('estimated_hours', 0) for p in data.get('phases', []))
            data['total_estimated_hours'] = total_hours
            
            # Add retrieval metadata for transparency
            data['docs_retrieved_count'] = docs_count
            data['retrieval_confidence'] = retrieval_confidence
            data['sources_used'] = sources_used
            
            return Roadmap(**data)
                
        except Exception as e:
            logger.error(f"Failed to parse roadmap generation: {e}")
            logger.debug(f"Raw response (first 1000 chars): {response_text[:1000]}...")
            logger.debug(f"Extracted json_str (first 500 chars): {json_str[:500] if 'json_str' in locals() else 'N/A'}...")
            raise ValueError("Failed to generate valid roadmap format") from e

    async def process_chat_stream(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        strict_mode: Optional[bool] = None
    ):
        """
        Process chat with streaming response (Server-Sent Events).
        Yields JSON: {"event": "status|token|error", "data": "..."}
        """
        try:
            # 1. Analyze (Non-streaming for now, fast enough)
            yield json.dumps({"event": "status", "data": "Analyzing your request..."}) + "\n"
            
            # Extract user info if provided
            user_id = user_context.get("user_id") if user_context else None
            user_name = user_context.get("user_name") if user_context else None
            
            # Quick greeting check
            if self._is_greeting(user_message):
                greeting_msg = f"Hello{' ' + user_name if user_name else ''}! 👋 I'm here to create your learning roadmap. What would you like to learn or build?"
                yield json.dumps({"event": "token", "data": greeting_msg}) + "\n"
                return
            
            # Analyze query
            extracted = await self._analyze_query_async(user_message, conversation_history)
            
            if not extracted.is_complete:
                # Need clarification - just stream the text
                clarification = self._create_clarification(extracted)
                yield json.dumps({"event": "token", "data": clarification.message}) + "\\n"
                return

            # 2. Retrieve (Non-streaming)
            yield json.dumps({"event": "status", "data": f"Searching documentation for {extracted.goal}..."}) + "\n"
            
            # Build filters logic (duplicate from generate_roadmap)
            search_filter = None
            if extracted.tech_stack:
                conditions = []
                available_keys = {k.lower(): k for k in (self.framework_patterns or {}).keys()}
                for tech in extracted.tech_stack:
                    tech_normalized = tech.lower().replace('.', '').replace('-', '').replace(' ', '')
                    normalized_key = tech_normalized
                    for key in available_keys:
                        key_normalized = key.replace('.', '').replace('-', '').replace(' ', '')
                        if key_normalized in tech_normalized or tech_normalized in key_normalized:
                            normalized_key = available_keys[key]
                            break
                    conditions.append(
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=normalized_key)
                        )
                    )
                if conditions:
                    search_filter = Filter(should=conditions)

            retrieved_docs = await self.retriever.retrieve_async(
                query=f"{extracted.goal} {extracted.intent} {' '.join(extracted.tech_stack or [])}",
                budget=self.token_planner.plan(Intent(extracted.intent), Depth.BALANCED),
                max_candidates=5,
                filters=search_filter
            )

            # Check if strict mode is enabled
            is_strict = strict_mode if strict_mode is not None else settings.STRICT_DOCUMENT_MODE
            
            if is_strict:
                docs_count = len(retrieved_docs)
                # Zero documents - reject immediately
                if docs_count == 0:
                    error_msg = (
                        f"No documentation found for {', '.join(extracted.tech_stack or ['the requested topic'])}. "
                        "Cannot generate roadmap without source material. "
                        "Please try a different technology or check if documentation is available."
                    )
                    logger.warning(f"Retrieval failed (Strict Mode): {error_msg}")
                    yield json.dumps({"event": "error", "data": error_msg}) + "\n"
                    return
            
            # 3. Generate Stream
            yield json.dumps({"event": "status", "data": "Generating roadmap..."}) + "\n"
            
            # Build context with URLs for LLM
            context_str = self._build_context_with_urls(retrieved_docs)
            
            # 2. Construct Prompt (High Context, Concise Output) - Matched with generate_roadmap
            min_phases = 5
            max_phases = 9
            proficiency = extracted.proficiency
            intent_val = extracted.intent
            tech_stack = extracted.tech_stack or []

            system_prompt = load_prompt(
                "roadmap_generator",
                GOAL=extracted.goal,
                INTENT=intent_val,
                PROFICIENCY=proficiency,
                TECH_STACK=', '.join(tech_stack),
                DOCS_COUNT=len(retrieved_docs),
                CONTEXT=context_str[:4500],
                MIN_PHASES=min_phases,
                MAX_PHASES=max_phases,
                KEY_TECHNOLOGIES=json.dumps(tech_stack)
            )
            
            generator_llm = self.llm_manager.get_generator_llm()
            
            # Stream the response
            # Stream the response
            async for chunk in generator_llm.astream([
                SystemMessage(content=system_prompt)
            ]):
                # logger.info(f"DEBUG CHUNK TYPE: {type(chunk)}")
                # logger.info(f"DEBUG CHUNK VARS: {vars(chunk) if hasattr(chunk, '__dict__') else 'no dict'}")
                
                content = None
                if hasattr(chunk, 'content'):
                    content = chunk.content
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    content = chunk.message.content
                
                if content:
                    yield json.dumps({"event": "token", "data": content}) + "\n"
                else:
                    logger.warning(f"Empty chunk received: {chunk}")
                    yield json.dumps({"event": "ping", "data": ""}) + "\n" # Keep alive
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"event": "error", "data": str(e)}) + "\n"

    def _update_metrics(self, roadmap: Roadmap, generation_time: float) -> None:
        """Update service metrics."""
        self._metrics["roadmaps_generated"] += 1
        self._metrics["total_phases_generated"] += len(roadmap.phases)
        
        # EMA for average time
        alpha = 0.1
        self._metrics["avg_generation_time_s"] = (
            alpha * generation_time +
            (1 - alpha) * self._metrics["avg_generation_time_s"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        return {
            **self._metrics,
            "cache": self.cache.get_stats(),
        }
    
    def get_health(self) -> Dict[str, str]:
        """Get service health status."""
        try:
            # Check critical components
            retriever_ok = self.retriever.client is not None
            llm_ok = self.llm_manager is not None
            analyzer_ok = self.analyzer is not None
            
            if retriever_ok and llm_ok and analyzer_ok:
                return {"status": "healthy", "message": "All systems operational"}
            else:
                return {"status": "degraded", "message": "Some components unavailable"}
        
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
