"""
Dynamic roadmap structure calculator.
Determines optimal phase/topic counts based on context and scope.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class RoadmapCalculator:
    """Calculate optimal phase/topic counts based on documentation coverage and topic scope."""
    
    def __init__(self):
        # Configuration bounds
        self.min_phases = 3
        self.max_phases = 10
        self.min_topics_per_phase = 3
        self.max_topics_per_phase = 10
        
        # Intent-specific defaults (learn vs build)
        self.intent_defaults = {
            "learn": {
                "narrow": {"phases": 5, "topics": 6},   # More comprehensive
                "medium": {"phases": 7, "topics": 6},
                "broad": {"phases": 9, "topics": 7}
            },
            "build": {
                "narrow": {"phases": 3, "topics": 5},   # More focused
                "medium": {"phases": 5, "topics": 6},
                "broad": {"phases": 8, "topics": 6}     # Increased for complex projects
            }
        }
        
        # Fallback defaults (for other intents)
        self.scope_defaults = {
            "narrow": {"phases": 4, "topics": 5},
            "medium": {"phases": 6, "topics": 6},
            "broad": {"phases": 8, "topics": 7}
        }
        
        logger.info("✅ RoadmapCalculator initialized with intent-aware defaults")
    
    def calculate_structure(
        self,
        scope: str,
        proficiency: str,
        intent: str,
        docs_retrieved: int,
        avg_doc_score: float = 0.5,
        expected_docs: int = 10
    ) -> Tuple[int, int]:
        """
        Calculate optimal roadmap structure based on scope, intent, and documentation.
        
        Args:
            scope: Topic scope ("narrow", "medium", "broad")
            proficiency: User proficiency level
            intent: User intent ("learn", "build", etc.)
            docs_retrieved: Number of documents retrieved
            avg_doc_score: Average similarity score of retrieved docs
            expected_docs: Expected number of docs (k parameter)
        
        Returns:
            Tuple of (num_phases, topics_per_phase)
        """
        # Determine intent type (learn vs build)
        learn_intents = ["learn", "understand", "study", "master", "explore"]
        build_intents = ["build", "create", "develop", "make", "implement"]
        
        if intent.lower() in learn_intents:
            intent_type = "learn"
        elif intent.lower() in build_intents:
            intent_type = "build"
        else:
            intent_type = None  # Use fallback defaults
        
        # Get base structure from intent + scope
        if intent_type and intent_type in self.intent_defaults:
            defaults = self.intent_defaults[intent_type].get(scope, self.scope_defaults.get(scope, {"phases": 6, "topics": 6}))
        else:
            defaults = self.scope_defaults.get(scope, {"phases": 6, "topics": 6})
        
        base_phases = defaults["phases"]
        base_topics = defaults["topics"]
        
        logger.info(
            f"📐 Base structure for '{scope}' scope + '{intent}' intent: "
            f"{base_phases} phases × {base_topics} topics"
        )
        
        # Calculate documentation coverage ratio
        doc_coverage_ratio = docs_retrieved / max(expected_docs, 1)
        
        # Adjust based on documentation coverage
        if doc_coverage_ratio < 0.5:  # Very low coverage (< 5 docs for k=10)
            logger.warning(f"⚠️ Low documentation coverage: {docs_retrieved}/{expected_docs} docs")
            base_phases = max(self.min_phases, base_phases - 2)
            base_topics = max(self.min_topics_per_phase, base_topics - 1)
        elif doc_coverage_ratio < 0.8:  # Below average coverage
            base_phases = max(self.min_phases, base_phases - 1)
        elif doc_coverage_ratio > 1.5:  # Excellent coverage (> 15 docs for k=10)
            logger.info(f"✅ Excellent documentation coverage: {docs_retrieved}/{expected_docs} docs")
            base_phases = min(self.max_phases, base_phases + 1)
            base_topics = min(self.max_topics_per_phase, base_topics + 1)
        
        # Adjust for proficiency level
        if proficiency == "beginner":
            # Beginners need more granular steps
            base_topics = min(self.max_topics_per_phase, base_topics + 1)
        elif proficiency == "advanced":
            # Advanced users can handle broader topics
            base_topics = max(self.min_topics_per_phase, base_topics - 1)
            base_phases = max(self.min_phases, base_phases - 1)
        
        # Quality adjustment based on average doc score
        if avg_doc_score < 0.3:  # Low relevance
            logger.warning(f"⚠️ Low document relevance: avg_score={avg_doc_score:.3f}")
            base_topics = max(self.min_topics_per_phase, base_topics - 1)
        
        # Final bounds check
        num_phases = max(self.min_phases, min(self.max_phases, base_phases))
        topics_per_phase = max(self.min_topics_per_phase, min(self.max_topics_per_phase, base_topics))
        
        total_topics = num_phases * topics_per_phase
        
        logger.info(
            f"📊 Final structure: {num_phases} phases × {topics_per_phase} topics = {total_topics} total topics"
        )
        logger.info(
            f"   Factors: scope={scope}, proficiency={proficiency}, "
            f"docs={docs_retrieved}/{expected_docs}, avg_score={avg_doc_score:.3f}"
        )
        
        return (num_phases, topics_per_phase)
    
    def detect_scope(
        self,
        goal: str,
        tech_stack: list,
        intent: str
    ) -> str:
        """
        Detect topic scope from query parameters.
        
        Args:
            goal: User's learning goal
            tech_stack: List of technologies
            intent: User intent (learn, build, etc.)
        
        Returns:
            Scope classification: "narrow", "medium", or "broad"
        """
        goal_lower = goal.lower()
        
        # Narrow scope indicators
        narrow_keywords = [
            "hooks", "middleware", "routing", "authentication",
            "state management", "context api", "custom hook",
            "lifecycle", "props", "component", "specific"
        ]
        
        # Broad scope indicators
        broad_keywords = [
            "full-stack", "full stack","fullstack", "mern", "mean", "complete",
            "entire", "comprehensive", "end-to-end", "production",
            "deployment", "architecture", "system design","system","bot"
        ]
        
        # Check for narrow indicators
        if any(keyword in goal_lower for keyword in narrow_keywords):
            return "narrow"
        
        # Check for broad indicators
        if any(keyword in goal_lower for keyword in broad_keywords):
            return "broad"
        
        # Check tech stack size
        if tech_stack:
            if len(tech_stack) >= 4:  # Multiple technologies
                return "broad"
            elif len(tech_stack) == 1:
                # Single tech could be narrow or medium
                # Check if goal mentions specific features
                if any(keyword in goal_lower for keyword in narrow_keywords):
                    return "narrow"
                return "medium"
        
        # Check goal length and complexity
        word_count = len(goal.split())
        if word_count <= 5:  # Short, specific goal
            return "narrow"
        elif word_count >= 15:  # Long, complex goal
            return "broad"
        
        # Default to medium
        return "medium"
