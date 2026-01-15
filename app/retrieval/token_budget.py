# token_budget.py
"""
Token budget management for RAG pipeline with DeepSeek V3.2.
Allocates tokens across system, query, retrieval, and generation phases.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

# Import enums from centralized constants
from app.core.constants import Intent, Depth

logger = logging.getLogger(__name__)



@dataclass
class TokenBudget:
    """
    Token allocation for complete RAG pipeline.
    
    Components:
    - system_tokens: System prompt + instructions
    - query_tokens: User query + extracted intent
    - retrieval_tokens: Retrieved document context
    - answer_tokens: LLM generation budget
    """
    system_tokens: int
    query_tokens: int
    retrieval_tokens: int
    answer_tokens: int
    
    @property
    def total(self) -> int:
        """Total token budget across all components."""
        return (
            self.system_tokens +
            self.query_tokens +
            self.retrieval_tokens +
            self.answer_tokens
        )
    
    @property
    def context_budget(self) -> int:
        """Tokens available for context (system + query + retrieval)."""
        return self.system_tokens + self.query_tokens + self.retrieval_tokens
    
    def __str__(self) -> str:
        return (
            f"TokenBudget(total={self.total}, "
            f"system={self.system_tokens}, "
            f"query={self.query_tokens}, "
            f"retrieval={self.retrieval_tokens}, "
            f"answer={self.answer_tokens})"
        )


class TokenBudgetPlanner:
    """
    Plans token allocation based on intent and depth.
    
    Optimized for:
    - DeepSeek V3.2 Special (context: 64K tokens, recommended: 32K)
    - BAAI/bge-small-en-v1.5 embeddings (384 dimensions)
    
    Strategy:
    - LEARN intent: More retrieval, moderate generation
    - BUILD intent: Balanced retrieval, more generation for code
    - CONCEPTUAL depth: Less retrieval, more explanation
    - PRACTICAL depth: More retrieval, less explanation
    """
    
    # Model configurations
    DEEPSEEK_V3_MAX = 65536      # DeepSeek V3.2 max context
    DEEPSEEK_V3_SAFE = 32768     # Safe limit for production
    DEEPSEEK_V3_RECOMMENDED = 16384  # Recommended for best performance
    
    def __init__(
        self,
        model_max_tokens: int = DEEPSEEK_V3_RECOMMENDED,
        conservative: bool = True
    ):
        """
        Initialize token budget planner.
        
        Args:
            model_max_tokens: Maximum context window to use
            conservative: Use conservative token limits (recommended)
        """
        if model_max_tokens > self.DEEPSEEK_V3_MAX:
            logger.warning(
                f"model_max_tokens ({model_max_tokens}) exceeds "
                f"DeepSeek V3.2 max ({self.DEEPSEEK_V3_MAX}), "
                f"clamping to safe limit"
            )
            model_max_tokens = self.DEEPSEEK_V3_SAFE
        
        self.model_max_tokens = model_max_tokens
        self.conservative = conservative
        
        logger.info(
            f"TokenBudgetPlanner initialized: "
            f"max_tokens={model_max_tokens}, conservative={conservative}"
        )
    
    def plan(
        self,
        intent: Intent,
        depth: Optional[Depth] = None
    ) -> TokenBudget:
        """
        Plan token allocation based on intent and depth.
        
        Args:
            intent: User intent (LEARN or BUILD)
            depth: Query depth (CONCEPTUAL, PRACTICAL, or BALANCED)
            
        Returns:
            TokenBudget with optimal allocation
            
        Examples:
        ---------
        # Learning conceptual topics
        planner = TokenBudgetPlanner()
        budget = planner.plan(Intent.LEARN, Depth.CONCEPTUAL)
        # More answer tokens for explanation
        
        # Building with practical details
        budget = planner.plan(Intent.BUILD, Depth.PRACTICAL)
        # More retrieval tokens for code examples
        """
        depth = depth or Depth.BALANCED
        
        # Base allocations (conservative defaults)
        system_tokens = 800 if self.conservative else 600
        query_tokens = 200 if self.conservative else 150
        
        # Intent-based base allocation
        if intent == Intent.LEARN:
            # Learning: moderate retrieval, more generation for explanations
            retrieval_tokens = 2500
            answer_tokens = 2000
        else:  # Intent.BUILD
            # Building: more retrieval for examples, moderate generation
            retrieval_tokens = 3500
            answer_tokens = 1500
        
        # Depth adjustments
        if depth == Depth.CONCEPTUAL:
            # Conceptual: Less context, more explanation
            retrieval_tokens -= 500
            answer_tokens += 500
        elif depth == Depth.PRACTICAL:
            # Practical: More context, less explanation
            retrieval_tokens += 800
            answer_tokens -= 500
        
        # Safety clamps (minimum viable budgets)
        retrieval_tokens = max(1000, retrieval_tokens)
        answer_tokens = max(800, answer_tokens)
        
        # Ensure we don't exceed model limit
        total = system_tokens + query_tokens + retrieval_tokens + answer_tokens
        
        if total > self.model_max_tokens:
            overflow = total - self.model_max_tokens
            
            # Reduce retrieval first, then answer if needed
            reduction = min(overflow, retrieval_tokens - 1000)
            retrieval_tokens -= reduction
            overflow -= reduction
            
            if overflow > 0:
                answer_tokens -= min(overflow, answer_tokens - 800)
        
        budget = TokenBudget(
            system_tokens=system_tokens,
            query_tokens=query_tokens,
            retrieval_tokens=retrieval_tokens,
            answer_tokens=answer_tokens
        )
        
        logger.debug(
            f"Planned budget for intent={intent.value if hasattr(intent, 'value') else intent}, "
            f"depth={depth.value if hasattr(depth, 'value') else depth}: "
            f"{budget}"
        )
        
        return budget
    
    def plan_custom(
        self,
        retrieval_ratio: float = 0.6,
        answer_ratio: float = 0.3
    ) -> TokenBudget:
        """
        Plan with custom ratios for advanced use cases.
        
        Args:
            retrieval_ratio: Proportion for retrieval (0-1)
            answer_ratio: Proportion for answer (0-1)
            
        Returns:
            TokenBudget with custom allocation
        """
        if retrieval_ratio + answer_ratio > 0.95:
            raise ValueError(
                "retrieval_ratio + answer_ratio must be <= 0.95 "
                "(need room for system + query)"
            )
        
        system_tokens = int(self.model_max_tokens * 0.03)  # 3%
        query_tokens = int(self.model_max_tokens * 0.02)   # 2%
        
        available = self.model_max_tokens - system_tokens - query_tokens
        
        retrieval_tokens = int(available * retrieval_ratio)
        answer_tokens = int(available * answer_ratio)
        
        return TokenBudget(
            system_tokens=system_tokens,
            query_tokens=query_tokens,
            retrieval_tokens=retrieval_tokens,
            answer_tokens=answer_tokens
        )


# Convenience functions for quick usage
def get_learn_budget(depth: Depth = Depth.BALANCED) -> TokenBudget:
    """Quick budget for learning queries."""
    planner = TokenBudgetPlanner()
    return planner.plan(Intent.LEARN, depth)


def get_build_budget(depth: Depth = Depth.BALANCED) -> TokenBudget:
    """Quick budget for building queries."""
    planner = TokenBudgetPlanner()
    return planner.plan(Intent.BUILD, depth)


# Production defaults for different scenarios
class BudgetPresets:
    """Pre-configured budgets for common scenarios."""
    
    @staticmethod
    def quick_answer() -> TokenBudget:
        """Quick answers with minimal context."""
        return TokenBudget(
            system_tokens=500,
            query_tokens=100,
            retrieval_tokens=1000,
            answer_tokens=800
        )
    
    @staticmethod
    def comprehensive() -> TokenBudget:
        """Comprehensive answers with full context."""
        return TokenBudget(
            system_tokens=800,
            query_tokens=200,
            retrieval_tokens=4000,
            answer_tokens=3000
        )
    
    @staticmethod
    def code_heavy() -> TokenBudget:
        """Code-heavy responses with examples."""
        return TokenBudget(
            system_tokens=600,
            query_tokens=150,
            retrieval_tokens=3500,
            answer_tokens=2000
        )


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    planner = TokenBudgetPlanner()
    
    # Test different scenarios
    scenarios = [
        (Intent.LEARN, Depth.CONCEPTUAL, "Learning high-level concepts"),
        (Intent.LEARN, Depth.PRACTICAL, "Learning with code examples"),
        (Intent.BUILD, Depth.CONCEPTUAL, "Building with architecture focus"),
        (Intent.BUILD, Depth.PRACTICAL, "Building with implementation focus"),
    ]
    
    print("\n" + "=" * 70)
    print("TOKEN BUDGET PLANNING - DEMO")
    print("=" * 70)
    
    for intent, depth, description in scenarios:
        budget = planner.plan(intent, depth)
        print(f"\n{description}")
        print(f"  Intent: {intent.value}, Depth: {depth.value}")
        print(f"  {budget}")
        print(f"  Context budget: {budget.context_budget} tokens")
    
    print("\n" + "=" * 70)
    print("PRESET BUDGETS")
    print("=" * 70)
    
    presets = [
        ("Quick Answer", BudgetPresets.quick_answer()),
        ("Comprehensive", BudgetPresets.comprehensive()),
        ("Code Heavy", BudgetPresets.code_heavy()),
    ]
    
    for name, budget in presets:
        print(f"\n{name}:")
        print(f"  {budget}")