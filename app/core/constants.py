"""
Centralized constants and enums for Bato-AI.
Simplified for college project - removed unnecessary helpers.
"""

from enum import Enum
from typing import List, Dict


# ============================================================================
# Proficiency Levels
# ============================================================================

class ProficiencyLevel(str, Enum):
    """User proficiency levels for learning."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

def get_proficiency_levels() -> List[str]:
    """Get all proficiency level values."""
    return [level.value for level in ProficiencyLevel]


def get_proficiency_pattern() -> str:
    """Get regex pattern for proficiency validation."""
    levels = "|".join(get_proficiency_levels())
    return f"^({levels})$"


def get_proficiency_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions for each proficiency level."""
    return {
        ProficiencyLevel.BEGINNER.value: "New to the technology, learning fundamentals",
        ProficiencyLevel.INTERMEDIATE.value: "Some experience, building projects",
        ProficiencyLevel.ADVANCED.value: "Strong understanding, advanced concepts"
    }


# ============================================================================
# Learning Intent
# ============================================================================

class Intent(str, Enum):
    """User's learning intent."""
    LEARN = "learn"
    BUILD = "build"


def get_intents() -> List[str]:
    """Get all intent values."""
    return [intent.value for intent in Intent]


def get_intent_pattern() -> str:
    """Get regex pattern for intent validation."""
    intents = "|".join(get_intents())
    return f"^({intents})$"


def get_intent_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions for each intent."""
    return {
        Intent.LEARN.value: "Focus on understanding concepts and theory",
        Intent.BUILD.value: "Focus on building projects and practical application"
    }


# ============================================================================
# Learning Depth
# ============================================================================

class Depth(str, Enum):
    """Learning depth preference."""
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"
    BALANCED = "balanced"


def get_depths() -> List[str]:
    """Get all depth values."""
    return [depth.value for depth in Depth]


# Aliases for consistency
get_intent_values = get_intents  # Alias for query_analyzer compatibility
get_depth_values = get_depths    # Alias for query_analyzer compatibility


def get_depth_pattern() -> str:
    """Get regex pattern for depth validation."""
    depths = "|".join(get_depths())
    return f"^({depths})$"


def get_depth_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions for each depth."""
    return {
        Depth.CONCEPTUAL.value: "Focus on theory, concepts, and understanding",
        Depth.PRACTICAL.value: "Focus on hands-on practice and implementation",
        Depth.BALANCED.value: "Mix of theory and practice"
    }


# ============================================================================
# Default Values
# ============================================================================

DEFAULT_PROFICIENCY = ProficiencyLevel.BEGINNER.value
DEFAULT_INTENT = Intent.LEARN.value
DEFAULT_DEPTH = Depth.BALANCED.value


# ============================================================================
# Suggestion Values for Clarification
# ============================================================================

def get_all_suggestions() -> Dict[str, List[str]]:
    """
    Get all suggestion values for clarification requests.
    
    Returns a dictionary mapping field names to lists of suggested values.
    Used when the system needs to ask the user for missing information.
    """
    return {
        "proficiency_level": get_proficiency_levels(),
        "intent": get_intents(),
        "depth": get_depths(),
    }