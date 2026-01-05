"""
Centralized constants and enums for Bato-AI.

This module provides a single source of truth for all enums and constants
used throughout the application. Adding new values (e.g., proficiency levels)
only requires editing this file.
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


def normalize_proficiency(value: str) -> str:
    """Map various terms to the 3 core proficiency levels."""
    value = value.lower().strip()
    
    mapping = {
        # Beginner mappings
        "novice": ProficiencyLevel.BEGINNER.value,
        "elementary": ProficiencyLevel.BEGINNER.value,
        "beginner": ProficiencyLevel.BEGINNER.value,
        "newbie": ProficiencyLevel.BEGINNER.value,
        "basic": ProficiencyLevel.BEGINNER.value,
        
        # Intermediate mappings
        "competent": ProficiencyLevel.INTERMEDIATE.value,
        "proficient": ProficiencyLevel.INTERMEDIATE.value,
        "intermediate": ProficiencyLevel.INTERMEDIATE.value,
        "medium": ProficiencyLevel.INTERMEDIATE.value,
        
        # Advanced mappings
        "expert": ProficiencyLevel.ADVANCED.value,
        "master": ProficiencyLevel.ADVANCED.value,
        "professional": ProficiencyLevel.ADVANCED.value,
        "advanced": ProficiencyLevel.ADVANCED.value,
        "senior": ProficiencyLevel.ADVANCED.value
    }
    
    return mapping.get(value, ProficiencyLevel.BEGINNER.value)


# ============================================================================
# Learning Intent
# ============================================================================

class Intent(str, Enum):
    """User's learning intent."""
    LEARN = "learn"
    BUILD = "build"
    # Easy to add: PRACTICE = "practice", REVIEW = "review"


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
# Suggestions for UI
# ============================================================================

def get_all_suggestions() -> Dict[str, List[str]]:
    """
    Get all suggestion values for UI dropdowns/autocomplete.
    
    Returns:
        Dict with keys: proficiency, intent, depth
    """
    return {
        "proficiency": get_proficiency_levels(),
        "intent": get_intents(),
        "depth": get_depths()
    }


def get_all_descriptions() -> Dict[str, Dict[str, str]]:
    """
    Get all descriptions for UI tooltips/help text.
    
    Returns:
        Dict with keys: proficiency, intent, depth
    """
    return {
        "proficiency": get_proficiency_descriptions(),
        "intent": get_intent_descriptions(),
        "depth": get_depth_descriptions()
    }


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_proficiency(value: str) -> bool:
    """Check if proficiency value is valid."""
    return value in get_proficiency_levels()


def validate_intent(value: str) -> bool:
    """Check if intent value is valid."""
    return value in get_intents()


def validate_depth(value: str) -> bool:
    """Check if depth value is valid."""
    return value in get_depths()


# ============================================================================
# Default Values
# ============================================================================

DEFAULT_PROFICIENCY = ProficiencyLevel.BEGINNER.value
DEFAULT_INTENT = Intent.LEARN.value
DEFAULT_DEPTH = Depth.BALANCED.value


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """Demonstrate usage of constants module."""
    
    print("=== Proficiency Levels ===")
    for level in ProficiencyLevel:
        desc = get_proficiency_descriptions()[level.value]
        print(f"  {level.value}: {desc}")
    
    print("\n=== Intents ===")
    for intent in Intent:
        desc = get_intent_descriptions()[intent.value]
        print(f"  {intent.value}: {desc}")
    
    print("\n=== Depths ===")
    for depth in Depth:
        desc = get_depth_descriptions()[depth.value]
        print(f"  {depth.value}: {desc}")
    
    print("\n=== Validation Patterns ===")
    print(f"Proficiency: {get_proficiency_pattern()}")
    print(f"Intent: {get_intent_pattern()}")
    print(f"Depth: {get_depth_pattern()}")
    
    print("\n=== All Suggestions (for UI) ===")
    print(get_all_suggestions())
