"""Request schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class UserPreferences(BaseModel):
    """User preferences for personalized roadmap generation."""
    default_proficiency: Optional[str] = Field(
        None, description="User's default proficiency level"
    )
    learning_style: Optional[str] = Field(
        None, description="Preferred learning style (conceptual/practical/balanced)"
    )
    time_availability: Optional[str] = Field(
        None, description="Available time per day/week"
    )
    learning_pace: Optional[str] = Field(
        None, description="Preferred learning pace (slow/moderate/fast)"
    )
    preferred_frameworks: Optional[List[str]] = Field(
        None, description="Frameworks user is interested in"
    )


class UserContext(BaseModel):
    """Optional user context for personalized roadmap generation."""
    user_id: Optional[str] = Field(None, description="User ID from authentication system")
    user_name: Optional[str] = Field(None, description="User's name for personalized greetings")
    known_technologies: Optional[List[str]] = Field(
        None, description="Technologies user already knows"
    )
    learning_history: Optional[List[str]] = Field(
        None, description="Previously completed topics/roadmaps"
    )
    preferences: Optional[UserPreferences] = Field(
        None, description="User preferences"
    )


class QueryIntent(str, Enum):
    LEARN = "learn"
    BUILD = "build"


class RoadmapRequest(BaseModel):
    goal: str = Field(..., description="User's learning or building goal")
    intent: QueryIntent = Field(..., description="Intent: 'learn' or 'build'")
    proficiency: str = Field(..., description="User's current skill level")
    strict_mode: Optional[bool] = Field(None, description="Force strict documentation adherence")


class ChatRequest(BaseModel):
    """Chat request for conversational roadmap generation."""
    message: str = Field(..., min_length=1, max_length=5000, description="User's message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Previous conversation messages"
    )
    user_context: Optional[UserContext] = Field(
        None, description="Optional user context for personalization"
    )
    strict_mode: Optional[bool] = Field(None, description="Force strict documentation adherence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "I want to learn React",
                "conversation_history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! What would you like to learn?"}
                ]
            }
        }
