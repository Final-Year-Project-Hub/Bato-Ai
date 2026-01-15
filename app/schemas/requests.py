"""Request schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


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
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context")
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
