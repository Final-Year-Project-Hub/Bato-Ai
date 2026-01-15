from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Literal, Any
from enum import Enum
from datetime import datetime


# ============================================================================
# User Context Schemas (Optional - for personalization)
# ============================================================================

class UserPreferences(BaseModel):
    """User preferences for personalized roadmap generation."""
    default_proficiency: Optional[str] = Field(None, description="User's default proficiency level")
    learning_style: Optional[str] = Field(None, description="Preferred learning style (conceptual/practical/balanced)")
    time_availability: Optional[str] = Field(None, description="Available time per day/week")
    learning_pace: Optional[str] = Field(None, description="Preferred learning pace (slow/moderate/fast)")
    preferred_frameworks: Optional[List[str]] = Field(None, description="Frameworks user is interested in")


class UserContext(BaseModel):
    """Optional user context for personalized roadmap generation."""
    user_id: Optional[str] = Field(None, description="User ID from authentication system")
    user_name: Optional[str] = Field(None, description="User's name for personalized greetings")
    known_technologies: Optional[List[str]] = Field(None, description="Technologies user already knows")
    learning_history: Optional[List[str]] = Field(None, description="Previously completed topics/roadmaps")
    preferences: Optional[UserPreferences] = Field(None, description="User preferences")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "user_name": "John Doe",
                "known_technologies": ["HTML", "CSS", "JavaScript"],
                "learning_history": ["Python Basics", "Git Fundamentals"],
                "preferences": {
                    "default_proficiency": "intermediate",
                    "learning_style": "practical",
                    "time_availability": "2-3 hours/day"
                }
            }
        }


# ============================================================================
# Query and Request Schemas
# ============================================================================

class QueryIntent(str, Enum):
    LEARN = "learn"
    BUILD = "build"

class RoadmapRequest(BaseModel):
    goal: str = Field(..., description="User's learning or building goal")
    intent: QueryIntent = Field(..., description="Intent: 'learn' or 'build'")
    proficiency: str = Field(..., description="User's current skill level")

class BestPractice(BaseModel):
    text: str
    url: Optional[str] = None

class Subtopic(BaseModel):
    title: str
    description: str
    estimated_hours: float
    doc_link: Optional[str] = None
    best_practices: Optional[List[str]] = None

class Topic(BaseModel):
    title: str
    description: str
    estimated_hours: float
    doc_link: Optional[str] = None
    subtopics: List[Subtopic] = Field(default_factory=list)
    best_practices: Optional[List[str]] = None

class Phase(BaseModel):
    title: str
    description: str
    estimated_hours: float
    topics: List[Topic]

class Roadmap(BaseModel):
    goal: str
    intent: str
    proficiency: str
    phases: List[Phase]
    total_estimated_hours: float
    key_technologies: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Retrieval metadata for transparency
    docs_retrieved_count: int = Field(default=0, description="Number of documents retrieved for generation")
    retrieval_confidence: float = Field(default=0.0, description="Confidence score based on retrieval quality")
    sources_used: List[str] = Field(default_factory=list, description="List of documentation sources used")
    
class Document(BaseModel):
    page_content: str
    metadata: Dict = Field(default_factory=dict)

class ClarificationRequest(BaseModel):
    status: str = "clarification_needed"
    message: str
    missing_fields: List[str]
    suggested_values: Optional[Dict[str, List[str]]] = None

class InsufficientDocumentationError(BaseModel):
    """Error response when insufficient documentation is found."""
    error_type: str = Field(default="insufficient_documentation")
    message: str
    tech_stack: List[str] = Field(default_factory=list)
    docs_found: int = Field(default=0)
    min_required: int = Field(default=3)

class ChatRequest(BaseModel):
    """
    Chat request for conversational roadmap generation.
    
    Supports optional user context for personalized responses.
    """
    message: str = Field(..., min_length=1, max_length=5000, description="User's message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Previous conversation messages"
    )
    user_context: Optional[UserContext] = Field(
        None,
        description="Optional user context for personalization (name, preferences, known tech)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "I want to learn React",
                "conversation_history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! What would you like to learn?"}
                ],
                "user_context": {
                    "user_name": "John",
                    "known_technologies": ["HTML", "CSS", "JavaScript"],
                    "preferences": {
                        "default_proficiency": "intermediate"
                    }
                }
            }
        }
