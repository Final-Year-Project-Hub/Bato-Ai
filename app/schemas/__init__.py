"""Schemas package."""

from app.schemas.roadmap import (
    Roadmap,
    Phase,
    Topic,
    TopicDetail,
    LearningResource,
    PracticeExercise
)
from app.schemas.requests import (
    RoadmapRequest,
    ChatRequest,
    QueryIntent,
    UserContext,
    UserPreferences,
)
from app.schemas.responses import ClarificationRequest, InsufficientDocumentationError

# Simple Document class (moved from common.py)
from pydantic import BaseModel, Field
from typing import Dict

class Document(BaseModel):
    page_content: str
    metadata: Dict = Field(default_factory=dict)

__all__ = [
    # Roadmap
    "Roadmap",
    "Phase",
    "Topic",
    "TopicDetail",
    "LearningResource",
    "PracticeExercise",
    # Requests
    "RoadmapRequest",
    "ChatRequest",
    "QueryIntent",
    "UserContext",
    "UserPreferences",
    # Responses
    "ClarificationRequest",
    "InsufficientDocumentationError",
    # Common
    "Document",
]
