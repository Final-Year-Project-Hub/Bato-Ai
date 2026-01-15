"""Schemas package."""

from .roadmap import Roadmap, Phase, Topic, Subtopic, BestPractice
from .requests import RoadmapRequest, ChatRequest, QueryIntent
from .responses import ClarificationRequest, InsufficientDocumentationError

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
    "Subtopic",
    "BestPractice",
    # Requests
    "RoadmapRequest",
    "ChatRequest",
    "QueryIntent",
    # Responses
    "ClarificationRequest",
    "InsufficientDocumentationError",
    # Common
    "Document",
]
