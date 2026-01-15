"""Roadmap-related schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


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
