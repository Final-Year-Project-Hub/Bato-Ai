"""Roadmap-related schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Topic(BaseModel):
    """Topic within a phase - brief overview for roadmap display."""
    title: str
    description: str = Field(..., description="Brief 1-2 sentence description")
    estimated_hours: float
    doc_link: Optional[str] = None
    
    # Optional fields for overview
    prerequisites: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list, description="3-5 key concepts")


class Phase(BaseModel):
    """Phase in learning roadmap - contains 6 topics."""
    phase_number: int = Field(..., ge=1, le=7, description="Phase number (1-7)")
    title: str
    description: str = Field(..., description="Brief phase overview")
    estimated_hours: float
    topics: List[Topic] = Field(..., min_length=5, max_length=7, description="Should have ~6 topics")
    
    # Checkpoint after phase
    checkpoint_project: Optional[str] = Field(None, description="Project to validate phase completion")


class Roadmap(BaseModel):
    """Learning roadmap following roadmap.sh structure: 7 phases, ~6 topics each."""
    goal: str
    intent: str
    proficiency: str
    
    # 7 phases with ~6 topics each
    phases: List[Phase] = Field(..., min_length=7, max_length=7, description="Must have exactly 7 phases")
    
    total_estimated_hours: float
    key_technologies: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Retrieval metadata for transparency
    docs_retrieved_count: int = Field(default=0, description="Number of documents retrieved for generation")
    retrieval_confidence: float = Field(default=0.0, description="Confidence score based on retrieval quality")
    sources_used: List[str] = Field(default_factory=list, description="List of documentation sources used")


# ============================================================================
# Topic Deep-Dive Schema (for detailed topic exploration)
# ============================================================================

class LearningResource(BaseModel):
    """Learning resource for a topic."""
    title: str
    type: str = Field(..., description="Type: article, video, tutorial, documentation")
    url: Optional[str] = None
    estimated_time: Optional[str] = None


class PracticeExercise(BaseModel):
    """Practice exercise for a topic."""
    title: str
    description: str
    difficulty: str = Field(..., description="Difficulty: beginner, intermediate, advanced")
    estimated_time: str


class TopicDetail(BaseModel):
    """Detailed topic content for deep-dive page."""
    # Basic info
    title: str
    phase_number: int
    phase_title: str
    
    # Detailed content
    overview: str = Field(..., description="Comprehensive overview (3-4 paragraphs)")
    why_important: str = Field(..., description="Why this topic matters")
    key_concepts: List[str] = Field(..., description="5-10 key concepts to learn")
    prerequisites: List[str] = Field(default_factory=list)
    
    # Learning path
    learning_objectives: List[str] = Field(..., description="What you'll learn")
    learning_resources: List[LearningResource] = Field(..., description="Curated resources")
    practice_exercises: List[PracticeExercise] = Field(..., description="Hands-on exercises")
    
    # Next steps
    related_topics: List[str] = Field(default_factory=list)
    next_topic: Optional[str] = None
    
    # Metadata
    estimated_hours: float
    difficulty_level: str
    doc_links: List[str] = Field(default_factory=list)
