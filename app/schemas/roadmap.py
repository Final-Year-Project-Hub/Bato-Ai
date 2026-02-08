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
    """Phase in learning roadmap - flexible topic count based on scope."""
    phase_number: int = Field(..., ge=1, le=10, description="Phase number (1-10)")
    title: str
    description: str = Field(..., description="Brief phase overview")
    estimated_hours: float
    topics: List[Topic] = Field(..., min_length=3, max_length=10, description="3-10 topics per phase")
    
    # Checkpoint after phase
    checkpoint_project: Optional[str] = Field(None, description="Project to validate phase completion")


class Roadmap(BaseModel):
    """Learning roadmap with dynamic structure based on topic scope and documentation coverage."""
    goal: str
    intent: str
    proficiency: str
    
    # Dynamic phase count (3-10 phases)
    phases: List[Phase] = Field(..., min_length=3, max_length=10, description="3-10 phases based on scope")
    
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


# New nested section models
class IntroductionSection(BaseModel):
    """Introduction section with markdown content."""
    markdown: str = Field(..., description="Markdown formatted introduction")


class DetailedCoreConcept(BaseModel):
    """A single detailed core concept."""
    title: str
    markdown: str = Field(..., description="Markdown explanation of the concept")
    key_points: List[str] = Field(default_factory=list, description="Key points for this concept")


class CodeExample(BaseModel):
    """Code example with explanation."""
    title: str
    language: str = Field(..., description="Programming language (jsx, ts, js, python, etc)")
    code: str = Field(..., description="Full code example")
    explanation_markdown: str = Field(..., description="Markdown explanation of the code")


class RealWorldExample(BaseModel):
    """Real-world usage example."""
    title: str = Field(..., description="Company or product name")
    markdown: str = Field(..., description="Markdown explanation of real-world usage")


class HypotheticalScenario(BaseModel):
    """Hypothetical scenario demonstrating the topic."""
    title: str
    markdown: str = Field(..., description="Markdown description of the scenario")


class TopicSections(BaseModel):
    """Nested sections containing detailed topic content."""
    introduction: IntroductionSection
    detailed_core_concepts: List[DetailedCoreConcept] = Field(..., min_length=1)
    code_examples: List[CodeExample] = Field(..., min_length=1)
    real_world_examples: List[RealWorldExample] = Field(default_factory=list)
    hypothetical_scenario: Optional[HypotheticalScenario] = None
    key_characteristics: List[str] = Field(default_factory=list)


class TopicDetail(BaseModel):
    """Detailed topic content for deep-dive page with nested sections."""
    # Basic info
    title: str
    phase_number: int
    phase_title: str
    
    # Main content sections (nested structure)
    sections: TopicSections
    
    # Supporting content
    why_important: str = Field(..., description="Why this topic matters")
    key_concepts: List[str] = Field(..., description="List of key concepts")
    
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
