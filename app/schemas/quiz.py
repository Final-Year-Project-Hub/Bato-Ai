"""
Quiz-related Pydantic schemas.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QuizOption(BaseModel):
    """Quiz question options."""
    A: str
    B: str
    C: str
    D: str


class QuizQuestion(BaseModel):
    """Individual quiz question."""
    id: int
    question: str
    options: QuizOption
    correctAnswer: str = Field(..., pattern="^[A-D]$")
    explanation: str
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    concept: str
    learningObjective: str
    sourceSection: str


class QuizMetadata(BaseModel):
    """Quiz metadata."""
    totalQuestions: int
    estimatedTime: str
    passingScore: int = Field(default=70, ge=0, le=100)


class Quiz(BaseModel):
    """Complete quiz structure."""
    questions: List[QuizQuestion]
    metadata: QuizMetadata


class QuizGenerationRequest(BaseModel):
    """Request to generate a quiz."""
    goal: str
    phase_title: str
    topic_title: str
    topic_content: Dict  # The full topic content JSON


class QuizGenerationResponse(BaseModel):
    """Response from quiz generation."""
    quiz: Quiz
    generated_at: str
