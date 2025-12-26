from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Literal
from enum import Enum

class QueryIntent(str, Enum):
    LEARN = "learn"
    BUILD = "build"

class RoadmapRequest(BaseModel):
    goal: str = Field(..., description="User's learning or building goal")
    intent: QueryIntent = Field(..., description="Intent: 'learn' or 'build'")
    proficiency: str = Field(..., description="User's current skill level")

class Topic(BaseModel):
    title: str
    description: str
    why_it_matters: str = Field(..., description="Explanation of why this is critical for production")
    key_concepts: List[str] = Field(default_factory=list, description="Bullet points of core concepts to master")
    doc_sources: List[str] = Field(default_factory=list, description="List of documentation URLs or file paths")
    best_practices: List[str] = Field(default_factory=list, description="List of best practices identified")
    estimated_hours: float

class Phase(BaseModel):
    title: str
    topics: List[Topic]
    estimated_hours: float

class Roadmap(BaseModel):
    phases: List[Phase]
    total_estimated_hours: float
    
class Document(BaseModel):
    page_content: str
    metadata: Dict = Field(default_factory=dict)

class ClarificationRequest(BaseModel):
    message: str
    missing_fields: List[str]

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's raw input message")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous messages")
