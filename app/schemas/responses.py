"""Response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


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
