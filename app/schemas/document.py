"""Document schema."""

from pydantic import BaseModel, Field
from typing import Dict, Any

class Document(BaseModel):
    """Simple document schema."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
