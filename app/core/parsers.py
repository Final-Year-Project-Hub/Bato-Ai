# parsers.py
"""
Unified output parsers for LLM responses.
Handles both query analysis and roadmap generation outputs.
"""

import json
import re
import logging
from typing import Dict, Any
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryOutputParser(BaseOutputParser):
    """
    Parser for query analyzer JSON responses.
    Handles conversational text mixed with JSON output.
    """
    
    pydantic_object: type[BaseModel]
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into query analysis JSON."""
        try:
            cleaned = self._clean_text(text)
            json_str = self._extract_json(cleaned)
            data = json.loads(json_str)
            
            logger.debug(f"✅ Parsed query JSON ({len(json_str)} chars)")
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Failed text: {text[:300]}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        
        except Exception as e:
            logger.error(f"Parsing failed: {e}", exc_info=True)
            raise
    
    def _clean_text(self, text: str) -> str:
        """Remove common formatting issues."""
        text = text.strip()
        if text.startswith('\ufeff'):
            text = text[1:]
        return text
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from conversational text."""
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def get_format_instructions(self) -> str:
        """Return format instructions for the LLM."""
        return """Return a JSON object with the following structure:
{
  "goal": "string or null",
  "intent": "string or null",
  "depth": "string or null",
  "proficiency": "string or null",
  "tech_stack": ["array of strings"] or null,
  "missing_fields": ["array of field names"],
  "confidence": 0.0 to 1.0
}

CRITICAL: Include ONLY the JSON object. No explanations or conversational text."""
    
    @property
    def _type(self) -> str:
        return "query_json"


class RoadmapOutputParser(BaseOutputParser):
    """
    Parser for roadmap generation JSON responses.
    
    Features:
    - Auto-fixes common JSON formatting issues
    - Extracts JSON from markdown code blocks
    - Validates roadmap structure
    - Handles nested wrappers
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into roadmap JSON."""
        try:
            # Clean and extract
            cleaned = self._clean_text(text)
            json_str = self._extract_json(cleaned)
            fixed_json = self._fix_common_issues(json_str)
            
            # Parse
            data = json.loads(fixed_json)
            
            # Normalize structure (unwrap nested keys)
            data = self._normalize_structure(data)
            
            # Validate
            self._validate_structure(data)
            
            logger.debug(f"✅ Parsed roadmap JSON ({len(json_str)} chars)")
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Failed JSON: {text[:500]}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        
        except Exception as e:
            logger.error(f"Parsing failed: {e}", exc_info=True)
            raise
    
    def _clean_text(self, text: str) -> str:
        """Remove formatting issues."""
        text = text.strip()
        if text.startswith('\ufeff'):
            text = text[1:]
        return text
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from markdown or plain text."""
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            logger.debug("Extracted from markdown")
            return json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def _fix_common_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix single quotes (carefully)
        json_str = re.sub(r"(?<!\\)'([^']*)'(?=\s*[,:\]}])", r'"\1"', json_str)
        
        return json_str
    
    def _normalize_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize structure by unwrapping nested keys."""
        # Handle wrappers like {"Next.js Roadmap": {...}}
        if "phases" not in data and "Phases" not in data:
            for key, value in data.items():
                if isinstance(value, dict) and ("phases" in value or "Phases" in value):
                    data = value
                    break
        
        # Normalize 'Phases' to 'phases'
        if "Phases" in data and "phases" not in data:
            data["phases"] = data.pop("Phases")
        
        return data
    
    def _validate_structure(self, data: Dict[str, Any]) -> None:
        """Validate roadmap structure."""
        # Check required top-level fields
        if "phases" not in data:
            logger.error(f"Missing 'phases'. Available keys: {list(data.keys())}")
            raise ValueError("Missing required field: 'phases'")
        
        if not isinstance(data["phases"], list):
            raise ValueError("'phases' must be a list")
        
        if len(data["phases"]) == 0:
            raise ValueError("'phases' cannot be empty")
        
        # Validate each phase
        for i, phase in enumerate(data["phases"]):
            if not isinstance(phase, dict):
                raise ValueError(f"Phase {i} must be a dictionary")
            
            required_fields = ["title", "description", "estimated_hours", "topics"]
            for field in required_fields:
                if field not in phase:
                    raise ValueError(f"Phase {i} missing field: '{field}'")
            
            if not isinstance(phase["topics"], list):
                raise ValueError(f"Phase {i} 'topics' must be a list")
            
            if len(phase["topics"]) == 0:
                raise ValueError(f"Phase {i} 'topics' cannot be empty")
        
        logger.debug(f"✅ Validated {len(data['phases'])} phases")
    
    @property
    def _type(self) -> str:
        return "roadmap_json"


# Convenience functions
def create_query_parser(pydantic_object: type[BaseModel]) -> QueryOutputParser:
    """Create a query output parser."""
    return QueryOutputParser(pydantic_object=pydantic_object)


def parse_roadmap_json(text: str) -> Dict[str, Any]:
    """Parse roadmap JSON from LLM output."""
    parser = RoadmapOutputParser()
    return parser.parse(text)
