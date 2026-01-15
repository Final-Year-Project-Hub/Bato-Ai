"""
Custom exceptions for Bato-AI.

Provides specific exception types for better error handling and debugging.
"""


class BatoAIException(Exception):
    """Base exception for all Bato-AI errors."""
    pass


class InsufficientDocumentationError(BatoAIException):
    """Raised when insufficient documentation is found for a query."""
    
    def __init__(self, message: str, tech_stack: list = None, docs_found: int = 0, min_required: int = 3):
        self.tech_stack = tech_stack or []
        self.docs_found = docs_found
        self.min_required = min_required
        super().__init__(message)


class JSONParseError(BatoAIException):
    """Raised when JSON parsing fails."""
    
    def __init__(self, message: str, raw_text: str = None):
        self.raw_text = raw_text
        super().__init__(message)


class PromptTemplateError(BatoAIException):
    """Raised when prompt template loading or formatting fails."""
    pass


class QueryAnalysisError(BatoAIException):
    """Raised when query analysis fails."""
    pass


class RetrievalError(BatoAIException):
    """Raised when document retrieval fails."""
    pass


class LLMError(BatoAIException):
    """Raised when LLM generation fails."""
    pass
