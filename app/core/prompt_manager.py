"""
Prompt template manager for Bato-AI.

Loads and manages prompt templates from the prompts/ directory.
Supports variable substitution and hot-reloading.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


# Prompt manager module
class PromptManager:
    """
    Manages prompt templates with variable substitution.
    
    Features:
    - Load prompts from text files
    - Variable substitution with {{VARIABLE}} syntax
    - Caching for performance
    - Hot-reload capability
    
    Example:
        manager = PromptManager()
        prompt = manager.load_prompt(
            "query_analyzer",
            PROFICIENCY_LEVELS='"beginner", "intermediate", "expert"'
        )
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt template files
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
    
    def load_prompt(self, name: str, **kwargs) -> str:
        """
        Load and format a prompt template.
        
        Args:
            name: Prompt template name (without .txt extension)
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
            
        Example:
            prompt = manager.load_prompt(
                "query_analyzer",
                PROFICIENCY_LEVELS='"beginner", "intermediate"'
            )
        """
        # Load template (from cache or file)
        template = self._load_template(name)
        
        # Substitute variables
        formatted = self._substitute_variables(template, kwargs)
        
        return formatted
    
    def _load_template(self, name: str) -> str:
        """Load template from cache or file."""
        # Check cache
        if name in self._cache:
            return self._cache[name]
        
        # Load from file
        template_file = self.prompts_dir / f"{name}.txt"
        
        if not template_file.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_file}\n"
                f"Available templates: {self.list_templates()}"
            )
        
        try:
            template = template_file.read_text(encoding='utf-8')
            self._cache[name] = template
            logger.debug(f"Loaded prompt template: {name}")
            return template
        
        except Exception as e:
            raise ValueError(f"Error loading template {name}: {e}")
    
    def _substitute_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in template.
        
        Variables use {{VARIABLE_NAME}} syntax.
        """
        result = template
        
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        
        # Check for unsubstituted variables
        import re
        unsubstituted = re.findall(r'\{\{(\w+)\}\}', result)
        if unsubstituted:
            logger.warning(f"Unsubstituted variables in template: {unsubstituted}")
        
        return result
    
    def reload(self, name: Optional[str] = None):
        """
        Reload templates from disk.
        
        Args:
            name: Specific template to reload, or None to reload all
        """
        if name:
            self._cache.pop(name, None)
            logger.info(f"Reloaded template: {name}")
        else:
            self._cache.clear()
            logger.info("Reloaded all templates")
    
    def list_templates(self) -> list[str]:
        """List available prompt templates."""
        if not self.prompts_dir.exists():
            return []
        
        return [
            f.stem for f in self.prompts_dir.glob("*.txt")
        ]
    
    def get_template_path(self, name: str) -> Path:
        """Get full path to a template file."""
        return self.prompts_dir / f"{name}.txt"


# Global instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get global PromptManager instance (singleton)."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


# Convenience functions
def load_prompt(name: str, **kwargs) -> str:
    """Load a prompt template (convenience function)."""
    return get_prompt_manager().load_prompt(name, **kwargs)


def reload_prompts():
    """Reload all prompts from disk."""
    get_prompt_manager().reload()


if __name__ == "__main__":
    """Test the prompt manager."""
    
    manager = PromptManager()
    
    print("Available templates:")
    for template in manager.list_templates():
        print(f"  - {template}")
    
    # Test loading with variables
    if "query_analyzer" in manager.list_templates():
        prompt = manager.load_prompt(
            "query_analyzer",
            PROFICIENCY_LEVELS='"beginner", "intermediate", "expert"',
            INTENTS='"learn", "build"',
            DEPTHS='"conceptual", "practical", "balanced"'
        )
        print(f"\nLoaded query_analyzer prompt ({len(prompt)} chars)")
        print(f"First 200 chars: {prompt[:200]}...")
