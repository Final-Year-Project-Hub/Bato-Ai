from app.core.llm import llm_client
from app.schemas.document import QueryIntent
from dataclasses import dataclass

@dataclass
class TokenBudget:
    max_tokens: int

@dataclass
class ExtractedRequest:
    goal: str | None
    intent: QueryIntent | None
    proficiency: str | None
    tech_stack: str | None
    missing_fields: list[str]

class QueryAnalyzer:
    """
    Uses LLM to understand user intent and extract slots.
    """

    async def analyze(self, query: str) -> ExtractedRequest:
        system_prompt = """
        You are an intelligent intent classifier for a roadmap generator.
        Analyze the User's Input and extract the following fields into JSON format:
        - "goal": What they want to do (e.g., "Build a portfolio", "Learn Routing").
        - "intent": "build" (if creating something) or "learn" (if studying concepts).
        - "proficiency": "beginner", "intermediate", or "expert".
        - "tech_stack": Explicit technologies mentioned (e.g., "Next.js", "React").
        
        If a field is NOT mentioned, set it to null.
        
        Example Input: "I want to create a blog using nextjs"
        Output: {"goal": "create a blog", "intent": "build", "proficiency": null, "tech_stack": "Next.js"}
        
        Example Input: "I want to learn React. I am completely new to this."
        Output: {"goal": "learn React", "intent": "learn", "proficiency": "beginner", "tech_stack": "React"}
        """
        
        response_text = await llm_client.generate(query, system_prompt)
        print(f"🔍 DEBUG ANALYZER RAW: {response_text}")
        
        try:
            # Clean generic json markers
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            
            # Robust extraction: Find first { and last }
            start_idx = clean_json.find("{")
            end_idx = clean_json.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                clean_json = clean_json[start_idx:end_idx+1]
            
            import json
            data = json.loads(clean_json)
            
            # Determine missing fields
            missing = []
            if not data.get("proficiency"):
                missing.append("proficiency")
            
            intent_str = str(data.get("intent", "")).lower()
            if not intent_str or intent_str not in ["build", "learn"]:
                missing.append("intent")
            
            return ExtractedRequest(
                goal=data.get("goal"),
                intent=QueryIntent(intent_str) if intent_str in ["build", "learn"] else None,
                proficiency=data.get("proficiency"),
                tech_stack=data.get("tech_stack"),
                missing_fields=missing
            )
        except Exception as e:
            print(f"Analyzer Error: {e}")
            # Fallback
            return ExtractedRequest(goal=query, intent=QueryIntent.LEARN, proficiency=None, tech_stack=None, missing_fields=["proficiency", "intent"])

    def get_token_budget(self, intent: QueryIntent) -> TokenBudget:
        if intent == QueryIntent.BUILD:
            return TokenBudget(max_tokens=2000)
        return TokenBudget(max_tokens=1000)
