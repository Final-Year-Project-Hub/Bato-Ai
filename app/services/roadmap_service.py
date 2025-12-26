import json
from typing import List, Union
from app.schemas import RoadmapRequest, Roadmap, QueryIntent, ChatRequest, ClarificationRequest
from app.core.llm import llm_client
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.retrieval.query_analyzer import QueryAnalyzer, TokenBudget

class RoadmapService:
    def __init__(self, retriever: QdrantRetriever, analyzer: QueryAnalyzer):
        self.retriever = retriever
        self.analyzer = analyzer

    async def process_chat(self, request: ChatRequest) -> Union[Roadmap, ClarificationRequest]:
        # 1. Analyze User Query
        extracted = await self.analyzer.analyze(request.message)
        
        # 2. Check for critical missing info
        if extracted.missing_fields:
            # Construct a single, professional clarification request
            questions = []
            if "proficiency" in extracted.missing_fields:
                questions.append("Your current proficiency level (Beginner/Intermediate/Expert)")
            if "intent" in extracted.missing_fields:
                questions.append("Whether you want to LEARN concepts or BUILD a specific project")
            # We can implicitly ask for tech stack if it wasn't clear, but let's stick to the critical ones for now
            
            question_str = ", ".join(questions)
            final_message = f"To generate a tailored production-ready roadmap, I need to know: {question_str}."
            
            return ClarificationRequest(
                message=final_message,
                missing_fields=extracted.missing_fields
            )
            
        # 3. If complete, generate roadmap
        # Create a synthetic RoadmapRequest
        roadmap_req = RoadmapRequest(
            goal=extracted.goal or request.message,
            intent=extracted.intent,
            proficiency=extracted.proficiency or "beginner" # Fallback if analyzer missed it but didn't flag
        )
        
        return await self.generate_roadmap(roadmap_req)

    async def generate_roadmap(self, request: RoadmapRequest) -> Roadmap:
        # 1. Analyze Intent (using simple analyzer for now, can be upgraded to LLM)
        # We already have intent in request, but might want to refine tokens
        intent_type = request.intent
        # map schema intent to analyzer intent (if they differ, but they are same enum)
        
        # 2. Retrieve Documents
        # Construct query from goal + proficiency
        # E.g. "Build login page Next.js beginner"
        search_query = f"{request.intent.value} {request.goal} {request.proficiency}"
        
        # Retrieve context
        # We need to update QdrantRetriever to return objects we can use
        # Currently it returns list of strings (docs)
        docs = self.retriever.retrieve(
            query=search_query, 
            intent=intent_type, 
            token_budget=2000 # Increase budget for generation
        )
        
        context_text = "\n\n".join(docs)
        print(f"📚 RAG: Retrieved {len(docs)} documents for context.")
        
        # 3. Generate Roadmap via LLM
        system_prompt = """You are a Senior Principal Engineer and Technical Mentor.
        Your goal is to create a comprehensive, PRODUCTION-READY learning roadmap.
        
        Do NOT generate generic "Hello World" tutorials.
        Focus on:
        - Scalability, Security, Performance, and Maintainability.
        - Industry standard patterns (e.g., specific State Management libraries, Testing strategies, CI/CD).
        - "Why" it matters, not just "What" to learn.
        
        Context is provided from the official documentation. Use it to recommend the LATEST features.
        
        Output strict JSON matching this schema:
        {
            "phases": [
                {
                    "title": "Phase 1: Foundations & Best Practices",
                    "estimated_hours": 10.0,
                    "topics": [
                        {
                            "title": "Topic Title",
                            "description": "In-depth explanation...",
                            "why_it_matters": "Critical for minimizing technical debt...",
                            "key_concepts": ["Concept A", "Concept B"],
                            "doc_sources": ["url/path"],
                            "best_practices": ["Actionable tip 1"],
                            "estimated_hours": 3.0
                        }
                    ]
                }
            ],
            "total_estimated_hours": 40.0
        }
        
        Important:
        - **OUTPUT MUST BE RAW JSON ONLY.** No markdown blocks.
        - Ensure `why_it_matters` provides professional context.
        - `key_concepts` should list 3-5 specific technical terms/patterns.
        """
        
        user_prompt = f"""
        Goal: {request.goal}
        Intent: {request.intent.value}
        Proficiency: {request.proficiency}
        
        Context Recommendation:
        {context_text}
        
        Generate the roadmap JSON.
        """
        
        response_text = await llm_client.generate(user_prompt, system_prompt)
        
        # 4. Parse Response
        try:
            # Clean possible markdown code blocks
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            roadmap_data = json.loads(clean_json)
            return Roadmap(**roadmap_data)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Response: {response_text}")
            # Return empty/error roadmap or raise
            raise ValueError("Failed to generate valid roadmap structure")

