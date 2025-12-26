from openai import AsyncOpenAI
from typing import Optional
from huggingface_hub import AsyncInferenceClient
from app.core.config import settings

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        
        if self.provider == "huggingface":
            self.client = AsyncInferenceClient(
                token=settings.HUGGINGFACE_API_TOKEN,
                model=settings.HF_MODEL_ID
            )
            print(f"🔹 LLM Client Initialized: Hugging Face ({settings.HF_MODEL_ID})")
        else:
            # Default to OpenAI compatible (DeepSeek, Ollama, etc)
            self.client = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY or "dummy-key",
                base_url="https://api.deepseek.com/v1" # Can be overridden in env for local models
            )
            self.model = settings.MODEL_NAME
            print(f"🔹 LLM Client Initialized: OpenAI Compatible ({settings.MODEL_NAME})")

    async def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        try:
            if self.provider == "huggingface":
                return await self._generate_hf(prompt, system_prompt)
            else:
                return await self._generate_openai(prompt, system_prompt)
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return ""

    async def _generate_openai(self, prompt: str, system_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.7
        )
        return response.choices[0].message.content

    async def _generate_hf(self, prompt: str, system_prompt: str) -> str:
        # Construct a prompt that includes system instructions if model supports it
        # For simple chat models, we combine them
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        # Check if model supports chat_completion (newer models usually do)
        # Using text-generation for broader compatibility or chat_completion if available
        # InferenceClient.chat_completion is preferred for chat models
        try:
            response = await self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )
            return response.choices[0].message.content
        except AttributeError:
             # Fallback for older library versions or models
            response = await self.client.text_generation(
                prompt=full_prompt,
                max_new_tokens=2048,
                temperature=0.7
            )
            return response

llm_client = LLMClient()
