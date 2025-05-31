import os
from google import genai
from google.genai import types
from models.llm_providers.base_llm import BaseLLM
from utils.logger import setup_logger

logger = setup_logger()

class GeminiProvider(BaseLLM):

    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("❌ Google API key is missing. Set the GEMINI_API_KEY environment variable.")

        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, system_instruction: str, prompt: str) -> tuple[str, int, int]:
        self.log_request("Gemini")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    system_instruction=system_instruction
                )
            )

            usage_metadata = response.usage_metadata
            input_tokens = usage_metadata.prompt_token_count if usage_metadata else 0
            output_tokens = usage_metadata.candidates_token_count if usage_metadata else 0

            return response.text.strip(), input_tokens, output_tokens

        except Exception as e:
            return self.handle_api_error("Gemini", e), 0, 0
