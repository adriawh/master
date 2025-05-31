import os
from openai import OpenAI
from models.llm_providers.base_llm import BaseLLM
from utils.logger import setup_logger

logger = setup_logger()

class GrokProvider(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.api_key = os.getenv("XAI_API_KEY")

        if not self.api_key:
            raise ValueError("❌ xAI API key is missing. Set XAI_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

    def generate_response(self, system_instruction: str, prompt: str) -> tuple[str, int, int]:
        self.log_request("Grok")

        try:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return response.choices[0].message.content.strip(), input_tokens, output_tokens

        except Exception as e:
            return self.handle_api_error("Grok", e), 0, 0
