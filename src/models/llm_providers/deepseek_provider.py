import os
from openai import OpenAI
from models.llm_providers.base_llm import BaseLLM
from utils.logger import setup_logger

logger = setup_logger()

class DeepSeekProvider(BaseLLM):

    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ DeepSeek API key is missing. Set DEEPSEEK_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate_response(self, system_instruction: str, prompt: str) -> tuple[str, int, int]:
        self.log_request("DeepSeek")

        try:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return response.choices[0].message.content.strip(), input_tokens, output_tokens

        except Exception as e:
            return self.handle_api_error("DeepSeek", e), 0, 0
