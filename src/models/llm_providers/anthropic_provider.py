import os
import anthropic
from models.llm_providers.base_llm import BaseLLM
from utils.logger import setup_logger

logger = setup_logger()

class AnthropicProvider(BaseLLM):
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ Anthropic API key is missing. Set ANTHROPIC_API_KEY environment variable.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_response(self, system_instruction: str, prompt: str) -> tuple[str, int, int]:
        self.log_request("Anthropic")

        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=self.temperature,
                system=system_instruction,
                messages=messages
            )
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            return response.content[0].text.strip(), input_tokens, output_tokens

        except anthropic.APIError as e:
            return self.handle_api_error("Anthropic", e), 0, 0
        except Exception as e:
            return self.handle_api_error("Anthropic", e), 0, 0
