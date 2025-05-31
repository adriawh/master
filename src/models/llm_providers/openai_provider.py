import os
import openai
from models.llm_providers.base_llm import BaseLLM
from utils.logger import setup_logger
from typing import List, Dict, Optional
from enum import Enum

logger = setup_logger()

class ReasoningLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class OpenAIProvider(BaseLLM):
    """
    Uses OpenAI's API to generate responses.
    """

    def __init__(self, model_name: str, temperature: Optional[float] = 0.7, reasoning_effort: Optional[ReasoningLevel] = None):
        """
        Initialize the OpenAIProvider.

        :param model_name: Name of the model to use.
        :param temperature: Sampling temperature for regular models.
        :param reasoning_effort: Reasoning level for reasoning models (low, medium, high).
        """
        super().__init__(model_name, temperature)  # Pass model_name and temperature to BaseLLM
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ OpenAI API key is missing. Set OPENAI_API_KEY environment variable.")

        self.client = openai.OpenAI(api_key=self.api_key)  # Use a persistent client
        self.reasoning_effort = reasoning_effort

        if self.reasoning_effort and not isinstance(self.reasoning_effort, ReasoningLevel):
            raise ValueError(f"❌ Invalid reasoning level: {self.reasoning_effort}. Must be one of {list(ReasoningLevel)}.")

    @classmethod
    def reasoning(cls, model_name: str, reasoning_effort: ReasoningLevel = ReasoningLevel.MEDIUM):
        """
        Create an OpenAIProvider instance for reasoning models.

        :param model_name: Name of the reasoning model.
        :param reasoning_effort: Reasoning level (default is MEDIUM).
        """
        return cls(model_name=model_name, temperature=None, reasoning_effort=reasoning_effort)

    @classmethod
    def regular(cls, model_name: str, temperature: float = 0.7):
        """
        Create an OpenAIProvider instance for regular models.

        :param model_name: Name of the regular model.
        :param temperature: Sampling temperature (default is 0.7).
        """
        return cls(model_name=model_name, temperature=temperature, reasoning_effort=None)

    def generate_response(self, system_instruction: str, prompt: str) -> tuple[str, int, int]:
        self.log_request("OpenAI")

        try:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]

            if self.reasoning_effort:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort.value
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature
                )

            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return response.choices[0].message.content.strip(), input_tokens, output_tokens

        except openai.OpenAIError as e:
            return self.handle_api_error("OpenAI", e), 0, 0
        except Exception as e:
            return self.handle_api_error("OpenAI", e), 0, 0
