from abc import ABC, abstractmethod
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """
    Base Class for LLM providers.
    """

    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate_response(self, system_instruction: str, prompt: str) -> tuple[str, int, int]:
        """
        Generates a response based on the given system instruction and user prompt.
        Returns a tuple containing the response string, input tokens, and output tokens.
        This method must be implemented by subclasses.
        """
        pass

    def log_request(self, api_name: str) -> None:
        """
        Logs the API request details.
        """
        logger.info(
            f"🌍 Sending request to {api_name} API (Model: {self.model_name}, Temperature: {self.temperature})..."
        )

    def handle_api_error(self, api_name: str, error: Exception) -> str:
        """
        Handles API errors and logs them.
        """
        logger.error(f"❌ {api_name} API request failed: {error}")
        return f"Error: Failed to generate response from {api_name} API."
