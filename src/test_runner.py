import os
import time
import json
import datetime
from typing import List, Optional

from dotenv import load_dotenv

from models.llm_providers.base_llm import BaseLLM
from models.llm_providers.openai_provider import OpenAIProvider, ReasoningLevel
from models.llm_providers.anthropic_provider import AnthropicProvider
from models.llm_providers.deepseek_provider import DeepSeekProvider
from models.llm_providers.gemini_provider import GeminiProvider
from models.llm_providers.grok_provider import GrokProvider

from models.data_models import LLMResponse, TestResult, Ticket, Persona
from utils.logger import setup_logger
from utils.load_json import load_json
from prompt_provider import PromptProvider

logger = setup_logger()
load_dotenv()


class Config:
    NUM_TRIALS_PER_PERSONA = 10
    TEMPERATURE = 0.7
    REASONING_EFFORT = ReasoningLevel.MEDIUM
    PERSONAS_FILE = "data/personas.json"
    INCLUDE_PERSONA = False
    INCLUDE_EXAMPLES = False
    INCLUDE_COT_EXAMPLES = True
    INCLUDE_COT_OUTPUT = True   
    LLM_MODELS: List[BaseLLM] = [
        #OpenAIProvider.regular("gpt-4o-mini", temperature=TEMPERATURE),
        #OpenAIProvider.regular("gpt-4o-2024-11-20", temperature=TEMPERATURE),
        #OpenAIProvider.regular("gpt-4.1-2025-04-14", temperature=TEMPERATURE),
        #OpenAIProvider.regular("gpt-4.1-mini-2025-04-14", temperature=TEMPERATURE),
        #OpenAIProvider.regular("gpt-4.1-nano-2025-04-14", temperature=TEMPERATURE),
        #OpenAIProvider.reasoning("o3-mini", reasoning_effort=REASONING_EFFORT), # o3-mini-2025-01-31
        #OpenAIProvider.reasoning("o3", reasoning_effort=REASONING_EFFORT),
        #OpenAIProvider.reasoning("o4-mini-2025-04-16", reasoning_effort=REASONING_EFFORT),

        #AnthropicProvider("claude-3-7-sonnet-20250219", temperature=TEMPERATURE),
        AnthropicProvider("claude-3-5-haiku-20241022", temperature=TEMPERATURE),       

        GrokProvider("grok-3-beta", temperature=TEMPERATURE),
        GrokProvider("grok-3-mini-beta", temperature=TEMPERATURE),
        #GrokProvider("grok-2-1212", temperature=TEMPERATURE),


        GeminiProvider("gemini-2.0-flash", temperature=TEMPERATURE),
        GeminiProvider("gemini-2.0-flash-lite", temperature=TEMPERATURE),
        GeminiProvider("gemini-2.5-pro-preview-03-25", temperature=TEMPERATURE),
        
        DeepSeekProvider("deepseek-chat", temperature=TEMPERATURE),
        DeepSeekProvider("deepseek-reasoner", temperature=TEMPERATURE),
    ]


class LLMTestRunner:
    def __init__(self) -> None:
        self.base_path = os.path.dirname(__file__)
        self.personas: List[Persona] = [
            Persona(**persona) for persona in load_json(self.base_path, Config.PERSONAS_FILE)
        ]

        self.prompt_provider = PromptProvider(os.path.join(self.base_path, "data"))
        self.system_prompt = self.prompt_provider.get_prompt(
            include_persona=Config.INCLUDE_PERSONA,
            include_examples=Config.INCLUDE_EXAMPLES,
            include_cot_examples=Config.INCLUDE_COT_EXAMPLES, 
            include_cot_output=Config.INCLUDE_COT_OUTPUT,     
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = os.path.join(self.base_path, "results", f"llm_test_results_run_{timestamp}.jsonl")

        with open(self.results_path, "w", encoding="utf-8") as f:
            # Save the configuration
            f.write(json.dumps({"config": {
                "num_trials_per_persona": Config.NUM_TRIALS_PER_PERSONA,
                "temperature": Config.TEMPERATURE,
                "reasoning_effort": Config.REASONING_EFFORT.name,
                "personas_file": Config.PERSONAS_FILE,
                "llm_models": [model.model_name for model in Config.LLM_MODELS],
                "include_persona": Config.INCLUDE_PERSONA,
                "include_examples": Config.INCLUDE_EXAMPLES,
                "include_examples_with_cot": Config.INCLUDE_COT_EXAMPLES,
            }}) + "\n")
            # Save the system prompt
            f.write(json.dumps({"system_prompt": self.system_prompt}) + "\n")

    def run_tests(self) -> None:
        for model in Config.LLM_MODELS:
            logger.info(f"🚀 Running tests for {model.model_name}...")
            for persona in self.personas:
                for trial in range(Config.NUM_TRIALS_PER_PERSONA):
                    try:
                        self.run_trial(model, persona, persona.expected_ticket, trial)
                    except Exception as e:
                        logger.error(f"🔥 Error in a test trial: {e}")

    def run_trial(self, model: BaseLLM, persona: Persona, expected_ticket: Ticket, trial: int) -> None:
        try:
            prompt = f"{persona.long_prompt} (Date: {persona.date})"
            start = time.time()
            response = self.get_llm_response(model, prompt)
            duration = time.time() - start

            is_correct = False
            recommended_ticket = None
            explanation = None
            question = None
            raw = None
            status = "error"
            input_tokens = 0
            output_tokens = 0
            cot_output = None

            if response:
                raw = response.raw
                status = response.status or "unknown"
                recommended_ticket = response.recommended_ticket
                explanation = response.explanation
                question = response.question
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                is_correct = response.recommended_ticket == expected_ticket
                cot_output = response.cot_output

            reasoning_effort = model.reasoning_effort.value if hasattr(model, "reasoning_effort") and model.reasoning_effort else None

            result = TestResult(
                persona_id=persona.persona_id,
                expected_ticket=expected_ticket,
                status=status,
                cot_output=cot_output,
                recommended_ticket=recommended_ticket,
                question=question,
                is_correct=is_correct,
                timestamp=datetime.datetime.now().isoformat(),
                model=model.model_name,
                time_used=duration if response else None,
                temperature=model.temperature,
                reasoning_effort=reasoning_effort,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                explanation=explanation,
                raw=raw  
            )

            self.save_result_to_run_file(result)

            if response:
                logger.info(
                    f"[{model.model_name}] Trial {trial+1}/{Config.NUM_TRIALS_PER_PERSONA} - "
                    f"Persona {persona.persona_id} - Expected: {expected_ticket.name} - "
                    f"Got: {recommended_ticket.name if recommended_ticket else 'None'} - "
                    f"Correct: {is_correct} - Time: {duration:.2f}s - "
                    f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}"
                )
            else:
                logger.warning(
                    f"[{model.model_name}] Trial {trial+1} - Persona {persona.persona_id}: "
                    "No or incomplete response"
                )

        except Exception as e:
            logger.error(f"🔥 Error in trial {trial+1} for {persona.persona_id} with {model.model_name}: {e}")

    def get_llm_response(self, model: BaseLLM, prompt: str) -> Optional[LLMResponse]:
        try:
            raw, input_tokens, output_tokens = model.generate_response(self.system_prompt, prompt)
            clean = self.clean_json_block(raw)
            parsed = json.loads(clean)
            return LLMResponse(
                raw=raw,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                **parsed
            )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"🚨 JSON decoding error: {e}")
            return LLMResponse(
                raw=raw or "",
                status="parsing_error",
                cot_output=None,
                recommended_ticket=None,
                explanation=None,
                question=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            logger.error(f"🚨 LLM response error: {e}")
            return None

    def save_result_to_run_file(self, result: TestResult) -> None:
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump()) + "\n")
        logger.debug(f"✅ Saved result for persona {result.persona_id}")

    def clean_json_block(self, text: str) -> str:
        return text.replace("```json", "").replace("```", "").strip()


if __name__ == "__main__":
    tester = LLMTestRunner()
    tester.run_tests()
