import os
import time
import json
import datetime
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

from models.llm_providers.base_llm import BaseLLM
from models.llm_providers.gemini_provider import GeminiProvider
from models.llm_providers.deepseek_provider import DeepSeekProvider
from models.llm_providers.grok_provider import GrokProvider
from models.llm_providers.anthropic_provider import AnthropicProvider
from models.llm_providers.openai_provider import OpenAIProvider, ReasoningLevel
from models.data_models import LLMResponse, Ticket, Persona, TestResultPerModel
from utils.logger import setup_logger
from utils.load_json import load_json
from prompt_provider import PromptProvider

logger = setup_logger()
load_dotenv()

@dataclass
class ModelTestConfig:
    model: BaseLLM
    include_persona: bool = True
    include_examples: bool = False
    include_cot_examples: bool = False
    include_cot_output: bool = False
    trials_per_persona: int = 10

class Config:
    TEMPERATURE = 0.7
    REASONING_EFFORT = ReasoningLevel.MEDIUM
    PERSONAS_FILE = "data/personas.json"
    NUM_TRIALS_PER_PERSONA = 10
    RUN_TESTS_IN_NORWEGIAN = True


    LLM_MODELS: List[ModelTestConfig] = [
        ModelTestConfig(
            model=GrokProvider("grok-3-beta", temperature=0.7),
            include_persona=True,
            include_examples=False,
            include_cot_examples=True,
            include_cot_output=True,
            trials_per_persona=NUM_TRIALS_PER_PERSONA
        ),
        ModelTestConfig(
            model=GrokProvider("grok-3-mini-beta", temperature=0.7),
            include_persona=False,
            include_examples=False,
            include_cot_examples=True,
            include_cot_output=True,
            trials_per_persona=NUM_TRIALS_PER_PERSONA
        ),
          
    ]



class LLMTestRunner:
    def __init__(self) -> None:
        self.base_path = os.path.dirname(__file__)
        self.personas: List[Persona] = [
            Persona(**persona) for persona in load_json(self.base_path, Config.PERSONAS_FILE)
        ]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(self.base_path, "results", f"llm_test_results_run_{timestamp}.jsonl")
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        with open(self.results_file, "w", encoding="utf-8") as f:
            # Save the configuration
            f.write(json.dumps({"config": {
                "num_trials_per_persona": Config.NUM_TRIALS_PER_PERSONA,
                "llm_models": [model.model.model_name for model in Config.LLM_MODELS],
                "temperature": Config.TEMPERATURE,
                "reasoning_effort": Config.REASONING_EFFORT.name,
                "personas_file": Config.PERSONAS_FILE,
                "run_tests_in_norwegian": Config.RUN_TESTS_IN_NORWEGIAN,
            }}) + "\n")

    def run_tests(self) -> None:
        for model_cfg in Config.LLM_MODELS:
            model = model_cfg.model

            logger.info(f"🚀 Running tests for {model.model_name}...")

            system_prompt = PromptProvider(os.path.join(self.base_path, "data")).get_prompt(
                include_persona=model_cfg.include_persona,
                include_examples=model_cfg.include_examples,
                include_cot_examples=model_cfg.include_cot_examples,
                include_cot_output=model_cfg.include_cot_output
            )

            for persona in self.personas:
                for trial in range(model_cfg.trials_per_persona):
                    try:
                        self.run_trial(model_cfg, model, persona, persona.expected_ticket, trial, system_prompt)
                    except Exception as e:
                        logger.error(f"🔥 Error in a test trial: {e}")

    def run_trial(self, model_cfg: ModelTestConfig, model: BaseLLM, persona: Persona, expected_ticket: Ticket, trial: int, system_prompt: str) -> None:
        if Config.RUN_TESTS_IN_NORWEGIAN:
            prompt = f"{persona.long_prompt_no} (Dato: {persona.date})"
        else:
            prompt = f"{persona.long_prompt} (Date: {persona.date})"
        start = time.time()
        response = self.get_llm_response(model, system_prompt, prompt)
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
            recommended_ticket = response.recommended_ticket
            explanation = response.explanation
            status = response.status or "unknown"
            question = response.question
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            is_correct = response.recommended_ticket == expected_ticket
            cot_output = response.cot_output


        trial_data = TestResultPerModel(
            persona_id=persona.persona_id,
            expected_ticket=expected_ticket.model_dump(),
            recommended_ticket=recommended_ticket.model_dump() if recommended_ticket else None,
            question=question,
            is_correct=is_correct,
            time_used=duration,
            cot_output = cot_output,
            status=status,
            timestamp=datetime.datetime.now().isoformat(),
            model=model.model_name,
            temperature=getattr(model, "temperature", None),
            reasoning_effort=getattr(model, "reasoning_effort", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            include_persona=model_cfg.include_persona,
            include_examples=model_cfg.include_examples,
            include_cot_examples=model_cfg.include_cot_examples,
            include_cot_output=model_cfg.include_cot_output,
            explanation=explanation,
            raw=raw 
        )

        self.save_result_to_file(trial_data)

        logger.info(
            f"[{model.model_name}] Trial {trial+1} - Persona {persona.persona_id} - "
            f"Expected: {expected_ticket.name} - Got: {recommended_ticket.name if recommended_ticket else 'None'} - "
            f"Correct: {is_correct} - Time: {duration:.2f}s - "
            f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}"
        )

    def get_llm_response(self, model: BaseLLM, system_prompt: str, prompt: str) -> Optional[LLMResponse]:
        try:
            raw, input_tokens, output_tokens = model.generate_response(system_prompt, prompt)
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

    def save_result_to_file(self, result: TestResultPerModel) -> None:
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump()) + "\n")
        logger.debug(f"✅ Saved result for persona {result.persona_id}")

    def clean_json_block(self, text: str) -> str:
        return text.replace("```json", "").replace("```", "").strip()

if __name__ == "__main__":
    tester = LLMTestRunner()
    tester.run_tests()
